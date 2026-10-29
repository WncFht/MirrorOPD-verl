# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Pinned model assets, vocabulary compatibility and buffer-preserving execution."""

import json
from concurrent.futures import ThreadPoolExecutor

from .artifacts import complete, file_hash, read, receipt, write
from .spec import WORK, fingerprint

FORWARD_FIELDS = (
    "model_type",
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "hidden_act",
    "rms_norm_eps",
    "tie_word_embeddings",
    "rope_theta",
    "rope_scaling",
    "max_position_embeddings",
    "use_sliding_window",
    "sliding_window",
)

PARAMETER_LAYOUT_FIELDS = (
    "model_type",
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "tie_word_embeddings",
)


def same_parameter_layout(student, teacher):
    return all(student[key] == teacher[key] for key in PARAMETER_LAYOUT_FIELDS)


def materialize(model):
    from huggingface_hub import HfApi, hf_hub_download

    target = model.directory
    inputs = dict(repo=model.repo, revision=model.revision)
    if complete(target, inputs):
        return read(target / "complete.json")
    info = HfApi().model_info(model.repo, revision=model.revision, files_metadata=True)
    assert info.sha == model.revision
    target.mkdir(parents=True, exist_ok=True)
    manifest = []
    for entry in info.siblings:
        name = entry.rfilename
        if "/" in name or not (
            name.endswith((".safetensors", ".json", ".jinja", ".txt", ".model", ".md")) or name == "LICENSE"
        ):
            continue
        if name.endswith(".safetensors"):
            expected = getattr(entry.lfs, "sha256", None)
            candidate = WORK / "model" / model.repo.split("/")[-1] / name
            local = target / name
            if not local.exists() and expected and candidate.exists() and file_hash(candidate) == expected:
                local.symlink_to(candidate.resolve())
            if not local.exists():
                hf_hub_download(model.repo, name, revision=model.revision, local_dir=target)
            if expected and file_hash(local) != expected:
                raise ValueError(f"Weight digest mismatch: {local}")
            manifest.append(dict(name=name, bytes=local.stat().st_size, sha256=expected))
        else:
            hf_hub_download(model.repo, name, revision=model.revision, local_dir=target)
    if not manifest or not (target / "config.json").exists():
        raise ValueError(f"A complete causal-LM checkpoint is required: {model.repo}")
    receipt(target, inputs, weights=manifest)
    print("ASSET", model.name, model.revision, flush=True)
    return read(target / "complete.json")


def assets(study, workers=3):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        values = list(pool.map(materialize, study.models.values()))
    write(study.directory / "assets.json", dict(zip(study.models, values, strict=True)))


def interface(model):
    from transformers import AutoConfig, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model.directory, local_files_only=True, trust_remote_code=False)
    config = AutoConfig.from_pretrained(model.directory, local_files_only=True, trust_remote_code=False)
    vocab = {index: token for token, index in tokenizer.get_vocab().items()}
    forward = {key: getattr(config, key, None) for key in FORWARD_FIELDS}
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    tokenization = {
        key: backend.get(key) for key in ["normalizer", "pre_tokenizer", "post_processor", "decoder", "model"]
    }
    return dict(
        model=model.identity,
        forward=forward,
        forward_hash=fingerprint(forward),
        vocab=vocab,
        vocabulary_hash=fingerprint(vocab),
        tokenization_hash=fingerprint(tokenization),
        eos_id=tokenizer.eos_token_id,
        template=tokenizer.chat_template,
        template_hash=fingerprint(tokenizer.chat_template),
    )


def audit(study):
    interfaces = {name: interface(model) for name, model in study.models.items()}
    base = interfaces[study.student]
    result = {}
    for name, value in interfaces.items():
        conflicts = [
            (i, token, value["vocab"].get(i)) for i, token in base["vocab"].items() if value["vocab"].get(i) != token
        ]
        support = value["forward"]["vocab_size"] >= base["forward"]["vocab_size"]
        result[name] = dict(
            identity=value["model"],
            conflicts=conflicts,
            teacher_support_covers_student=support,
            full_vocab_eligible=not conflicts and support,
            same_forward_configuration=value["forward"] == base["forward"],
            configuration=value["forward"],
            vocabulary_hash=value["vocabulary_hash"],
            tokenization_hash=value["tokenization_hash"],
            eos_id=value["eos_id"],
            native_template_hash=value["template_hash"],
        )
        if value["eos_id"] not in study.evaluation.stop_ids:
            raise ValueError(f"Explicit common stop set omits native EOS for {name}")
    write(study.directory / "interfaces.json", result)
    for arm in study.arms:
        if not result[arm.teacher]["full_vocab_eligible"]:
            raise ValueError(f"Native full-vocabulary KL is invalid for {arm.teacher}; see interfaces.json")
    return result


def setup():
    import os

    import torch

    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(devices) != 1 or devices[0] not in {"4", "5", "6", "7"}:
        raise ValueError("GPU workers require exactly one explicitly assigned physical GPU4-7")

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load(model, *, trainable=False, path=None):
    import torch
    from transformers import AutoModelForCausalLM

    value = AutoModelForCausalLM.from_pretrained(
        path or model.directory,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float32 if trainable else torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": "cuda:0"},
    ).eval()
    output = value.get_output_embeddings()
    if not isinstance(output, torch.nn.Linear) or output.bias is not None:
        raise ValueError("This adapter requires an audited bias-free linear vocabulary projection")
    if trainable and any(p.dtype != torch.float32 for p in value.parameters()):
        raise ValueError("Trainable master parameters must all remain FP32")
    value.requires_grad_(trainable)
    for name, buffer in value.named_buffers():
        if "inv_freq" in name:
            assert buffer.dtype == torch.float32, (name, buffer.dtype)
    return value


def training_mode(model, execution):
    import torch

    if execution.activation_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.gradient_checkpointing_disable()
    model.get_decoder().train()
    for layer in model.modules():
        if isinstance(layer, torch.nn.Dropout):
            layer.eval()


def hidden(model, rows, *, pad_id):
    """Concatenate response-aligned states; singleton path matches the original engine."""
    import torch

    sequences = [r["prompt_ids"] + r["response_ids"][:-1] for r in rows]
    device = model.get_input_embeddings().weight.device
    if len(rows) == 1:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model.get_decoder()(
                input_ids=torch.tensor([sequences[0]], device=device), use_cache=False
            ).last_hidden_state
        return output[0, len(rows[0]["prompt_ids"]) - 1 :]
    width = max(map(len, sequences))
    ids = torch.tensor([[pad_id] * (width - len(s)) + s for s in sequences], device=device)
    mask = torch.tensor([[0] * (width - len(s)) + [1] * len(s) for s in sequences], device=device)
    positions = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model.get_decoder()(
            input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False
        ).last_hidden_state
    return torch.cat(
        [
            output[i, width - len(s) + len(r["prompt_ids"]) - 1 :]
            for i, (s, r) in enumerate(zip(sequences, rows, strict=True))
        ]
    )
