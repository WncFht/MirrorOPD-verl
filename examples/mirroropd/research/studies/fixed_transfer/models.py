# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Pinned model assets, vocabulary compatibility and buffer-preserving execution."""

import json
from concurrent.futures import ThreadPoolExecutor

from examples.mirroropd.research.core.artifacts import complete, file_hash, receipt
from examples.mirroropd.research.core.io import dump as write
from examples.mirroropd.research.core.io import read
from examples.mirroropd.research.studies.fixed_transfer.spec import WORK, fingerprint

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
