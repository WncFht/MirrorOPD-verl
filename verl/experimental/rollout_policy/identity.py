# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Pin model inputs once and validate their content when manually resuming."""

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

from .models import local_model_path, requires_token_generator, resolve_bindings


def checkpoint_files(root):
    """Manifest of supported HF weights and tokenizer inputs, excluding run artifacts."""
    root = Path(root).resolve()
    files = {root / "config.json"}
    for stem, index in (
        ("model.safetensors", "model.safetensors.index.json"),
        ("pytorch_model.bin", "pytorch_model.bin.index.json"),
    ):
        if (root / index).is_file():
            files.add(root / index)
            manifest = json.loads((root / index).read_text())
            for name in set(manifest["weight_map"].values()):
                relative = Path(name)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("Checkpoint shard must be inside its model directory")
                files.add(root / relative)
            break
        if (root / stem).is_file():
            files.add(root / stem)
            break
    else:
        raise ValueError(f"Incomplete model checkpoint: {root}")
    for name in (
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "vocab.txt",
        "merges.txt",
        "tokenizer.model",
        "spiece.model",
        "sentencepiece.bpe.model",
        "tokenizer.tiktoken",
        "chat_template.jinja",
        "stage-protocol.json",
    ):
        if (root / name).is_file():
            files.add(root / name)
    files.update((root / "chat_templates").glob("*.jinja"))
    # Explicitly trusted custom HF architectures are part of the model function.
    files.update(root.glob("*.py"))
    if not all(file.is_file() for file in files):
        raise ValueError(f"Incomplete model checkpoint: {root}")
    return sorted(files)


def checkpoint_identity(path):
    root = Path(path).resolve()
    files = checkpoint_files(root)
    digest = hashlib.sha256()
    for file in files:
        before = file.stat()
        content = hashlib.sha256()
        with file.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024**2), b""):
                content.update(block)
        after = file.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError(f"Checkpoint changed while binding: {file}")
        digest.update(json.dumps([str(file.relative_to(root)), content.hexdigest()]).encode())
    return digest.hexdigest()


def bind_policy(config, policy, *, resolve_path=None):
    teachers = config.get("distillation", {}).get("teacher_models", {})
    teacher_path = next(iter(teachers.values())).model_path if len(teachers) == 1 else None
    if resolve_path is None:
        resolver = config.get("rollout_model_resolver")
        if resolver:
            from hydra.utils import get_method

            resolve_path = get_method(resolver)
        else:
            resolve_path = local_model_path
    bindings = resolve_bindings(policy, config.actor_rollout_ref.model.path, teacher_path, resolve_path=resolve_path)
    identities = {path: checkpoint_identity(path) for path in {b.path for b in bindings.values()}}
    bindings = {slot: replace(binding, identity=identities[binding.path]) for slot, binding in bindings.items()}
    from omegaconf import OmegaConf

    def value(key, default=None):
        item = OmegaConf.select(config, key, default=default)
        return OmegaConf.to_container(item, resolve=True) if OmegaConf.is_config(item) else item

    state = {
        "schema": 2,
        "semantics": {
            "mode": policy.mode,
            "coefficient": asdict(policy.right_weight) if policy.mode != "single" else None,
            "fusion_space": policy.fusion_space if policy.mode == "token_mixture" else None,
            "bindings": {slot: {"kind": b.kind, "sha256": b.identity} for slot, b in bindings.items()},
            "sampling_contract": "mirroropd-crossing-token-v1",
            "backend": policy.execution.backend if requires_token_generator(policy, bindings) else "vllm",
            "sampling": {
                name: value("actor_rollout_ref.rollout." + name)
                for name in (
                    "temperature",
                    "top_p",
                    "top_k",
                    "response_length",
                    "prompt_length",
                    "stop_token_ids",
                    "ignore_eos",
                    "n",
                    "seed",
                    "repetition_penalty",
                    "full_determinism",
                    "dtype",
                )
            },
        },
        "execution": {
            "fusion": asdict(policy.execution),
            "micro_batch_size": policy.micro_batch_size,
            "deadline_seconds": policy.deadline_seconds,
            "drain_seconds": policy.drain_seconds,
        },
        "locations": {slot: b.path for slot, b in bindings.items()},
    }
    trainer = config.get("trainer", {})
    checkpoint = None
    if trainer.get("resume_mode") == "resume_path":
        checkpoint = Path(trainer["resume_from_path"])
    elif trainer.get("resume_mode") == "auto":
        root = Path(trainer["default_local_dir"])
        marker = root / "latest_checkpointed_iteration.txt"
        if marker.is_file():
            checkpoint = root / f"global_step_{int(marker.read_text().strip())}"
    if checkpoint is not None:
        verify_resume_bindings(checkpoint, state)
    return bindings, state


def verify_resume_bindings(checkpoint_dir, state):
    path = Path(checkpoint_dir) / "rollout_state.json"
    if not path.is_file():
        raise ValueError("Checkpoint lacks rollout_state.json; resume with its original frozen code or start a new run")
    previous = json.loads(path.read_text())
    if previous.get("schema") != 2 or state.get("schema") != 2:
        raise ValueError("Rollout state schema changed; resume with the original frozen code or start a new run")
    if previous["semantics"] != state["semantics"]:
        raise ValueError("Rollout policy or model identity changed on resume; keep original bindings")


def save_bindings(checkpoint_dir, state):
    path = Path(checkpoint_dir) / "rollout_state.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)
