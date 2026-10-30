# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Pin model inputs once and validate their content when manually resuming."""

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path


def checkpoint_identity(path):
    root = Path(path).resolve()
    files = sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix in {".json", ".safetensors", ".bin", ".model", ".txt", ".tiktoken"}
    )
    if not (root / "config.json").is_file() or not any(p.suffix in {".safetensors", ".bin"} for p in files):
        raise ValueError(f"Incomplete model checkpoint: {root}")
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


def bind_policy(config, policy):
    teachers = config.distillation.teacher_models
    teacher_path = next(iter(teachers.values())).model_path if len(teachers) == 1 else None
    bindings = policy.bindings(config.actor_rollout_ref.model.path, teacher_path)
    identities = {path: checkpoint_identity(path) for path in {b.path for b in bindings.values()}}
    bindings = {slot: replace(binding, identity=identities[binding.path]) for slot, binding in bindings.items()}
    state = {
        "schema": 1,
        "policy": asdict(policy),
        "bindings": {
            slot: {"kind": b.kind, "path": b.path, "sha256": identities[b.path]} for slot, b in bindings.items()
        },
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
    if previous != state:
        raise ValueError("Rollout policy or model identity changed on resume; keep original bindings")


def save_bindings(checkpoint_dir, state):
    path = Path(checkpoint_dir) / "rollout_state.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)
