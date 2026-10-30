# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Atomic stage-boundary optimizer/RNG state, tied to immutable model weights and science inputs."""

import random
from pathlib import Path

import numpy as np
import torch

from examples.mirroropd.research.core.artifacts import file_hash, fingerprint
from examples.mirroropd.research.core.io import dump, read


def parameter_names(model, optimizer):
    names = {id(p): name for name, p in model.named_parameters()}
    return [[names[id(p)] for p in group["params"]] for group in optimizer.param_groups]


def rng_state():
    array = np.random.get_state()
    return dict(
        python=random.getstate(),
        numpy=(array[0], array[1].tolist(), *array[2:]),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
    )


def restore_rng(state):
    random.setstate(state["python"])
    a = state["numpy"]
    np.random.set_state((a[0], np.asarray(a[1], dtype=np.uint32), *a[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume CUDA device count differs")
        torch.cuda.set_rng_state_all(state["cuda"])


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def save(directory, model, optimizer, *, step, signature, weights, weight_identity):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = dict(
        step=step,
        signature=fingerprint(signature),
        weights=str(weights),
        weight_identity=weight_identity,
        parameter_names=parameter_names(model, optimizer),
        optimizer=cpu_tree(optimizer.state_dict()),
        rng=rng_state(),
    )
    temporary = directory / "optimizer.pt.tmp"
    torch.save(payload, temporary)
    path = directory / "optimizer.pt"
    temporary.replace(path)
    dump(
        directory / "complete.json",
        dict(
            step=step,
            signature=payload["signature"],
            weights=str(weights),
            weight_identity=weight_identity,
            bytes=path.stat().st_size,
            sha256=file_hash(path),
        ),
    )


def restore(directory, model, optimizer, *, signature, weight_identity):
    directory = Path(directory)
    marker = read(directory / "complete.json")
    if marker["signature"] != fingerprint(signature) or marker["weight_identity"] != weight_identity:
        raise ValueError("Resume science or weight identity mismatch")
    if marker.get("optimizer_pruned"):
        raise ValueError("Resume optimizer was superseded; use the recorded replacement")
    path = directory / "optimizer.pt"
    if path.stat().st_size != marker["bytes"] or file_hash(path) != marker["sha256"]:
        raise ValueError("Resume state content changed")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if (value["signature"], value["weight_identity"], value["step"]) != (
        marker["signature"],
        marker["weight_identity"],
        marker["step"],
    ):
        raise ValueError("Resume payload and receipt disagree")
    if value["parameter_names"] != parameter_names(model, optimizer):
        raise ValueError("Resume optimizer parameter ordering changed")
    optimizer.load_state_dict(value["optimizer"])
    if {int(v["step"]) for v in optimizer.state.values()} != {value["step"]}:
        raise ValueError("Resume Adam steps differ")
    restore_rng(value["rng"])
    return value["step"]


def retire(directory, *, replacement):
    """Remove only an older optimizer payload after verifying a later compatible resume point."""
    directory, replacement = Path(directory).resolve(), Path(replacement).resolve()
    old, new = read(directory / "complete.json"), read(replacement / "complete.json")
    if directory == replacement or old["signature"] != new["signature"] or new["step"] <= old["step"]:
        raise ValueError("Replacement must be a later compatible resume point")
    current = replacement / "optimizer.pt"
    if current.stat().st_size != new["bytes"] or file_hash(current) != new["sha256"]:
        raise ValueError("Replacement optimizer failed integrity check")
    old.update(optimizer_pruned=True, superseded_by=str(replacement))
    dump(directory / "complete.json", old)
    (directory / "optimizer.pt").unlink(missing_ok=True)
