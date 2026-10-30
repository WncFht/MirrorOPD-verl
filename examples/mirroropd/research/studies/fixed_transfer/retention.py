# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Remove only audited transient readout weights, retaining final weights and evidence."""

import time
from pathlib import Path

from examples.mirroropd.research.core.artifacts import file_hash, fingerprint
from examples.mirroropd.research.core.io import dump as write
from examples.mirroropd.research.core.io import read


def prune_weights(directory, *, apply=False):
    """Caller must audit all consumers first; validate every file before removing any."""
    directory = Path(directory)
    marker = directory / "complete.json"
    saved = read(marker)
    if saved.get("transient") is not True:
        raise ValueError(f"Refusing to prune final weights: {directory}")
    if saved.get("weights_pruned"):
        return 0
    if saved["fingerprint"] != fingerprint(saved["inputs"]):
        raise ValueError("Checkpoint receipt changed")
    files = saved["files"]
    names = [item["name"] for item in files]
    if not files or len(set(names)) != len(names):
        raise ValueError("Empty or duplicate weight manifest")
    for item in files:
        name = item["name"]
        if Path(name).name != name or not name.endswith(".safetensors"):
            raise ValueError(f"Unsafe weight filename: {name}")
        source = directory / name
        if source.is_symlink():
            raise ValueError(f"Refusing symlink: {source}")
        if source.exists():
            if source.stat().st_size != item["bytes"] or file_hash(source) != item["sha256"]:
                raise ValueError(f"Weight content changed: {source}")
        elif not saved.get("prune_started_at"):
            raise ValueError(f"Weight missing before prune: {source}")
    total = sum((directory / name).stat().st_size for name in names if (directory / name).exists())
    if apply:
        # Record intent before unlink: an interrupted cleanup can resume from the same manifest.
        saved.setdefault("prune_started_at", time.time())
        write(marker, saved)
        for name in names:
            (directory / name).unlink(missing_ok=True)
        saved.update(weights_pruned=True, pruned_at=time.time())
        write(marker, saved)
    return total


def prune(study, *, apply=False):
    from examples.mirroropd.research.studies.fixed_transfer.analysis import analyze
    from examples.mirroropd.research.studies.fixed_transfer.training import checkpoint_directory

    # Full validation requires all endpoints, paired task statistics, risks, and numerical gates.
    analyze(study, partial=False)
    targets = [
        checkpoint_directory(study, arm, step) for arm in study.arms for step in arm.evaluate_at if step != arm.updates
    ]
    planned = [dict(directory=str(path), bytes=prune_weights(path)) for path in targets]
    # Dry-run validation of every target precedes the first deletion.
    if apply:
        for path in targets:
            prune_weights(path, apply=True)
    result = dict(applied=apply, checkpoints=planned, bytes=sum(x["bytes"] for x in planned), time=time.time())
    write(study.directory / "analysis/retention.json", result)
    return result
