# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Content identities and atomic, verified signature shards."""

import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from examples.mirroropd.research.core.io import dump, read


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            checksum.update(block)
    return checksum.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def model_identity(path):
    """Bind config contents and weight shard metadata without rereading GB of weights."""
    path = Path(path).resolve()
    index = path / "model.safetensors.index.json"
    names = sorted(set(read(index)["weight_map"].values())) if index.exists() else ["model.safetensors"]
    return dict(
        path=str(path),
        config_sha256=digest(path / "config.json"),
        index_sha256=digest(index) if index.exists() else None,
        weights=[dict(name=n, bytes=(path / n).stat().st_size, mtime_ns=(path / n).stat().st_mtime_ns) for n in names],
    )


def save_arrays(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def cached(path, key):
    path = Path(path)
    receipt = path.with_suffix(".json")
    if not path.exists() or not receipt.exists():
        return False
    record = read(receipt)
    if record["key"] != key or record["sha256"] != digest(path):
        raise ValueError(f"Signature cache identity or content changed: {path}")
    return True


def publish(path, key, arrays, **metadata):
    save_arrays(path, **arrays)
    dump(Path(path).with_suffix(".json"), dict(key=key, sha256=digest(path), **metadata))
