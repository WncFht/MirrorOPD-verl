# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Atomic artifacts with explicit input fingerprints and immutable completion receipts."""

import hashlib
import json
import time
from pathlib import Path

from examples.mirroropd.research.core.io import dump as write
from examples.mirroropd.research.core.io import read


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def receipt(directory, inputs, **result):
    write(
        Path(directory) / "complete.json",
        dict(fingerprint=fingerprint(inputs), inputs=inputs, completed_at=time.time(), **result),
    )


def complete(directory, inputs):
    path = Path(directory) / "complete.json"
    if not path.exists():
        return False
    stored = read(path)
    if stored["fingerprint"] != fingerprint(inputs):
        raise ValueError(f"Artifact identity mismatch: {directory}")
    return True


def records(path, *, repair=False):
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_bytes().splitlines(keepends=True)
    result = []
    offset = 0
    for index, line in enumerate(lines):
        try:
            if line.strip():
                result.append(json.loads(line))
        except (json.JSONDecodeError, UnicodeDecodeError):
            if not repair or index != len(lines) - 1 or line.endswith(b"\n"):
                raise
            backup = path.with_name(path.name + f".incomplete-{time.time_ns()}")
            backup.write_bytes(line)
            with path.open("r+b") as stream:
                stream.truncate(offset)
            write(path.with_name("append-repair.json"), dict(valid_records=len(result), backup=str(backup)))
            break
        offset += len(line)
    return result
