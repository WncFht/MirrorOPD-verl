# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Strict JSON and atomic writes for research artifacts, without model imports."""

import csv
import gzip
import io
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile


def read(path: Path):
    return json.loads(Path(path).read_text())


def write_text(path, text):
    """Replace a complete UTF-8 file atomically; failed writes preserve its predecessor."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def dump(path, value, *, allow_nan=False):
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=allow_nan) + "\n")


def read_rows(path):
    """Read plain or gzipped JSONL; malformed complete records are errors."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_csv(path, rows):
    """Write measured rows with stable first-seen column ordering."""
    rows = list(rows)
    if not rows:
        raise ValueError("Do not publish an empty result table")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
    writer.writeheader()
    writer.writerows(rows)
    write_text(path, stream.getvalue())


def finite(value):
    """Check finite floating-point leaves in a nested JSON artifact."""
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for child in value.values():
            finite(child)
    elif isinstance(value, list):
        for child in value:
            finite(child)
