#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Create immutable per-benchmark validation copies with an explicit rollout n."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROLLOUTS = {"MATH-500": 4, "AMC23": 16, "AIME24": 16, "AIME25": 16, "AIME26": 16}


def prepare(output_dir: Path, inputs: list[Path]) -> dict:
    if len(inputs) != len(ROLLOUTS):
        raise ValueError(f"Expected {len(ROLLOUTS)} benchmark files, got {len(inputs)}")

    manifest = {"profile": "math4_rest16", "datasets": {}}
    seen = set()
    for source_path in inputs:
        table = pq.read_table(source_path)
        if "data_source" not in table.column_names:
            raise ValueError(f"Validation file lacks data_source: {source_path}")
        names = set(table.column("data_source").to_pylist())
        if len(names) != 1:
            raise ValueError(f"Expected one data_source in {source_path}, got {sorted(names)}")
        name = names.pop()
        if name not in ROLLOUTS or name in seen:
            raise ValueError(f"Unexpected or duplicate validation benchmark {name!r}")
        if "__rollout_n__" in table.column_names:
            raise ValueError(f"Refusing to replace existing __rollout_n__ in {source_path}")
        seen.add(name)

        rollout_n = ROLLOUTS[name]
        output_path = output_dir / name / source_path.name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        table = table.append_column("__rollout_n__", pa.array([rollout_n] * table.num_rows, type=pa.int64()))
        pq.write_table(table, output_path)
        manifest["datasets"][name] = {
            "source": str(source_path.resolve()),
            "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "output": str(output_path.resolve()),
            "rows": table.num_rows,
            "rollout_n": rollout_n,
        }

    if seen != set(ROLLOUTS):
        raise ValueError(f"Missing benchmarks: {sorted(set(ROLLOUTS) - seen)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()
    prepare(args.output_dir, args.inputs)


if __name__ == "__main__":
    main()
