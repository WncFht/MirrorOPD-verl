# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Extend the fixed regression batch without modifying its original evidence."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def prepare(source, out, batches):
    if batches < 2:
        raise ValueError("Regression needs one complete warmup and at least one measured batch")
    table = pq.read_table(source / "prompts.parquet")
    if not table.num_rows or table.num_rows % 64:
        raise ValueError("The source must contain complete fixed batches of 64 prompts")
    first = table.slice(0, 64)
    if any(not table.slice(start, 64).equals(first) for start in range(0, table.num_rows, 64)):
        raise ValueError("Source batches differ; refusing to change the frozen regression workload")
    for name in ("baseline.json", "tokenizer_support.py"):
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    out.mkdir(parents=True, exist_ok=False)
    pq.write_table(pa.concat_tables([first] * batches), out / "prompts.parquet")
    for name in ("baseline.json", "tokenizer_support.py"):
        shutil.copyfile(source / name, out / name)
    manifest = {
        "source": str(source.resolve()),
        "source_sha256": hashlib.sha256((source / "prompts.parquet").read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256((out / "prompts.parquet").read_bytes()).hexdigest(),
        "source_rows": table.num_rows,
        "output_rows": batches * 64,
        "batch_size": 64,
        "warmup_batches": 1,
        "measured_batches": batches - 1,
        "scope": "Exact repetition of the original fixed Arrow rows; original evidence unchanged",
    }
    (out / "workload.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batches", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.out, args.batches)))


if __name__ == "__main__":
    main()
