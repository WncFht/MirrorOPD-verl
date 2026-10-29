#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Aggregate sampled system telemetry using timestamps and the testing timer.

Phase boundaries are approximate: a validation timer ends near its history row's
timestamp. Means are sample means, not continuous GPU utilization or FLOP counts.
GPU groups are physical indices, not an asserted Ray rank-to-device mapping.
"""

import argparse
import json
from statistics import mean, median

from examples.mirroropd.research.core.io import dump as write_json
from examples.mirroropd.research.core.io import read_rows
from examples.mirroropd.research.studies.pipeline.analyze_run import is_number


def aggregate(rows):
    out = {}
    for gpu in range(8):
        out[str(gpu)] = {}
        for metric in ("gpu", "memoryAllocatedBytes", "memoryAllocated", "powerWatts", "temp"):
            values = [r[f"system.gpu.{gpu}.{metric}"] for r in rows if is_number(r.get(f"system.gpu.{gpu}.{metric}"))]
            if values:
                out[str(gpu)][metric] = {
                    "mean": mean(values),
                    "max": max(values),
                    "min": min(values),
                    "samples": len(values),
                }
    return out


def main():
    from pathlib import Path

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    args = p.parse_args()
    history = read_rows(args.data / "history.jsonl")
    system_path = args.data / "system.jsonl.gz"
    if not system_path.is_file():
        system_path = args.data / "system-sampled.jsonl.gz"
    system = read_rows(system_path)
    source = json.loads((args.data / "source.json").read_text())
    gpu_rows = [r for r in system if is_number(r.get("system.gpu.0.gpu"))]
    times = sorted(r["_timestamp"] for r in gpu_rows)
    val = [
        (r["_timestamp"] - r["timing_s/testing"], r["_timestamp"])
        for r in history
        if is_number(r.get("timing_s/testing"))
    ]
    first, last = min(r["_timestamp"] for r in history), max(r["_timestamp"] for r in history)
    is_val = lambda t: any(a <= t <= b for a, b in val)  # noqa: E731
    groups = {
        "all_observed": gpu_rows,
        "validation_approx": [r for r in gpu_rows if is_val(r["_timestamp"])],
        "training_between_steps_1_and_last": [
            r for r in gpu_rows if first <= r["_timestamp"] <= last and not is_val(r["_timestamp"])
        ],
        "after_last_training_row": [r for r in gpu_rows if r["_timestamp"] > last],
    }
    result = {
        "method": (
            source.get("system_method", "system API sampling completeness unknown")
            + "; sample means; timestamp-derived phase boundaries approximate"
        ),
        "gpu_rows": len(gpu_rows),
        "stream_rows": len(system),
        "first_gpu_timestamp": times[0],
        "last_gpu_timestamp": times[-1],
        "median_gpu_sample_interval_seconds": median(b - a for a, b in zip(times, times[1:], strict=False)),
        "validation_timestamp_intervals": val,
        "phases": {k: {"sample_rows": len(v), "per_gpu": aggregate(v)} for k, v in groups.items()},
    }
    write_json(args.data / "system-analysis.json", result)
    print({"gpu_rows": len(gpu_rows), "phase_samples": {k: len(v) for k, v in groups.items()}})


if __name__ == "__main__":
    main()
