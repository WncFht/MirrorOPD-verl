#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Summarize frozen engine queue samples; fractions are samples, not integrated time."""

import argparse
import gzip
import json
import statistics
from pathlib import Path

from examples.mirroropd.research.core.paths import REPO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    root = REPO
    base = root / "examples/mirroropd/docs/archive/evidence/root-data/validation-optimization-20261021"
    out = {}
    for name in ["valid-graph-w16-20261021-1", "valid-graph-w32-20261021-4", "valid-graph-w32-s64-20261021-3"]:
        with gzip.open(base / name / "engine-metrics.jsonl.gz", "rt") as f:
            rows = [json.loads(line) for line in f]
        rounds = []
        for i in range(0, len(rows) - 3, 4):
            b = rows[i : i + 4]
            if len({x["address"] for x in b}) != 4 or not all(x.get("metrics") for x in b):
                continue
            m = [x["metrics"] for x in b]
            done = sum(x.get("vllm:request_success_total", 0) for x in m)
            generated = sum(x.get("vllm:generation_tokens_total", 0) for x in m)
            if not generated or done >= 512:
                continue
            running = [x.get("vllm:num_requests_running", 0) for x in m]
            waiting = [x.get("vllm:num_requests_waiting", 0) for x in m]
            rounds.append({"time": max(x["time"] for x in b), "done": done, "running": running, "waiting": waiting})
        d = {
            "sample_rounds": len(rounds),
            "mean_total_running": statistics.mean(sum(x["running"]) for x in rounds),
            "max_total_running": max(sum(x["running"]) for x in rounds),
            "fraction_any_idle_replica": statistics.mean(any(v == 0 for v in x["running"]) for x in rounds),
            "fraction_idle_replica_while_other_queued": statistics.mean(
                any(v == 0 for v in x["running"]) and sum(x["waiting"]) > 0 for x in rounds
            ),
            "fraction_total_running_below16": statistics.mean(sum(x["running"]) < 16 for x in rounds),
            "fraction_all_waiting_empty": statistics.mean(sum(x["waiting"]) == 0 for x in rounds),
        }
        out[name] = d
        print(name, d)
    args.out.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
