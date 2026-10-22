# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Attribute GPU kernel work to the deepest semantic CPU launch range.

External id links CUDA events to their launching PyTorch operation. A CPU range
is used for attribution only: its host duration is never used as GPU elapsed time.
Unmapped launches stay explicitly unattributed. Nested ranges are not double counted.
"""

import argparse
import bisect
import gzip
import json
from collections import defaultdict
from pathlib import Path

from analyze import trace_stats


def attribute(events):
    annotations = defaultdict(list)
    cpu = {}
    for cat in ("user_annotation", "cpu_op"):
        for event in events:
            if event.get("cat") != cat or event.get("ph") != "X":
                continue
            external = event.get("args", {}).get("External id")
            if external is not None:
                cpu[external] = event
            if cat == "user_annotation" and event["name"].startswith("full_vocab/"):
                annotations[(event["pid"], event["tid"])].append(event)
    starts = {}
    for key, spans in annotations.items():
        spans.sort(key=lambda e: e["ts"])
        starts[key] = [e["ts"] for e in spans]
    launches = {
        e.get("args", {}).get("correlation"): e
        for e in events
        if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})
    }
    work, count = defaultdict(float), defaultdict(int)
    top = defaultdict(lambda: {"sum_ms": 0.0, "count": 0})
    for event in events:
        if event.get("cat") != "kernel":
            continue
        label = "unattributed"
        # Prefer the actual launch timestamp. Triton's External id can refer to
        # the enclosing autograd Function, whose start predates the semantic range.
        launch = launches.get(event.get("args", {}).get("correlation"))
        if launch is None:
            launch = cpu.get(event.get("args", {}).get("External id"))
        if launch is not None:
            key = (launch["pid"], launch["tid"])
            stamp = launch["ts"]
            for idx in range(bisect.bisect_right(starts.get(key, []), stamp) - 1, -1, -1):
                span = annotations[key][idx]
                if span["ts"] + span.get("dur", 0) >= stamp:
                    label = span["name"]
                    break
        duration = event.get("dur", 0) / 1000
        work[label] += duration
        count[label] += 1
        top[event["name"]]["sum_ms"] += duration
        top[event["name"]]["count"] += 1
    return {
        "semantic_kernel_sum_ms": dict(work),
        "semantic_kernel_count": dict(count),
        "top_kernels": sorted(({"name": k, **v} for k, v in top.items()), key=lambda r: -r["sum_ms"])[:20],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("traces", type=Path, nargs="+")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    results = []
    for path in args.traces:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as f:
            events = json.load(f)["traceEvents"]
        result = attribute(events)
        result["trace_statistics"], _ = trace_stats(path)
        result["source"] = str(path)
        results.append(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
