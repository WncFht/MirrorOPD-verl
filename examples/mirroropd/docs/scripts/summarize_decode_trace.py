#!/usr/bin/env python3
"""Summarize CUDA activity in a short Torch trace without double-counting overlap.

The denominator is the first-to-last device activity envelope. This is an
instrumented interval, not nvidia-smi utilization, SM occupancy, or FLOP usage.
"""

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path


def union_duration(intervals):
    """Union of half-open intervals; overlapping streams count only once."""
    ordered = sorted((a, b) for a, b in intervals if b > a)
    if not ordered:
        return 0.0
    total = 0.0
    start, end = ordered[0]
    for left, right in ordered[1:]:
        if left > end:
            total += end - start
            start, end = left, right
        else:
            end = max(end, right)
    return total + end - start


def summarize(trace):
    events = trace["traceEvents"]
    device_events = [
        e
        for e in events
        if e.get("ph") == "X" and e.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset"} and e.get("dur", 0) > 0
    ]
    if not any(e.get("cat") == "kernel" for e in device_events):
        raise ValueError("Trace has no CUDA kernels; CPU-only traces do not validate GPU profiling")
    devices = defaultdict(list)
    for event in device_events:
        devices[event["args"]["device"]].append(event)
    annotations = Counter(e["name"] for e in events if e.get("cat") == "user_annotation")
    out = {"annotations": dict(annotations), "devices": {}}
    for device, records in devices.items():
        kernels = [e for e in records if e["cat"] == "kernel"]
        start = min(e["ts"] for e in records)
        end = max(e["ts"] + e["dur"] for e in records)
        busy = union_duration((e["ts"], e["ts"] + e["dur"]) for e in records)
        kernel_busy = union_duration((e["ts"], e["ts"] + e["dur"]) for e in kernels)
        counts, durations = Counter(), Counter()
        for event in kernels:
            counts[event["name"]] += 1
            durations[event["name"]] += event["dur"]
        out["devices"][device] = {
            "activity_envelope_ms": (end - start) / 1000,
            "device_activity_union_ms": busy / 1000,
            "kernel_union_ms": kernel_busy / 1000,
            "device_activity_envelope_fraction": busy / (end - start),
            "kernel_count": len(kernels),
            "top_kernels_by_summed_duration": [
                {"name": name, "count": counts[name], "sum_ms": duration / 1000}
                for name, duration in durations.most_common(8)
            ],
        }
    out["interpretation"] = (
        "Profiled interval only; overhead changes gaps. Activity union is not compute/bandwidth utilization. "
        "Top-kernel durations are summed and can overlap. Use separate unprofiled runs for speed comparisons."
    )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    opener = gzip.open if args.trace.suffix == ".gz" else open
    with opener(args.trace, "rt") as stream:
        result = summarize(json.load(stream))
    rendered = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
