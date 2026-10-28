# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Summarize complete standalone runs without hiding tokens, EOS, or resource costs."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def quantile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    lo = int(index)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def summarize(directory):
    if not (directory / "summary.json").exists():
        raise ValueError(f"Incomplete run: {directory}")
    metadata = []
    groups = defaultdict(list)
    for path in sorted(directory.glob("rank-*.jsonl")):
        for line in path.open():
            row = json.loads(line)
            if row["kind"] == "metadata":
                metadata.append(row)
            elif row["kind"] == "measurement":
                row.pop("results")
                groups[row["compact"], row["microbatch"]].append(row)
    first = metadata[0]
    reports = []
    for (compact, microbatch), rows in sorted(groups.items()):
        raw = []
        gpu = defaultdict(lambda: {"allocated_peak_bytes": 0, "reserved_peak_bytes": 0, "nvml_peak_bytes": 0})
        utilization = defaultdict(list)
        rss = defaultdict(int)
        for row in rows:
            for peak in row["gpu_peaks"]:
                item = gpu[peak["index"]]
                item["allocated_peak_bytes"] = max(item["allocated_peak_bytes"], peak["allocated"])
                item["reserved_peak_bytes"] = max(item["reserved_peak_bytes"], peak["reserved"])
            for sample in row["telemetry"]:
                rss[row["rank"]] = max(rss[row["rank"]], sample["rss_bytes"])
                for entry in sample["gpus"]:
                    gpu[entry["index"]]["nvml_peak_bytes"] = max(
                        gpu[entry["index"]]["nvml_peak_bytes"], entry["used_bytes"]
                    )
                    utilization[entry["index"]].append(entry["util"])
        for repeat in sorted({row["repeat"] for row in rows}):
            batch = [row for row in rows if row["repeat"] == repeat]
            if (
                len(batch) != first["args"]["replicas"]
                or sum(row["completed"] for row in batch) != first["args"]["batch"]
            ):
                raise ValueError(f"Missing replica/request in {directory}, repeat {repeat}")
            lengths = [length for row in batch for length in row["lengths"]]
            wall = max(row["seconds"] for row in batch)
            tokens = sum(row["effective_tokens"] for row in batch)
            raw.append(
                {
                    "repeat": repeat,
                    "seconds": wall,
                    "effective_tokens": tokens,
                    "tokens_per_second": tokens / wall,
                    "completed": len(lengths),
                    "eos": sum(row["eos"] for row in batch),
                    "length_min": min(lengths),
                    "length_median": statistics.median(lengths),
                    "length_p95": quantile(lengths, 0.95),
                    "length_max": max(lengths),
                }
            )
        for device, values in utilization.items():
            gpu[device]["utilization_mean_percent"] = statistics.mean(values)
            gpu[device]["utilization_range_percent"] = [min(values), max(values)]
        reports.append(
            {
                "compact": compact,
                "microbatch": microbatch,
                "raw": raw,
                "median_seconds": statistics.median(row["seconds"] for row in raw),
                "range_seconds": [min(row["seconds"] for row in raw), max(row["seconds"] for row in raw)],
                "median_effective_tokens_per_second": statistics.median(row["tokens_per_second"] for row in raw),
                "gpu": dict(gpu),
                "rss_peak_bytes_by_rank": dict(rss),
                "sum_of_rank_rss_peaks_bytes": sum(rss.values()),
                "kv_peak_bytes_per_replica": max(
                    stage["kv_peak_bytes"] for row in rows for stage in row["host_stage_times"]
                ),
            }
        )
    return {
        "directory": str(directory.resolve()),
        "arguments": first["args"],
        "resolved_precision": first.get("resolved_precision"),
        "prompt_lengths": first["prompt_lengths"],
        "sampling": first["sampling"],
        "stop_ids": first["stop_ids"],
        "native_eos": first["native_eos"],
        "cold_checkpoint_load_seconds_by_rank": {row["rank"]: row["cold_load_seconds"] for row in metadata},
        "cache_policy": first["cache_policy"],
        "profiler": first["profiler"],
        "preemption": first["preemption"],
        "recompute": first["recompute"],
        "source": first.get("source"),
        "measurements": reports,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps([summarize(path) for path in args.runs], indent=2) + "\n")


if __name__ == "__main__":
    main()
