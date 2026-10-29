# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Summarize complete native generation batches without implying lifecycle acceptance."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def summarize(root):
    config = json.loads((root / "config.json").read_text())
    rows = [json.loads(line) for line in (root / "results.jsonl").read_text().splitlines()]
    samples = [row for row in rows if row.get("repeat", -1) >= 0]
    groups = {}
    for row in samples:
        responses = row["responses"]
        if len(responses) != config["batch"] or len({r["request_id"] for r in responses}) != config["batch"]:
            raise ValueError("Incomplete or duplicate response batch")
        lengths = [len(r["token_ids"]) for r in responses]
        if any(length < 1 or length > config["cap"] for length in lengths):
            raise ValueError("Invalid response length")
        if any(len(r["log_probs"]) != len(r["token_ids"]) for r in responses):
            raise ValueError("Behavior probability coverage differs from output tokens")
        if any(not math.isfinite(value) for r in responses for value in r["log_probs"]):
            raise ValueError("Nonfinite selected-token probability")
        tokens = sum(lengths)
        seconds = row["seconds"]
        groups.setdefault(str(row["alpha"]), []).append(
            {
                "repeat": row["repeat"],
                "seconds": seconds,
                "effective_tokens": tokens,
                "tokens_per_second": tokens / seconds,
                "gpu_hours_per_million_tokens": seconds / 3600 / (tokens / 1e6),
                "length_p50": percentile(lengths, 0.5),
                "length_p90": percentile(lengths, 0.9),
                "length_max": max(lengths),
                "cap_hits": sum(r["finish_reason"] == "length" for r in responses),
                "length_at_cap": sum(n == config["cap"] for n in lengths),
                "eos": sum(r["finish_reason"] == "stop" for r in responses),
            }
        )
    if {float(alpha) for alpha in groups} != set(config["alphas"]) or any(
        sorted(row["repeat"] for row in values) != list(range(config["repeats"])) for values in groups.values()
    ):
        raise ValueError("Missing measured repeats")
    resources = [json.loads(line) for line in (root / "resources.jsonl").read_text().splitlines()]
    allocated = resources[0]["physical_gpus"]
    if not allocated or not set(allocated) <= {6, 7} or resources[-1]["kind"] != "end":
        raise ValueError("Incomplete single-GPU resource interval")
    gpu_rows = [gpu for row in resources if row["kind"] == "sample" for gpu in row["gpus"]]
    if any(gpu["index"] not in allocated for gpu in gpu_rows):
        raise ValueError("Unexpected GPU in measurement")
    for values in groups.values():
        for row in values:
            row["gpu_hours_per_million_tokens"] *= len(allocated)
    summary = {}
    for alpha, values in groups.items():
        summary[alpha] = {
            "repeats": values,
            "median_seconds": statistics.median(v["seconds"] for v in values),
            "range_seconds": [min(v["seconds"] for v in values), max(v["seconds"] for v in values)],
            "median_tokens_per_second": statistics.median(v["tokens_per_second"] for v in values),
        }
    return {
        "scope": "Standalone generation only; cold startup, training, numerical and lifecycle gates are separate",
        "config": config,
        "source": json.loads((root / "source.json").read_text()),
        "prompt_sha256": hashlib.sha256((root / "prompts.json").read_bytes()).hexdigest(),
        "formal_generation_complete": config["batch"] == 64 and config["cap"] == 7168 and config["repeats"] >= 3,
        "groups": summary,
        "resource_scope": "Full monitored process lifetime; allocator and device contexts included",
        "physical_gpus": allocated,
        "nvml_peak_bytes": max(row["memory_used"] for row in gpu_rows),
        "process_wall_seconds": resources[-1]["wall_seconds"],
        "preemptions": "unavailable",
        "recompute": "unavailable",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.out.open("x") as stream:
        json.dump(summarize(args.run), stream, indent=2)


if __name__ == "__main__":
    main()
