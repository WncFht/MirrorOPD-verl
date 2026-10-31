# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Audit placement repeats and summarize separate CUDA traces without double counting parents."""

import argparse
import gzip
import json
import statistics
from collections import defaultdict
from pathlib import Path

PRIMARY = ["qwen4-n768-batched", "qwen8-n768", "r1_14-n768"]


def trace_summary(path):
    with gzip.open(path, "rt") as stream:
        events = json.load(stream)["traceEvents"]
    kernels = defaultdict(float)
    for e in events:
        if e.get("ph") != "X" or e.get("cat") != "kernel":
            continue
        name = e.get("name", "").lower()
        if "nccl" in name:
            group = "nccl"
        elif any(s in name for s in ("gemm", "mma", "matmul")):
            group = "matrix_multiply"
        elif "flash" in name or "attention" in name:
            group = "attention"
        elif "softmax" in name:
            group = "softmax"
        elif "reduce" in name or "reduction" in name:
            group = "reduction"
        else:
            group = "other"
        kernels[group] += e.get("dur", 0) / 1e6
    return dict(kernels)


def summarize(root, run):
    directory = root / run
    records = [json.loads(f.read_text()) for f in sorted(directory.glob("measure-*.json"))]
    if len(records) != 6:
        return dict(run=run, complete=False, observed=len(records))
    groups = {m: [r for r in records if r["mode"] == m] for m in ["serial", "parallel"]}
    assert all(len(v) == 3 for v in groups.values())
    reference = {x["id"]: x for x in groups["serial"][0]["results"]}
    delta = 0.0
    restore_delta = 0.0
    for r in records:
        assert not r["profile"] and len(r["results"]) == r["rows"] == len(reference)
        assert {x["id"] for x in r["results"]} == set(reference)
        for x in r["results"]:
            ref = reference[x["id"]]
            assert x["length"] == ref["length"]
            delta = max(delta, abs(x["rkl_sum"] - ref["rkl_sum"]) / x["length"])
        for x, y in zip(r["ranks"], groups["serial"][0]["ranks"], strict=True):
            restore_delta = max(
                restore_delta, max(abs(a - b) for a, b in zip(x["checksum"][0], y["checksum"][0], strict=True))
            )
    result = dict(
        run=run,
        complete=True,
        rows=len(reference),
        response_tokens=records[0]["response_tokens"],
        prompt_tokens=records[0]["prompt_tokens"],
        max_abs_per_token_rkl_delta=delta,
        restore_checksum_max_abs=restore_delta,
        numerical_check_pass=delta <= 1e-5 and restore_delta <= 1e-4,
        placements={},
    )
    for mode, rows in groups.items():
        values = [r["wall_seconds"] for r in rows]
        median = statistics.median(values)
        representative = min(rows, key=lambda r: abs(r["wall_seconds"] - median))
        roles = (
            {"all": representative["ranks"]}
            if mode == "serial"
            else {"teacher": representative["ranks"][:2], "student": representative["ranks"][2:]}
        )
        phases = {}
        for role, ranks in roles.items():
            names = set().union(*(set(r["cuda_elapsed_seconds"]) for r in ranks))
            phases[role] = {
                n: statistics.mean(r["cuda_elapsed_seconds"].get(n, 0) for r in ranks) for n in sorted(names)
            }
        payloads = defaultdict(int)
        for rank in representative["ranks"]:
            for k, v in rank["payload_bytes"].items():
                payloads[k] += v
        traces = {f.name: trace_summary(f) for f in sorted(directory.glob(f"{mode}-rank*.trace.json.gz"))}
        result["placements"][mode] = dict(
            samples_seconds=values,
            median_seconds=median,
            min_seconds=min(values),
            max_seconds=max(values),
            peak_allocated_gib=max(x["peak_allocated_gib"] for r in rows for x in r["ranks"]),
            peak_reserved_gib=max(x["peak_reserved_gib"] for r in rows for x in r["ranks"]),
            median_run=representative["tag"],
            mean_rank_cuda_range_seconds=phases,
            payload_bytes=dict(payloads),
            trace_kernel_work_seconds=traces,
        )
    result["serial_over_parallel_speedup"] = (
        result["placements"]["serial"]["median_seconds"] / result["placements"]["parallel"]["median_seconds"]
    )
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = dict(
        scope="Four-GPU FSDP2 validation prototype, CPU layouts prebuilt, replicated FP32 heads, no optimizer",
        phase_semantics="CUDA range elapsed includes waits; role rows overlap; kernel work sums are not wall time",
        runs=[summarize(args.root, run) for run in PRIMARY],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    for r in report["runs"]:
        print(json.dumps({k: v for k, v in r.items() if k != "placements"}))


if __name__ == "__main__":
    main()
