# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Audit completed placement/configuration sweeps without mixing timing scopes."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def audit(run):
    if json.loads((run / "exit.json").read_text())["exit_code"] != 0:
        raise ValueError("Placement sweep did not finish successfully")
    records = [json.loads(f.read_text()) for f in sorted(run.glob("*measure-*.json"))]
    references = [r for r in records if r["mode"] == "serial" and r["configuration"]["projection"] == "strict_fp32"]
    if not references:
        raise ValueError("Include a strict_fp32 serial reference in the sweep")
    reference = references[0]
    index = {r["id"]: r for r in reference["results"]}
    groups = defaultdict(list)
    for record in records:
        if record["profile"] or len(record["results"]) != len(index):
            raise ValueError("Trace timing or incomplete sample set")
        if {r["id"] for r in record["results"]} != set(index):
            raise ValueError("Sample identity mismatch")
        if any(r["length"] != index[r["id"]]["length"] for r in record["results"]):
            raise ValueError("Different response token counts")
        groups[(record["tag"].split("-measure-")[0], record["mode"])].append(record)
    report = dict(
        rows=len(index),
        response_tokens=reference["response_tokens"],
        prompt_tokens=reference["prompt_tokens"],
        scope=(
            "Prebuilt CPU FSDP layouts; teacher load through restored four-rank student "
            "and forward probe; no optimizer or rollout"
        ),
        reference=reference["tag"],
        groups={},
    )
    for (name, mode), rows in groups.items():
        label = name if sum(n == name for n, _ in groups) == 1 else f"{name}/{mode}"
        if any(row["configuration"] != rows[0]["configuration"] for row in rows):
            raise ValueError("Configuration changed between repeats")
        times = [r["wall_seconds"] for r in rows]
        median = statistics.median(times)
        representative = min(rows, key=lambda r: abs(r["wall_seconds"] - median))
        ranks = representative["ranks"]
        roles = {"all": ranks} if representative["mode"] == "serial" else {"teacher": ranks[:2], "student": ranks[2:]}
        report["groups"][label] = dict(
            configuration=rows[0]["configuration"],
            seconds=times,
            median_seconds=median,
            peak_allocated_gib=max(r["peak_allocated_gib"] for row in rows for r in row["ranks"]),
            max_abs_row_token_mean_delta=max(
                abs(r["rkl_sum"] - index[r["id"]]["rkl_sum"]) / r["length"] for row in rows for r in row["results"]
            ),
            restored_forward_probe_max_abs=max(
                abs(a - b)
                for row in rows
                for x, y in zip(row["ranks"], reference["ranks"], strict=True)
                for a, b in zip(x["checksum"][0], y["checksum"][0], strict=True)
            ),
            cuda_ranges_mean_seconds={
                role: {
                    stage: statistics.mean(r["cuda_elapsed_seconds"].get(stage, 0) for r in members)
                    for stage in set().union(*(r["cuda_elapsed_seconds"] for r in members))
                }
                for role, members in roles.items()
            },
        )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
