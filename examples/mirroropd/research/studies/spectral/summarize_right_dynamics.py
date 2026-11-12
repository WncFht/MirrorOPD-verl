# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Validate and freeze right-frame CPU measurements beside the report."""

import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

from .update_dynamics import atomic_json


def summarize(source, output, training=None):
    status = json.loads((source / "exit.json").read_text())
    if status["failed"] or status["complete"] != status["total"]:
        raise ValueError("Incomplete measurement attempt")
    records = [json.loads(gzip.decompress(p.read_bytes())) for p in sorted(source.glob("matrix-*.json.gz"))]
    if len(records) != status["total"]:
        raise ValueError("Missing result records")
    identities = set()
    groups = {key: defaultdict(list) for key in ["pairs", "increments", "weight_angles", "stages"]}
    max_self_angle = 0
    for rec in records:
        task = rec["task"]
        identity = (task["condition"], task["tensor"])
        if identity in identities:
            raise ValueError("Duplicate condition/tensor")
        identities.add(identity)
        cond, module, layer = task["condition"], task["tensor"].split(".")[-2], int(task["tensor"].split(".")[2])
        assert max(p[0] for p in task["points"]) <= 320
        for row in rec["pairs"]:
            assert len(row["angles_deg"]) == row["rank"]
            assert all(0 <= a <= 90 for a in row["angles_deg"])
            assert abs(row["overlap"] + row["chordal_distance"] ** 2 - 1) < 1e-10
            if row["step_a"] == row["step_b"]:
                max_self_angle = max(max_self_angle, row["max_angle_deg"])
                assert row["overlap"] > 1 - 1e-10
            groups["pairs"][cond, module, row["rank"], row["step_a"], row["step_b"]].append(row)
        for row in rec["weight_angles"]:
            groups["weight_angles"][cond, module, row["rank"], row["step"]].append(row)
        for row in rec["stages"]:
            groups["stages"][cond, module, row["step"]].append(row)
        for inc in rec["increments"]:
            for row in inc["captures"]:
                assert -1e-10 <= row["capture"] <= row["optimal_capture"] + 1e-8
                groups["increments"][cond, module, row["rank"], row["anchor"], inc["start"], inc["end"]].append(
                    dict(row, energy=inc["energy"], layer=layer)
                )
    summary = {}
    for kind, grouped in groups.items():
        rows = []
        for key, values in grouped.items():
            cond, module, *rest = key
            row = dict(condition=cond, module=module, matrices=len(values))
            fields = {
                "pairs": ["rank", "step_a", "step_b"],
                "weight_angles": ["rank", "step"],
                "increments": ["rank", "anchor", "start", "end"],
                "stages": ["step"],
            }[kind]
            row.update(zip(fields, rest, strict=True))
            metrics = {
                "pairs": [
                    "mean_angle_deg",
                    "median_angle_deg",
                    "p90_angle_deg",
                    "max_angle_deg",
                    "overlap",
                    "adjusted_overlap",
                    "chordal_distance",
                    "a_captures_b",
                    "b_captures_a",
                ],
                "weight_angles": ["mean_angle_deg", "p90_angle_deg", "max_angle_deg", "overlap"],
                "increments": ["energy", "capture", "optimal_capture", "relative_to_optimal"],
                "stages": ["energy", "stable_rank", "rank90"],
            }[kind]
            for metric in metrics:
                present = [v[metric] for v in values if v[metric] is not None]
                row[metric] = median(present) if present else None
            rows.append(row)
        summary[kind] = rows
    risks = []
    if training:
        for cond in ["E63", "E64", "E65"]:
            for event in sorted((training / cond / "evaluation/events").glob("outer-*/manifest.json")):
                item = json.loads(event.read_text())
                meta = item["metadata"]
                step = meta["optimizer_steps"]
                if step > 320 or item["status"] != "complete":
                    continue
                metrics = meta["metrics"]
                for bank in ["public_initial", "public_teacher"]:
                    for domain in ["all", "math", "code", "if"]:
                        prefix = f"eval/rkl/{bank}/{domain}"
                        risks.append(
                            dict(
                                condition=cond,
                                step=step,
                                bank=bank,
                                domain=domain,
                                fixed_h1024=metrics[f"{prefix}/fixed_h1024"],
                                question_mean=metrics[f"{prefix}/question_mean"],
                                n=metrics[f"{prefix}/rows"],
                            )
                        )
    output.mkdir(parents=True, exist_ok=False)
    (output / "measurements.json.gz").write_bytes(gzip.compress(json.dumps(records).encode(), mtime=0))
    atomic_json(output / "summary.json", dict(**summary, risks=risks))
    provenance = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.glob("matrix-*.json.gz"))}
    atomic_json(output / "provenance.json", dict(source=str(source), files=provenance))
    atomic_json(
        output / "verification.json",
        dict(
            status=status,
            unique_matrices=len(identities),
            max_checkpoint_step=320,
            max_self_angle_deg=max_self_angle,
            pair_rows=sum(len(r["pairs"]) for r in records),
            increment_rows=sum(len(i["captures"]) for r in records for i in r["increments"]),
            full_weight_rows=sum(len(r["weight_angles"]) for r in records),
            new_training=False,
            gpu_used=False,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--training", type=Path)
    args = parser.parse_args()
    summarize(args.source, args.output, args.training)


if __name__ == "__main__":
    main()
