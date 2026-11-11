# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Recompute paired intervention results with calibration-only amplitude selection."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.run_root / "manifest.json").read_text())
    rows = []
    selections = []
    coverage = []
    for job in manifest["jobs"]:
        out = args.run_root / "jobs" / f"{job['job_id']:03d}" / "attempt-001"
        scores = {
            p.stem: json.loads(p.read_text())
            for p in (out / "scores").glob("*.json")
            if not p.name.endswith(".progress.json")
        }
        coverage.append(
            dict(
                job_id=job["job_id"],
                run_id=job["run_id"],
                scores=len(scores),
                complete=(out / "complete.json").exists(),
                gradient_complete=[p.parent.name for p in (out / "gradients").glob("*/complete.json")],
            )
        )
        if "endpoint" not in scores:
            continue
        base = {(x["origin"], x["row_id"]): x for x in scores.get("base", dict(rows=[]))["rows"]}
        endpoint = {(x["origin"], x["row_id"]): x for x in scores["endpoint"]["rows"]}
        for condition, data in scores.items():
            groups = defaultdict(list)
            for x in data["rows"]:
                groups[x["origin"], x["split"], x["domain"]].append(x)
            for (origin, split, domain), values in groups.items():
                for metric in ["rkl", "fkl", "js", "student_entropy", "teacher_entropy"]:
                    current = np.array([v[metric] for v in values])
                    old = np.array([endpoint[origin, v["row_id"]][metric] for v in values])
                    delta = old - current
                    rng = np.random.default_rng(20261110)
                    boot = np.mean(delta[rng.integers(0, len(delta), size=(2000, len(delta)))], axis=1)
                    initial = np.array([base[origin, v["row_id"]][metric] for v in values]) if base else None
                    rows.append(
                        dict(
                            job_id=job["job_id"],
                            run_id=job["run_id"],
                            family=job["family"],
                            teacher=job["teacher"],
                            cohort=job["cohort"],
                            step=job["step"],
                            condition=condition,
                            origin=origin,
                            split=split,
                            domain=domain,
                            metric=metric,
                            n=len(values),
                            absolute=float(current.mean()),
                            endpoint=float(old.mean()),
                            improvement=float(delta.mean()),
                            improvement_ci_low=float(np.quantile(boot, 0.025)),
                            improvement_ci_high=float(np.quantile(boot, 0.975)),
                            remaining=float(current.mean() / initial.mean())
                            if initial is not None and initial.mean() != 0
                            else None,
                        )
                    )
        for band in ["off", "null"]:
            candidates = {"endpoint": scores["endpoint"]}
            candidates.update({k: v for k, v in scores.items() if k.startswith(f"missing-{band}-beta-")})
            if len(candidates) < 4:
                continue
            means = {
                k: float(np.mean([r["rkl"] for r in v["rows"] if r["split"] == "calibration"]))
                for k, v in candidates.items()
            }
            selected = min(means, key=means.get)
            selections.append(
                dict(
                    job_id=job["job_id"],
                    band=band,
                    selection="joint-origin calibration RKL, equal prompt counts",
                    calibration=means,
                    selected=selected,
                )
            )
    (args.run_root / "analysis").mkdir(exist_ok=True)
    (args.run_root / "analysis/coverage.json").write_text(json.dumps(coverage, indent=2))
    (args.run_root / "analysis/calibration-selections.json").write_text(json.dumps(selections, indent=2))
    if rows:
        with (args.run_root / "analysis/paired-results.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (args.run_root / "analysis/summary.json").write_text(
        json.dumps(
            dict(
                jobs=len(manifest["jobs"]),
                complete=sum(x["complete"] for x in coverage),
                conditions=sum(x["scores"] for x in coverage),
                rows=len(rows),
                notes=[
                    "bootstrap unit is question within domain/origin",
                    "one training seed; three random control seeds",
                    "H128 diagnostic panel separate from historical H1024",
                    "gradients are on newly generated frozen states, not historical training rollout",
                ],
            ),
            indent=2,
        )
    )
    print(
        json.dumps(
            dict(
                jobs=len(manifest["jobs"]),
                complete=sum(x["complete"] for x in coverage),
                conditions=sum(x["scores"] for x in coverage),
            )
        )
    )


if __name__ == "__main__":
    main()
