# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Audit measured matrix costs and replay them on an explicit report cohort.

The legacy weights task and direction preparation are different workloads.
Unmatched shapes stay missing; direction timings are a separately named proxy,
never silently converted into exact spectral-only runtime.
"""

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

from .absorption.cohort import report_eligible


def schedule(tasks, workers):
    """Deterministic longest-processing-time assignment; no startup/I/O contention model."""
    loads = [0.0] * workers
    for value in sorted(tasks, reverse=True):
        lane = min(range(workers), key=lambda i: loads[i])
        loads[lane] += value
    return {"lane_seconds": loads, "makespan_seconds": max(loads), "work_seconds": sum(tasks)}


def summarize(rows):
    values = [r["seconds"] for r in rows]
    return dict(
        count=len(values),
        total_seconds=sum(values),
        mean_seconds=statistics.mean(values),
        median_seconds=statistics.median(values),
        minimum_seconds=min(values),
        maximum_seconds=max(values),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-root", type=Path, required=True)
    parser.add_argument("--expansion-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    provenance, rows, groups = {}, [], defaultdict(list)

    def read(path):
        payload = path.read_bytes()
        provenance[str(path)] = hashlib.sha256(payload).hexdigest()
        return json.loads(payload)

    for job in sorted((args.history_root / "jobs").glob("*.weights-*")):
        for path in sorted((job / "attempt-01/weights").glob("[0-9]*.json")):
            d = read(path)
            row = dict(
                workload="legacy-exact-weights-with-controls",
                job=job.name,
                tensor=d["tensor"],
                shape=d["teacher"]["shape"],
                cache_hit=d.get("reference_cache_hit", False),
                seconds=d["wall_seconds"],
                path=str(path),
            )
            rows.append(row)
            groups[(tuple(row["shape"]), row["cache_hit"])].append(row)
    manifest = read(args.expansion_root / "manifest.json")
    jobs = [j for j in manifest["jobs"] if report_eligible(j)]
    forecasts, directions, matrix_tasks = [], [], []
    for job in jobs:
        folder = args.expansion_root / "jobs" / f"{job['job_id']:03d}" / "attempt-001"
        records = [read(p) for p in sorted((folder / "directions").glob("[0-9]*.json"))]
        if not records:
            continue
        proxy = [r["elapsed_seconds"] for r in records]
        matrix_tasks.extend(proxy)
        directions.append(
            dict(
                job_id=job["job_id"],
                teacher=job["teacher"],
                family=job["family"],
                matrices=len(records),
                measured_compute_seconds=sum(proxy),
                stages=len({v for k, v in job["gradient_stages"].items() if k != "initial"}),
            )
        )
        for cache_hit in (False, True):
            tasks, missing = [], []
            for record in records:
                matches = groups.get((tuple(record["shape"]), cache_hit), [])
                if not matches:
                    missing.append(record["shape"])
                else:
                    tasks.append(statistics.mean(r["seconds"] for r in matches))
            forecasts.append(
                dict(
                    job_id=job["job_id"],
                    teacher=job["teacher"],
                    family=job["family"],
                    reference_cache_hit=cache_hit,
                    matched_matrices=len(tasks),
                    missing_matrices=len(missing),
                    unmatched_shapes=sorted({tuple(s) for s in missing}),
                    matched_only_seconds=sum(tasks),
                    full_job_seconds=sum(tasks) if not missing else None,
                )
            )
    groups_out = [dict(shape=list(shape), cache_hit=hit, **summarize(v)) for (shape, hit), v in sorted(groups.items())]
    by_arm = defaultdict(list)
    for row in rows:
        by_arm[row["job"].split(".weights-")[0]].append(row)
    result = dict(
        schema=1,
        measured_weights={k: summarize(v) for k, v in by_arm.items()},
        shape_costs=groups_out,
        legacy_weights_same_shape_replay=forecasts,
        measured_direction_preparation=directions,
        direction_proxy_schedule=schedule([r["measured_compute_seconds"] for r in directions], args.workers),
        direction_proxy_matrix_schedule=schedule(matrix_tasks, args.workers),
        weights_history_schedule=schedule([r["seconds"] for r in rows], args.workers),
        direction_proxy_scope=(
            "base and learned SVD, missing off/null, three random-frame seeds; excludes final tensor write"
        ),
        forecast_scope=(
            "same-shape historical means, same legacy weights workload; "
            "excludes startup and contention; unmatched shapes are null"
        ),
        spectral_only_total_seconds=None,
        spectral_only_missing_measurement=(
            "Llama/MiMo and large MLP exact weights workload; changed implementation needs fresh calibration"
        ),
        sources=provenance,
        weight_measurements=rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {k: result[k] for k in ("measured_weights", "direction_proxy_schedule", "weights_history_schedule")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
