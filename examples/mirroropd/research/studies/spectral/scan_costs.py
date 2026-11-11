# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Replay grouped spectral workloads using shape-matched synchronized GPU samples."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from examples.mirroropd.research.core.io import dump

from .costs import schedule
from .scan import selected_jobs
from .weights import DEFAULT_SCOPE, TensorCheckpoint


def calculate(benchmark_root, manifest, workers=8):
    provenance = {}

    def read(path):
        payload = Path(path).read_bytes()
        provenance[str(path)] = hashlib.sha256(payload).hexdigest()
        return json.loads(payload)

    samples, runs = {}, []
    roots = [benchmark_root] if isinstance(benchmark_root, str | Path) else benchmark_root
    candidates = sorted(p for parent in roots for p in Path(parent).glob("benchmark-*"))
    for root in candidates:
        if not (root / "complete.json").exists():
            continue
        receipt = read(root / "complete.json")
        run = dict(receipt=receipt, modes={})
        for mode in ("weights", "weights-transpose", "updates", "updates-warm"):
            run["modes"][mode] = read(root / mode / "performance.json")
        run["gate"] = read(root / "numerical-gate.json")
        perf = run["modes"]["updates"]
        count = perf["completed"]
        overhead = max(0, perf["wall_seconds"] - sum(perf["phases_seconds"].values())) / count
        for path in (root / "updates").glob("*.json"):
            row = read(path)
            if "targets" not in row:
                continue
            shape = tuple(row["shape"])
            phases = row["phases_seconds"]
            # The benchmark uses one base, two distinct targets and one pair.
            # Keep shape orientation exact; never infer a missing shape by size.
            samples[shape] = dict(
                shape=list(shape),
                source=str(path),
                per_base={k: phases.get(k, 0) for k in ("h2d_base", "base_svd")},
                per_target={
                    k: phases.get(k, 0) / 2
                    for k in (
                        "cache_read",
                        "h2d_target",
                        "target_svdvals",
                        "iso_projection",
                        "update_svdvals",
                        "cache_publish",
                    )
                },
                per_read_tensor=phases["read_hash"] / 3,
                per_pair=phases["missing_svdvals"],
                publication=perf["phases_seconds"]["result_publish"] / count,
                overhead=overhead,
            )
        runs.append(run)
    jobs = selected_jobs(read(manifest), None)
    checkpoints, unavailable = {}, {}
    for job in jobs:
        for path in {job["base"], job["teacher_path"], job["endpoint"], *job["gradient_stages"].values()}:
            if path not in checkpoints and path not in unavailable:
                try:
                    checkpoints[path] = TensorCheckpoint(path)
                except (FileNotFoundError, ValueError) as error:
                    unavailable[path] = str(error)
    runnable, excluded = [], []
    groups = defaultdict(list)
    for job in jobs:
        absent = [job[k] for k in ("base", "teacher_path", "endpoint") if job[k] not in checkpoints]
        if absent:
            excluded.append(dict(job_id=job["job_id"], absent=absent))
            continue
        job = dict(job, gradient_stages={k: v for k, v in job["gradient_stages"].items() if v in checkpoints})
        runnable.append(job)
        groups[job["base"]].append(job)
    workloads = {}
    for mode in ("weights", "updates", "stages"):
        tasks, task_rows, missing, group_costs = [], [], [], defaultdict(float)
        totals = defaultdict(float)
        lane_seconds = [0.0] * workers
        target_matrices, pair_matrices = 0, 0
        for base, paired in groups.items():
            targets = {j[k] for j in paired for k in ("teacher_path", "endpoint")}
            if mode == "stages":
                targets.update(p for j in paired for p in j["gradient_stages"].values())
            pair_count = sum(
                len(dict(endpoint=j["endpoint"], **{k: v for k, v in j["gradient_stages"].items() if k != "endpoint"}))
                if mode == "stages"
                else 1
                for j in paired
            )
            for index, key in enumerate(checkpoints[base].selected(DEFAULT_SCOPE)):
                shape = checkpoints[base].shapes[key]
                target_matrices += len(targets)
                pair_matrices += pair_count
                if shape not in samples:
                    missing.append(dict(base=base, tensor=key, shape=list(shape)))
                    continue
                sample = samples[shape]
                phases = dict(sample["per_base"])
                for phase, seconds in sample["per_target"].items():
                    if phase == "update_svdvals" and mode == "weights":
                        continue
                    count = len(targets) - int(base in targets) if phase == "target_svdvals" else len(targets)
                    phases[phase] = seconds * count
                phases["read_hash"] = sample["per_read_tensor"] * (len(targets) + 1)
                phases["missing_svdvals"] = sample["per_pair"] * pair_count if mode != "weights" else 0
                phases["result_publish"] = sample["publication"]
                phases["other_wall"] = sample["overhead"]
                seconds = sum(phases.values())
                lane_seconds[index % workers] += seconds
                task_rows.append(
                    dict(
                        base=base,
                        tensor=key,
                        shape=list(shape),
                        shard=index % workers,
                        seconds=seconds,
                        phases_seconds=phases,
                    )
                )
                tasks.append(seconds)
                group_costs[base] += seconds
                for name, value in phases.items():
                    totals[name] += value
        workloads[mode] = dict(
            tasks=task_rows,
            base_matrix_tasks=len(tasks) + len(missing),
            target_matrices=target_matrices,
            pair_matrices=pair_matrices,
            missing=missing,
            phases_seconds=dict(totals),
            serial_seconds=sum(tasks) if not missing else None,
            implemented_strided_schedule=dict(
                lane_seconds=lane_seconds, makespan_seconds=max(lane_seconds), work_seconds=sum(tasks)
            )
            if not missing
            else None,
            current_base_group_schedule=schedule(list(group_costs.values()), workers) if not missing else None,
            hypothetical_matrix_schedule=schedule(tasks, workers) if not missing else None,
        )
    return dict(
        schema=1,
        workers=workers,
        eligible_jobs=len(jobs),
        runnable_jobs=len(runnable),
        excluded_jobs=excluded,
        unavailable_checkpoints=unavailable,
        runnable_manifest=dict(jobs=runnable),
        base_groups=len(groups),
        unique_stage_paths=len({p for j in runnable for p in j["gradient_stages"].values()}),
        benchmark_runs=runs,
        shape_samples=list(samples.values()),
        workloads=workloads,
        provenance=provenance,
        assumptions=[
            "One measured layer per model family; extrapolate by exact ordered matrix shape.",
            "Cold result cache; filesystem page cache not reset. Targets deduplicated by path within each base group.",
            "No content dedup across distinct target paths assumed; all stages of a base matrix stay on one shard.",
            "Matrix schedule is a planning bound, not an implemented distributed matrix scheduler.",
            "Per-target times average teacher and endpoint; full-stage contention and memory residency unmeasured.",
            "No statistical confidence interval; no extra model-forward, export, generation or training time included.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, nargs="+", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    dump(args.output, calculate(args.benchmark_root, args.manifest, args.workers))


if __name__ == "__main__":
    main()
