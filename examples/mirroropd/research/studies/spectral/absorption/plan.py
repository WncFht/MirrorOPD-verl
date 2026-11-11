# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Validate local checkpoints and prepare an explicit absorption queue."""

import argparse
from pathlib import Path

from examples.mirroropd.research.core.io import dump, read

from ...teacher_difference.runtime import checkpoint_identity
from .queue import initialize


def missing_shards(path):
    try:
        checkpoint_identity(path)
    except FileNotFoundError as error:
        return [str(error)]
    return []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--job-ids", type=int, nargs="+")
    parser.add_argument("--input-run", type=Path)
    parser.add_argument("--projection", choices=("cpu-reference", "cuda-stream"), default="cpu-reference")
    parser.add_argument("--stop-path", type=Path, action="append", default=[])
    args = parser.parse_args()
    manifest = read(args.manifest)
    selected = args.job_ids or [j["job_id"] for j in manifest["jobs"]]
    tasks, checks = [], {}
    for job in manifest["jobs"]:
        if job["job_id"] not in selected:
            continue
        for path in [job["base"], job["teacher_path"], job["endpoint"], *job["gradient_stages"].values()]:
            checks[path] = missing_shards(path)
        if any(
            checks[p] for p in [job["base"], job["teacher_path"], job["endpoint"], *job["gradient_stages"].values()]
        ):
            raise ValueError(f"Incomplete checkpoint for job {job['job_id']}: {checks}")
        out = args.output / "jobs" / f"{job['job_id']:03d}" / "attempt-001"
        task = dict(job_id=job["job_id"], phase="all", output_attempt=str(out.absolute()), projection=args.projection)
        if args.input_run:
            previous = args.input_run / "jobs" / f"{job['job_id']:03d}" / "attempt-001"
            task.update(
                phase="gradients",
                input_attempt=str(previous.absolute()),
                ready_file=str((previous / "interventions-complete.json").absolute()),
            )
        tasks.append(task)
    if len(tasks) != len(set(selected)) or len(selected) != len(set(selected)):
        raise ValueError("Unknown or duplicate requested job IDs")
    initialize(args.output, manifest, read(args.prompts), tasks, args.repo, stop_paths=args.stop_path)
    dump(args.output / "checkpoint-preflight.json", checks)
    # Preserve read-only visibility of jobs that continue under the predecessor's guard.
    if args.input_run:
        for job in manifest["jobs"]:
            if job["job_id"] not in selected:
                target = args.output / "jobs" / f"{job['job_id']:03d}"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to((args.input_run / "jobs" / target.name).absolute(), target_is_directory=True)
        dump(args.output / "predecessor.json", dict(run=str(args.input_run.absolute()), migrated_job_ids=selected))


if __name__ == "__main__":
    main()
