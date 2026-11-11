# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Launch one explicitly prepared successor only after its predecessor releases leases."""

import argparse
import os
import time
from pathlib import Path

from examples.mirroropd.research.core.io import dump, read
from examples.mirroropd.research.runtime.broker import occupied
from examples.mirroropd.research.runtime.processes import identity

from ..absorption.queue import run


def predecessor_ready(root, predecessor, tasks):
    """Accept successful tasks and audited interruptions with completed replacements."""
    receipt = root / "training-setting-correction.json"
    cancelled = {}
    if receipt.exists():
        cancelled = {row["task"]["task_id"]: row["task"] for row in read(receipt)["cancelled_children"]}

    retirement = predecessor / "rear-lanes-retirement.json"
    retired = {}
    if retirement.exists():
        retired = {row["task"]["task_id"]: row["task"] for row in read(retirement)["tasks"]}

    def accepted(task):
        replacement = any(t.get("retry_of") == task["task_id"] and t["status"] == "complete" for t in tasks)
        interrupted = retired.get(task["task_id"], {})
        if (
            task["status"] == "interrupted"
            and interrupted.get("output_attempt") == task["output_attempt"]
            and replacement
        ):
            return True
        original = cancelled.get(task["task_id"], {})
        return task["status"] == "complete" or (
            task["status"] == "failed"
            and original.get("output_attempt") == task["output_attempt"]
            and original.get("label") == task["label"]
            and str(predecessor) == str(Path(task["output_attempt"]).parents[2])
        )

    return all(accepted(task) for task in tasks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "predecessor", "successor", "repo", "resource"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    args = parser.parse_args()
    if len(set(args.gpus)) != len(args.gpus) or any(g < 0 or g > 7 for g in args.gpus):
        parser.error("Distinct physical GPUs 0..7 required")
    allocation = args.root / "allocation.json"
    if allocation.exists() and not set(args.gpus).issubset(read(allocation)["physical_gpus"]):
        parser.error("Requested GPUs exceed the active allocation")
    receipt = args.successor / "continuation-process.json"
    if receipt.exists():
        record = read(receipt)["identity"]
        if identity(record[0]) == record:
            raise RuntimeError("A live continuation already owns this successor")
    dump(receipt, dict(identity=identity(os.getpid()), predecessor=str(args.predecessor), started=time.time()))
    while not (args.root / "STOP").exists() and not (args.successor / "STOP").exists():
        tasks = read(args.predecessor / "queue.json")["jobs"]
        ready = (
            (args.predecessor / "exit.json").exists()
            and predecessor_ready(args.root, args.predecessor, tasks)
            and all(occupied(args.resource, gpu) for gpu in args.gpus)
        )
        dump(args.successor / "continuation-status.json", dict(at=time.time(), ready=ready))
        if ready:
            run(args.successor, args.repo, args.resource, args.gpus)
            return
        time.sleep(10)
    dump(args.successor / "continuation-stopped.json", dict(at=time.time()))


if __name__ == "__main__":
    main()
