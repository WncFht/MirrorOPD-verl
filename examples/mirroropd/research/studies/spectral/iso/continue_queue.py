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
    """Accept only successful tasks or audited setting cancellations in this predecessor."""
    receipt = root / "training-setting-correction.json"
    cancelled = {}
    if receipt.exists():
        cancelled = {row["task"]["task_id"]: row["task"] for row in read(receipt)["cancelled_children"]}

    def accepted(task):
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
    args = parser.parse_args()
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
            and all(occupied(args.resource, gpu) for gpu in range(8))
        )
        dump(args.successor / "continuation-status.json", dict(at=time.time(), ready=ready))
        if ready:
            run(args.successor, args.repo, args.resource, list(range(8)))
            return
        time.sleep(10)
    dump(args.successor / "continuation-stopped.json", dict(at=time.time()))


if __name__ == "__main__":
    main()
