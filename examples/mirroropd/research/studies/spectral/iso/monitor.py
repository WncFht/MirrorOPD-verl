# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Local overnight coverage/health monitor; no remote messaging or GPU allocation."""

import argparse
import collections
import fcntl
import os
import time
import traceback
from pathlib import Path

from examples.mirroropd.research.core.io import dump, read
from examples.mirroropd.research.runtime.broker import occupied
from examples.mirroropd.research.runtime.processes import identity

from .analyze import publish, queues, summarize
from .plot import render


def run(root, resource, docs, interval=120):
    root, resource, docs = Path(root), Path(resource), Path(docs)
    destination = root / "monitor"
    destination.mkdir(exist_ok=True)
    with (destination / "monitor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        dump(destination / "process.json", dict(identity=identity(os.getpid()), started=time.time()))
        previous = None
        while not (root / "MONITOR_STOP").exists():
            queue_roots, queue = queues(root)
            now = time.time()
            counts = dict(collections.Counter(t["status"] for t in queue))
            scans = []
            for index in range(8):
                progress = root / f"scan-v1/shard-{index}/progress.json"
                scans.append(dict(shard=index, progress=read(progress) if progress.exists() else None))
            broker = read(resource / "status.json")
            health = []
            for task in queue:
                if task["status"] == "running":
                    record = task.get("child_identity")
                    health.append(
                        dict(
                            task_id=task["task_id"],
                            queue_root=task["queue_root"],
                            label=task["label"],
                            heartbeat_age=now - task.get("heartbeat_at", 0),
                            child_alive=bool(record and identity(record[0]) == record),
                            output_attempt=task["output_attempt"],
                        )
                    )
            status = dict(
                at=now,
                counts=counts,
                scan=scans,
                running=health,
                gpus={
                    k: dict(phase=v["phase"], owner=v.get("owner"), occupier_alive=v.get("occupier_alive"))
                    for k, v in broker["gpus"].items()
                },
                failed=[
                    dict(task=t, error=str(Path(t["queue_root"]) / "errors" / f"task-{t['task_id']}.json"))
                    for t in queue
                    if t["status"] == "failed"
                ],
            )
            dump(destination / "status.json", status)
            signature = (
                tuple(sorted(counts.items())),
                tuple((t["queue_root"], t["task_id"]) for t in queue if t["status"] == "complete"),
                all((root / f"scan-v1/shard-{i}/complete.json").exists() for i in range(8)),
            )
            if signature != previous:
                try:
                    summary = summarize(root, root / "analysis")
                    if summary["scan_complete"] or summary["training"]:
                        render(summary, docs / "figures/iso-20261111")
                    publish(root, summary, docs)
                    dump(destination / f"review-{time.time_ns()}.json", status)
                    previous = signature
                except Exception:
                    dump(destination / f"analysis-error-{time.time_ns()}.json", dict(traceback=traceback.format_exc()))
            if previous == signature and all((path / "exit.json").exists() for path in queue_roots):
                exit_record = {str(path): read(path / "exit.json") for path in queue_roots}
                restored = all(occupied(resource, int(gpu)) for gpu in broker["gpus"])
                if restored:
                    dump(
                        destination / "complete.json",
                        dict(
                            at=now,
                            queue_exit=exit_record,
                            all_tasks_passed=all(t["status"] == "complete" for t in queue),
                            occupancy_restored=True,
                        ),
                    )
                    return
            # Short waits keep local stop responsive. This does not create another hourly hook.
            until = time.monotonic() + interval
            while time.monotonic() < until and not (root / "MONITOR_STOP").exists():
                time.sleep(2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["root", "resource", "docs"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--interval", type=int, default=120)
    run(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
