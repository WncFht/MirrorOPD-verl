# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Audit a continuation and its imported attempts; run CPU analyses after completion."""

import argparse
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from examples.mirroropd.research.core.io import dump, read

MODULE = "examples.mirroropd.research.studies.spectral.absorption"


def inspect(root):
    manifest = read(root / "manifest.json")
    rows = []
    for job in manifest["jobs"]:
        folder = root / "jobs" / f"{job['job_id']:03d}" / "attempt-001"
        stages = {stage: (folder / "gradients" / stage / "complete.json").exists() for stage in job["gradient_stages"]}
        rows.append(
            dict(
                job_id=job["job_id"],
                family=job["family"],
                interventions=(folder / "interventions-complete.json").exists(),
                stages=stages,
                complete=bool(stages) and all(stages.values()),
            )
        )
    return dict(
        at=time.time(),
        jobs=rows,
        complete_jobs=sum(r["complete"] for r in rows),
        queue_counts=dict(Counter(t["status"] for t in read(root / "queue.json")["jobs"])),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    root = args.run_root
    while True:
        status = inspect(root)
        dump(root / "monitor/status.json", status)
        if args.once:
            return
        predecessor = read(root / "predecessor.json") if (root / "predecessor.json").exists() else None
        if (root / "exit.json").exists() and (predecessor is None or (Path(predecessor["run"]) / "exit.json").exists()):
            break
        if (root / "STOP").exists():
            return
        time.sleep(args.interval)
    output = root / "analysis-final"
    output.mkdir(exist_ok=False)
    for module, arguments in [
        ("analyze", []),
        ("analyze_interventions", ["--output-dir", str(output / "interventions")]),
        ("analyze_gradients", ["--output", str(output / "gradients.json")]),
    ]:
        with (output / (module + ".log")).open("x") as log:
            subprocess.run(
                [sys.executable, "-m", MODULE + "." + module, "--run-root", str(root), *arguments],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    dump(root / "monitor/complete.json", dict(status=inspect(root), queue_exit=read(root / "exit.json")))


if __name__ == "__main__":
    main()
