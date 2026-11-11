# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Run compiled lanes through explicit resource guards and occupancy leases.

Broker/guard locations are supplied by the operator. Each child owns a broker
lease, so verified child exit restores occupancy even if this supervisor dies.
The manifest lock prevents duplicate supervisors for the same experiment.
"""

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .batch import status
from .runtime import checkpoint_identity, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--broker", type=Path, required=True)
    parser.add_argument("--guard", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--report", action="store_true", help="Refresh common plots and CSV as measurements finish")
    parser.add_argument("--style", type=Path, help="figure-viz assets/mpl directory for --report")
    args = parser.parse_args()
    args.manifest = args.manifest.resolve()
    manifest = json.loads(args.manifest.read_text())
    execution = manifest.get("execution", {})
    args.report = args.report or execution.get("report", False)
    if args.style is None and execution.get("figure_style"):
        args.style = Path(execution["figure_style"])
    if args.report and not (args.style or os.environ.get("FIGURE_VIZ_MPL")):
        raise ValueError("Plotting requires --style, execution.figure_style, or FIGURE_VIZ_MPL")
    lanes = sorted({t["lane"] for t in manifest["tasks"]})
    if not set(lanes) <= set(args.gpus):
        raise ValueError("Manifest lanes exceed the explicitly authorized GPU allocation")
    for path in sorted({t[key] for t in manifest["tasks"] for key in ["student", "teacher"]}):
        checkpoint_identity(path)
    root = args.manifest.parent
    owner = root.name
    children, streams = {}, []
    stopping = False
    report_completed, report_time, report_code = -1, 0.0, None

    def broker(*arguments):
        subprocess.run([sys.executable, str(args.broker), *arguments], check=True, stdout=subprocess.DEVNULL)

    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        for child in children.values():
            if child.poll() is None:
                child.send_signal(signum)

    def refresh(*, final=False):
        nonlocal report_completed, report_time, report_code
        current = status(manifest, root)
        write_json(root / "progress.json", dict(updated_at=time.time(), **current))
        if (
            args.report
            and not stopping
            and (final or (current["completed"] != report_completed and time.monotonic() - report_time >= 60))
        ):
            command = [
                sys.executable,
                "-m",
                "examples.mirroropd.research.studies.teacher_difference",
                "report",
                "--manifest",
                str(args.manifest),
            ]
            if args.style:
                command += ["--style", str(args.style.resolve())]
            with (root / "report.log").open("a") as stream:
                report_code = subprocess.run(command, cwd=args.repo, stdout=stream, stderr=subprocess.STDOUT).returncode
            if report_code == 0:
                report_completed = current["completed"]
            report_time = time.monotonic()
        return current

    with (root / "sweep.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        try:
            current = status(manifest, root)
            for lane in lanes:
                if stopping:
                    raise SystemExit(128 + signal.SIGTERM)
                if all(r["state"] == "complete" for r in current["tasks"] if r["lane"] == lane):
                    continue
                broker("reserve", "--gpu", str(lane), "--owner", owner)
                command = [
                    sys.executable,
                    str(args.guard),
                    "run",
                    "--repo",
                    str(args.repo.resolve()),
                    "--gpus",
                    str(lane),
                    "--",
                    sys.executable,
                    "-u",
                    "-m",
                    "examples.mirroropd.research.studies.teacher_difference",
                    "worker",
                    "--manifest",
                    str(args.manifest),
                    "--lane",
                    str(lane),
                ]
                env = dict(
                    os.environ,
                    OMP_NUM_THREADS="4",
                    OPENBLAS_NUM_THREADS="4",
                    MKL_NUM_THREADS="4",
                    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                )
                try:
                    stream = (root / f"lane-{lane}.log").open("a")
                    streams.append(stream)
                    child = subprocess.Popen(
                        command, cwd=args.repo, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
                    )
                    children[lane] = child
                    broker("attach", "--gpu", str(lane), "--owner", owner, "--pid", str(child.pid))
                except BaseException:
                    if lane in children and children[lane].poll() is None:
                        children[lane].terminate()
                        children[lane].wait()
                    broker("release", "--gpu", str(lane), "--owner", owner, "--reason", "Lane launch failed")
                    raise
            while any(child.poll() is None for child in children.values()):
                refresh()
                time.sleep(1 if stopping else 5)
            codes = {lane: child.wait() for lane, child in children.items()}
            final = refresh(final=True)
            write_json(root / "sweep-exit.json", dict(returncodes=codes, report_returncode=report_code, **final))
            if stopping or any(codes.values()) or final["completed"] != final["planned"] or report_code:
                raise SystemExit(1)
        finally:
            for child in children.values():
                if child.poll() is None:
                    child.terminate()
            for child in children.values():
                child.wait()
            for stream in streams:
                stream.close()


if __name__ == "__main__":
    main()
