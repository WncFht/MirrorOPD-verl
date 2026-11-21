# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Run authorized, serialized Codex training reviews at a persisted interval.

The reviewer diagnoses failures and can prepare bounded follow-up experiments.
Original logs, frozen sources and checkpoints remain immutable evidence.
"""

import argparse
import fcntl
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from examples.mirroropd.research.core.io import dump, read


def review_once(root, repo, prompt, *, timeout=1800):
    codex = shutil.which("codex")
    if codex is None:
        raise FileNotFoundError("codex CLI required for diagnosis and repair reviews")
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    folder = root / stamp
    folder.mkdir()
    command = [
        codex,
        "exec",
        "--ephemeral",
        "--sandbox",
        "danger-full-access",
        "-c",
        'approval_policy="never"',
        "-C",
        str(repo),
        "-o",
        str(folder / "analysis.md"),
        "-",
    ]
    started = time.time()
    dump(root / "status.json", dict(state="reviewing", pid=os.getpid(), started_at=started, output=str(folder)))
    with (folder / "events.log").open("x") as log:
        child = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, text=True
        )
        try:
            child.communicate(prompt, timeout=timeout)
        except subprocess.TimeoutExpired:
            # Only terminate this review's isolated process group, never training jobs.
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            dump(
                folder / "exit.json",
                dict(returncode=child.returncode, error="review timeout", started_at=started, finished_at=time.time()),
            )
            raise TimeoutError("Review timed out; inspect events.log; training remains independent") from None
    receipt = dict(
        returncode=child.returncode,
        started_at=started,
        finished_at=time.time(),
        summary_written=(folder / "analysis.md").is_file(),
    )
    dump(folder / "exit.json", receipt)
    if child.returncode or not receipt["summary_written"]:
        raise RuntimeError(f"Review failed: {folder}; training remains independent")
    dump(root / "latest.json", dict(**receipt, output=str(folder)))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    root, repo = args.root.resolve(), args.repo.resolve()
    plan = read(root / "plan.json")
    interval = plan["interval_seconds"]
    if interval < 60:
        raise ValueError("Review interval must be at least one minute")
    with (root / "review.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = root / "schedule.json"
        next_at = read(state_path)["next_at"] if state_path.exists() else time.time() + interval
        while not (root / "STOP").exists():
            if not args.once and time.time() < next_at:
                dump(root / "status.json", dict(state="scheduled", pid=os.getpid(), next_at=next_at))
                time.sleep(min(30, max(0, next_at - time.time())))
                continue
            started = time.time()
            try:
                review_once(root, repo, Path(plan["prompt"]).read_text(), timeout=plan.get("timeout_seconds", 1800))
            except Exception as error:
                dump(root / "review-error.json", dict(time=time.time(), error=repr(error)))
            next_at = max(started + interval, time.time() + 1)
            dump(state_path, dict(next_at=next_at, interval_seconds=interval))
            if args.once:
                break


if __name__ == "__main__":
    main()
