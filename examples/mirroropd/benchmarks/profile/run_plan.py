# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Run an explicit confirmation plan sequentially under the shared idle GPU lock."""

import argparse
import fcntl
import os
import signal
import subprocess
import time
from pathlib import Path

import psutil

from examples.mirroropd.benchmarks.profile.run_study import load_module, save


def main():
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    plan = json.loads((root / "plan.json").read_text())
    source = Path(plan["source"])
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if head != plan["source_commit"]:
        raise RuntimeError("Confirmation plan source commit mismatch")
    helpers = load_module(source / "examples/mirroropd/tools/queue_full_vocab_ablations.py", "helpers")
    state = {"pid": os.getpid(), "jobs": {}, "source_commit": head}

    def status(**updates):
        state.update(updates, updated_at=datetime.now(timezone.utc).isoformat())
        save(root / "status.json", state)

    def stopped():
        if (root / "STOP").exists():
            raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        with Path("/tmp/mirroropd-gpus-0-1-2-3.lock").open("a") as lock:
            status(state="waiting_lock")
            while True:
                stopped()
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(30)
            for job in plan["jobs"]:
                directory = root / job["name"]
                if (directory / "exit.json").exists():
                    state["jobs"][job["name"]] = json.loads((directory / "exit.json").read_text())
                    continue
                since = None
                while True:
                    stopped()
                    idle, snapshot = helpers.gpu_snapshot()
                    since = (since or time.monotonic()) if idle else None
                    status(state="waiting_gpus", current_job=job["name"], gpu_snapshot=snapshot)
                    if since is not None and time.monotonic() - since >= 120:
                        break
                    time.sleep(30)
                directory.mkdir(parents=True, exist_ok=True)
                env = {
                    "CUDA_VISIBLE_DEVICES": "0,1,2,3",
                    "PYTHONPATH": str(source) + os.pathsep + str(Path(__file__).resolve().parents[4]),
                    "OMP_NUM_THREADS": "1",
                    "TOKENIZERS_PARALLELISM": "false",
                    "NCCL_DEBUG": "WARN",
                } | job.get("environment", {})
                save(directory / "launch.json", job | {"source_commit": head, "environment": env})
                status(state="running", current_job=job["name"])
                owned = {}
                start = time.monotonic()
                with (directory / "stdout.log").open("w") as log:
                    child = subprocess.Popen(
                        job["command"],
                        cwd=source,
                        env=os.environ | env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    try:
                        while child.poll() is None:
                            try:
                                for proc in psutil.Process(child.pid).children(recursive=True):
                                    owned[proc.pid] = proc.create_time()
                            except psutil.NoSuchProcess:
                                pass
                            if time.monotonic() - start > job.get("timeout", 1800):
                                raise TimeoutError
                            time.sleep(2)
                        code = child.returncode
                    except TimeoutError:
                        code = 124
                    finally:
                        helpers.cleanup_owned(child, owned)
                record = {"returncode": code, "wall_seconds": time.monotonic() - start}
                save(directory / "exit.json", record)
                state["jobs"][job["name"]] = record
                status(state="between_jobs")
            status(
                state="completed"
                if all(r["returncode"] == 0 for r in state["jobs"].values())
                else "completed_with_failures"
            )
    except KeyboardInterrupt:
        status(state="stopped")
    except Exception as exc:
        status(state="failed", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
