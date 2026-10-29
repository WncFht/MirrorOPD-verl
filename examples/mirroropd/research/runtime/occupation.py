# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Boundary GPU occupation recovery with explicit per-run output paths."""

import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import time

from examples.mirroropd.research.core.io import dump, read
from examples.mirroropd.research.core.paths import workspace
from examples.mirroropd.research.runtime.processes import send, tagged

WORK = workspace()
TAG = "MIRROROPD_TB_RUN"
OCCUPY = WORK / "cluster_tools/gpu_occupy.sh"


def select_back4_state(output):
    directory = WORK / "cluster_tools/gpu_occupy" / socket.gethostname() / f"uid-{os.getuid()}-teacher-boundary-back4"
    dump(output / "occupy-management.json", dict(state_dir=str(directory), physical_gpus=[4, 5, 6, 7]))


def occupy_env(output):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="4,5,6,7", GPU_OCCUPY_PYTHON=sys.executable)
    env.pop(TAG, None)
    env.pop("GPU_OCCUPY_STATE_DIR", None)
    if (output / "occupy-management.json").exists():
        env["GPU_OCCUPY_STATE_DIR"] = read(output / "occupy-management.json")["state_dir"]
    return env


def occupy_status(output):
    result = subprocess.run(
        ["bash", str(OCCUPY), "status", "--json"],
        env=occupy_env(output),
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    return json.loads(result.stdout)


def restored(status):
    state = status.get("status", {})
    return (
        status.get("health") == "running"
        and state.get("cuda_visible_devices") == "4,5,6,7"
        and set(state.get("workers", {})) == {"0", "1", "2", "3"}
        and all(w["phase"] == "running" and w["allocated_mib"] > 1000 for w in state["workers"].values())
    )


def recover(directory, tag, output):
    with (directory / "recovery.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (directory / "recovered.json").exists():
            return
        for sig, wait in [(signal.SIGTERM, 20), (signal.SIGKILL, 5)]:
            for record in tagged(tag, TAG):
                send(record, sig)
            deadline = time.monotonic() + wait
            while tagged(tag, TAG) and time.monotonic() < deadline:
                time.sleep(1)
        with (directory / "recovery.log").open("a") as log:
            while True:
                status = occupy_status(output)
                if restored(status):
                    time.sleep(6)
                    status = occupy_status(output)
                    if restored(status):
                        dump(directory / "recovered.json", status)
                        return
                if status.get("process_state") != "alive":
                    subprocess.run(
                        [
                            "bash",
                            str(OCCUPY),
                            "start",
                            "--gpus",
                            "0,1,2,3",
                            "--memory-fraction",
                            "0.95",
                            "--heartbeat",
                            "5",
                        ],
                        env=occupy_env(output),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=60,
                    )
                elif status.get("status", {}).get("cuda_visible_devices") != "4,5,6,7":
                    log.write("Unrelated occupy controller owns other GPUs; selecting separate back-four state.\n")
                    log.flush()
                    select_back4_state(output)
                time.sleep(5)
