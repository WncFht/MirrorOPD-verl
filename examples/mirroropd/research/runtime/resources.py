# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Identity-checked back-four GPU ownership and independent occupation recovery."""

import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from examples.mirroropd.research.core.io import dump as write
from examples.mirroropd.research.core.io import read
from examples.mirroropd.research.core.paths import workspace
from examples.mirroropd.research.runtime.processes import identity, send, tagged

WORK = workspace()

TAG = "MIRROROPD_STUDY_RUN"


class Resource:
    def __init__(self):
        self.path = WORK / "wandb-logs/locks" / socket.gethostname() / "teacher-studies.json"
        self.config = read(self.path)
        assert self.config["gpus"] == [4, 5, 6, 7]
        self.script = WORK / "cluster_tools/gpu_occupy.sh"

    def environment(self):
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES="4,5,6,7",
            GPU_OCCUPY_PYTHON=sys.executable,
            GPU_OCCUPY_STATE_DIR=self.config["state_dir"],
        )
        env.pop(TAG, None)
        return env

    def status(self):
        return json.loads(
            subprocess.check_output(
                ["bash", str(self.script), "status", "--json"], env=self.environment(), text=True, timeout=20
            )
        )

    def acquire(self):
        lock = Path(self.config["lock"]).open("a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return lock

    def prepare(self):
        state = self.status()
        if state.get("process_state") == "alive":
            assert state["status"]["cuda_visible_devices"] == "4,5,6,7"
            subprocess.run(["bash", str(self.script), "stop", "--timeout", "30"], env=self.environment(), check=True)
        memory = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True
        )
        assert all(
            int(line.split(",")[1]) < 1024
            for line in memory.splitlines()
            if int(line.split(",")[0]) in self.config["gpus"]
        ), memory

    @staticmethod
    def restored(state):
        value = state.get("status", {})
        return (
            state.get("health") == "running"
            and value.get("cuda_visible_devices") == "4,5,6,7"
            and set(value.get("workers", {})) == {"0", "1", "2", "3"}
            and all(w["phase"] == "running" and w["allocated_mib"] > 1000 for w in value["workers"].values())
        )

    def recover(self, directory, tag):
        with (directory / "recovery.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if (directory / "recovered.json").exists():
                return
            for sig, seconds in [(signal.SIGTERM, 20), (signal.SIGKILL, 5)]:
                for record in tagged(tag, TAG):
                    send(record, sig)
                deadline = time.monotonic() + seconds
                while tagged(tag, TAG) and time.monotonic() < deadline:
                    time.sleep(1)
            with (directory / "recovery.log").open("a") as stream:
                while True:
                    state = self.status()
                    if self.restored(state):
                        time.sleep(6)
                        state = self.status()
                        if self.restored(state):
                            write(directory / "recovered.json", state)
                            return
                    if state.get("process_state") != "alive":
                        subprocess.run(
                            [
                                "bash",
                                str(self.script),
                                "start",
                                "--gpus",
                                "0,1,2,3",
                                "--memory-fraction",
                                "0.95",
                                "--heartbeat",
                                "5",
                            ],
                            env=self.environment(),
                            stdout=stream,
                            stderr=subprocess.STDOUT,
                            timeout=60,
                        )
                    elif state.get("status", {}).get("cuda_visible_devices") != "4,5,6,7":
                        self.config["state_dir"] = str(Path(self.config["state_dir"]).with_name("uid-0-study-back4"))
                        write(self.path, self.config)
                    time.sleep(5)


def guard(directory):
    directory = Path(directory)
    owner = read(directory / "owner.json")
    resource = Resource()
    write(directory / "guard-ready.json", dict(identity=identity(os.getpid())))
    while identity(owner["identity"][0]) == owner["identity"]:
        if (directory / "recovered.json").exists():
            return
        time.sleep(3)
    resource.recover(directory, owner["tag"])
