# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Identity-checked allocated GPU ownership and independent occupation recovery."""

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


def live_waiters(directory):
    """Priority then arrival order; dead PID/start-time identities cannot block the queue."""
    result = []
    for path in directory.glob("*.json"):
        try:
            value = read(path)
        except FileNotFoundError:
            continue
        if identity(value["identity"][0]) != value["identity"]:
            path.unlink(missing_ok=True)
            continue
        result.append((value["priority"], value["enqueued_ns"], value["identity"][0], path))
    return sorted(result)


class Resource:
    def __init__(self):
        self.path = Path(
            os.environ.get(
                "MIRROROPD_RESOURCE_CONFIG", WORK / "wandb-logs/locks" / socket.gethostname() / "teacher-studies.json"
            )
        )
        self.config = read(self.path)
        assert self.config["gpus"] and len(set(self.config["gpus"])) == len(self.config["gpus"])
        assert all(isinstance(g, int) and g >= 0 for g in self.config["gpus"])
        self.devices = ",".join(map(str, self.config["gpus"]))
        self.script = WORK / "cluster_tools/gpu_occupy.sh"

    def environment(self):
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=self.devices,
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

    def acquire(self, *, wait=False):
        lock_path = Path(self.config["lock"])
        lock = lock_path.open("a")
        ticket = None
        try:
            if not wait:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return lock
            directory = lock_path.with_suffix(".waiters")
            directory.mkdir(parents=True, exist_ok=True)
            owner = identity(os.getpid())
            enqueued = time.time_ns()
            ticket = directory / f"{enqueued}-{owner[0]}.json"
            write(
                ticket,
                dict(
                    identity=owner,
                    priority=int(os.environ.get("MIRROROPD_QUEUE_PRIORITY", "100")),
                    enqueued_ns=enqueued,
                ),
            )
            while True:
                pending = live_waiters(directory)
                if pending and pending[0][-1] == ticket:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        return lock
                    except BlockingIOError:
                        pass
                time.sleep(0.5)
        except BaseException:
            lock.close()
            raise
        finally:
            if ticket is not None:
                ticket.unlink(missing_ok=True)

    def prepare(self):
        state = self.status()
        if state.get("process_state") == "alive":
            assert state["status"]["cuda_visible_devices"] == self.devices
            subprocess.run(["bash", str(self.script), "stop", "--timeout", "30"], env=self.environment(), check=True)
        memory = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True
        )
        assert all(
            int(line.split(",")[1]) < 1024
            for line in memory.splitlines()
            if int(line.split(",")[0]) in self.config["gpus"]
        ), memory

    def restored(self, state):
        value = state.get("status", {})
        return (
            state.get("health") == "running"
            and value.get("cuda_visible_devices") == self.devices
            and set(value.get("workers", {})) == {str(i) for i in range(len(self.config["gpus"]))}
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
                                ",".join(map(str, range(len(self.config["gpus"])))),
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
                    elif state.get("status", {}).get("cuda_visible_devices") != self.devices:
                        self.config["state_dir"] = str(
                            Path(self.config["state_dir"]).with_name("uid-0-study-" + self.devices.replace(",", "-"))
                        )
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
