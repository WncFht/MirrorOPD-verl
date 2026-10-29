# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Exclusive physical GPU4--7 queue with independent, identity-checked recovery."""

import argparse
import fcntl
import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from .common import ARMS, MODELS, OUT, REPO, WORK, dump, read

TAG = "MIRROROPD_TB_RUN"
MODULE = "examples.mirroropd.research.transfer.boundary"
OCCUPY = WORK / "cluster_tools/gpu_occupy.sh"
CONTROL = OUT / "control"


def identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return [int(pid), fields[19]] if fields[0] not in ("Z", "X") else None
    except (FileNotFoundError, ProcessLookupError):
        return None


def tagged(tag):
    result = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            if f"{TAG}={tag}".encode() in (p / "environ").read_bytes().split(b"\0"):
                record = identity(int(p.name))
                if record:
                    result.append(record)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return result


def send(record, sig):
    spec = importlib.util.spec_from_file_location("boundary_occupy_control", OCCUPY.with_name("gpu_occupy_ctl.py"))
    control = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(control)
    expected = control.identity(record[0])
    if expected and expected["start_ticks"] == record[1]:
        control.signal_process({"identity": expected}, sig)


def select_back4_state():
    directory = WORK / "cluster_tools/gpu_occupy" / socket.gethostname() / f"uid-{os.getuid()}-teacher-boundary-back4"
    dump(OUT / "occupy-management.json", dict(state_dir=str(directory), physical_gpus=[4, 5, 6, 7]))


def occupy_env():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="4,5,6,7", GPU_OCCUPY_PYTHON=sys.executable)
    env.pop(TAG, None)
    env.pop("GPU_OCCUPY_STATE_DIR", None)
    if (OUT / "occupy-management.json").exists():
        env["GPU_OCCUPY_STATE_DIR"] = read(OUT / "occupy-management.json")["state_dir"]
    return env


def occupy_status():
    result = subprocess.run(
        ["bash", str(OCCUPY), "status", "--json"],
        env=occupy_env(),
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


def recover(directory, tag):
    with (directory / "recovery.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (directory / "recovered.json").exists():
            return
        for sig, wait in [(signal.SIGTERM, 20), (signal.SIGKILL, 5)]:
            for record in tagged(tag):
                send(record, sig)
            deadline = time.monotonic() + wait
            while tagged(tag) and time.monotonic() < deadline:
                time.sleep(1)
        with (directory / "recovery.log").open("a") as log:
            while True:
                status = occupy_status()
                if restored(status):
                    time.sleep(6)
                    status = occupy_status()
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
                        env=occupy_env(),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=60,
                    )
                elif status.get("status", {}).get("cuda_visible_devices") != "4,5,6,7":
                    log.write("Unrelated occupy controller owns other GPUs; selecting separate back-four state.\n")
                    log.flush()
                    select_back4_state()
                time.sleep(5)


def queue(jobs, directory, tag):
    pending = list(jobs)
    running = []
    while pending or running:
        used = {g for _, _, gpus, _ in running for g in gpus}
        for job in list(pending):
            name, command, width = job
            free = [g for g in [4, 5, 6, 7] if g not in used]
            if len(free) < width:
                continue
            gpus = free[:width]
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=",".join(map(str, gpus)),
                OMP_NUM_THREADS="4",
                OPENBLAS_NUM_THREADS="4",
                TOKENIZERS_PARALLELISM="false",
                HF_HUB_OFFLINE="1",
                VLLM_WORKER_MULTIPROC_METHOD="spawn",
            )
            env[TAG] = tag
            log = (directory / f"{name}.log").open("a")
            process = subprocess.Popen(
                [sys.executable, "-m", f"{MODULE}.{command[0]}", *command[1:]],
                cwd=REPO,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            dump(directory / f"{name}-launch.json", dict(identity=identity(process.pid), gpus=gpus, command=command))
            running.append((name, process, gpus, log))
            pending.remove(job)
            used.update(gpus)
        for item in list(running):
            name, process, gpus, log = item
            code = process.poll()
            if code is not None:
                log.close()
                running.remove(item)
                dump(directory / f"{name}-exit.json", dict(code=code, time=time.time(), gpus=gpus))
                if code:
                    raise RuntimeError(f"{name} failed ({code}); see {directory / name}.log")
        dump(
            directory / "queue.json",
            dict(running=[x[0] for x in running], pending=[x[0] for x in pending], time=time.time()),
        )
        if running:
            time.sleep(5)


def phases(phase, directory, tag):
    if phase in ["all", "banks"]:
        queue(
            [
                (f"bank-{s}-{i}", ["native", "generate", s, "--shard", str(i)], 1)
                for s in ["r1", "r7"]
                for i in range(4)
            ],
            directory,
            tag,
        )
    if phase in ["all", "audit"]:
        queue([(f"audit-{n}", ["native", "audit", n], 1) for n in MODELS], directory, tag)
    if phase in ["all", "cache"]:
        queue([(f"cache-{a}", ["native", "cache", t, "--student", s], 1) for a, (s, t) in ARMS.items()], directory, tag)
    if phase in ["all", "smoke"]:
        queue(
            [
                ("smoke-small", ["train", "TB1-J", "--steps", "1", "--smoke"], 1),
                ("smoke-large", ["train", "TB2-P7", "--steps", "1", "--smoke"], 2),
            ],
            directory,
            tag,
        )
    if phase in ["all", "baseline"]:
        jobs = [(f"baseline-{n}", ["evaluate", n, "--port", str(18900 + i)], 1) for i, n in enumerate(MODELS)]
        for i, (name, audit) in enumerate(read(OUT / "interface-audit.json").items()):
            if audit["template_different_indices"]:
                jobs.append(
                    (f"baseline-native-{name}", ["evaluate", name, "--port", str(18920 + i), "--template", "native"], 1)
                )
        queue(jobs, directory, tag)
    if phase in ["all", "train"]:
        for prefix in ["TB1", "TB2"]:
            queue(
                [
                    (f"train-{a}", ["train", a], 2 if s == "r7" else 1)
                    for a, (s, t) in ARMS.items()
                    if a.startswith(prefix)
                ],
                directory,
                tag,
            )

    if phase in ["all", "cross"]:
        queue([(f"cross-{arm}", ["cross", arm], 1) for arm in ["TB1-P", "TB2-P7", "TB2-L1", "TB2-L7"]], directory, tag)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["run", "guard"])
    parser.add_argument(
        "--phase", default="all", choices=["all", "banks", "audit", "cache", "smoke", "baseline", "train", "cross"]
    )
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--owner", nargs=2)
    parser.add_argument("--tag")
    args = parser.parse_args()
    if args.action == "guard":
        owner = [int(args.owner[0]), args.owner[1]]
        dump(args.directory / "guard-ready.json", dict(owner=owner, guard=identity(os.getpid())))
        while identity(owner[0]) == owner:
            if (args.directory / "recovered.json").exists():
                return
            time.sleep(5)
        recover(args.directory, args.tag)
        return
    CONTROL.mkdir(parents=True, exist_ok=True)
    latest = CONTROL / "latest.json"
    if latest.exists():
        previous = read(latest)
        if identity(previous["owner"][0]) == previous["owner"]:
            raise RuntimeError("The previous boundary supervisor is still active")
    lease = (CONTROL / "gpu-lease.lock").open("a")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    directory = CONTROL / f"{time.strftime('%Y%m%d-%H%M%S')}-{args.phase}"
    directory.mkdir()
    owner = identity(os.getpid())
    tag = directory.name
    dump(CONTROL / "latest.json", dict(directory=str(directory), owner=owner, phase=args.phase))
    with (directory / "guard.log").open("a") as log:
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                f"{MODULE}.launch",
                "guard",
                "--directory",
                str(directory),
                "--owner",
                str(owner[0]),
                owner[1],
                "--tag",
                tag,
            ],
            cwd=REPO,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    while not (directory / "guard-ready.json").exists():
        time.sleep(0.1)

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGHUP]:
        signal.signal(sig, interrupted)
    try:
        status = occupy_status()
        if status.get("process_state") == "alive" and status["status"]["cuda_visible_devices"] != "4,5,6,7":
            select_back4_state()
            status = occupy_status()
        if status.get("process_state") == "alive":
            assert status["status"]["cuda_visible_devices"] == "4,5,6,7", status
            subprocess.run(["bash", str(OCCUPY), "stop", "--timeout", "30"], env=occupy_env(), check=True)
        free = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True
        )
        assert all(
            int(line.split(",")[1]) < 1024 for line in free.splitlines() if int(line.split(",")[0]) in [4, 5, 6, 7]
        ), free
        phases(args.phase, directory, tag)
        dump(directory / "complete.json", dict(phase=args.phase, time=time.time()))
    except BaseException as error:
        dump(directory / "error.json", dict(error=repr(error), time=time.time()))
        raise
    finally:
        for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGHUP]:
            signal.signal(sig, signal.SIG_IGN)
        recover(directory, tag)


if __name__ == "__main__":
    main()
