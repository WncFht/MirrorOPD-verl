# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Protect approved local GPUs with parent-owned locks and physical UUID checks."""

import argparse
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

GPU_UUIDS = {
    0: "GPU-854ebda7-1637-a3ec-a50b-3f413dc80232",
    1: "GPU-f13bcddf-ca2b-7f2f-0d90-51ba37c8b5a2",
    2: "GPU-b3243d5c-7894-0582-9734-cc9785510459",
    3: "GPU-13b99baf-8482-0b3c-ba97-59befde42be8",
    4: "GPU-fa6b674e-60b0-c680-882d-08e92e9226e1",
    5: "GPU-aa3853a1-b2f6-f974-aec2-efa4b4455205",
    6: "GPU-44bf7e96-2d92-fcfd-7da0-471d76517118",
    7: "GPU-76f3510f-d1e3-46ec-7cd6-a339d3a9d933",
}


def parse_gpus(value):
    if not re.fullmatch(r"[0-7](?:,[0-7])*", value):
        raise ValueError("Use an explicit comma-separated allocation within physical GPUs 0-7")
    devices = [int(part) for part in value.split(",")]
    if len(set(devices)) != len(devices):
        raise ValueError("Duplicate physical GPU in allocation")
    return devices


def process_identity(pid):
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return {"pid": pid, "start_ticks": fields[19]}, int(fields[1])


def lock_path(workspace, gpu):
    # Same names as the frozen GPU6/7 launcher and its handoff guards.
    return workspace / "runs/vllm-fusion-20261029" / f"gpu{gpu}.lock"


def verify_allocation(workspace, value, *, visible=None):
    devices = parse_gpus(value)
    expected_visible = value if visible is None else visible
    if not set(parse_gpus(expected_visible)) <= set(devices):
        raise ValueError("Worker devices are outside the protected allocation")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_visible:
        raise ValueError("Visible devices differ from the protected allocation")
    receipt = json.loads(os.environ.get("FUSION_ALLOCATION", "null"))
    if not isinstance(receipt, dict) or receipt.get("devices") != devices:
        raise ValueError("Missing or mismatched allocation receipt")
    guardian = receipt["guardian"]
    pid = guardian["pid"]
    if str(pid) != os.environ.get("FUSION_RESOURCE_PARENT") or process_identity(pid)[0] != guardian:
        raise ValueError("GPU allocation guardian identity changed")
    ancestor = os.getpid()
    while ancestor > 1 and ancestor != pid:
        _, ancestor = process_identity(ancestor)
    if ancestor != pid:
        raise ValueError("GPU allocation guardian is not an ancestor of this worker")
    descriptors = receipt["descriptors"]
    if set(descriptors) != {str(gpu) for gpu in devices}:
        raise ValueError("Incomplete GPU lock ownership receipt")
    for gpu in devices:
        path = lock_path(workspace, gpu).resolve()
        descriptor = descriptors[str(gpu)]
        if type(descriptor) is not int or descriptor < 0:
            raise ValueError("Invalid guardian file descriptor")
        if Path(f"/proc/{pid}/fd/{descriptor}").resolve(strict=True) != path:
            raise ValueError(f"GPU {gpu} lock descriptor no longer belongs to the guardian")
        with path.open("a") as probe:
            try:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            fcntl.flock(probe, fcntl.LOCK_UN)
            raise ValueError(f"GPU {gpu} allocation lock is not held")
    return devices


def nvidia_smi(*arguments):
    return subprocess.check_output(["nvidia-smi", *arguments], text=True).strip()


def allocated_busy(devices):
    current = set(nvidia_smi("--query-compute-apps=gpu_uuid", "--format=csv,noheader").splitlines())
    return any(GPU_UUIDS[gpu] in current for gpu in devices)


def run(repo, value, command):
    devices = parse_gpus(value)
    if not command:
        raise ValueError("An explicit command is required")
    repo = repo.resolve()
    workspace = repo.parent
    locks = []
    try:
        for gpu in sorted(devices):
            path = lock_path(workspace, gpu)
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            locks.append((gpu, descriptor))
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError(f"GPU {gpu} allocation is already locked") from error
            actual = nvidia_smi(f"--id={gpu}", "--query-gpu=uuid", "--format=csv,noheader")
            if actual != GPU_UUIDS[gpu]:
                raise ValueError(f"GPU {gpu} UUID changed; refusing allocation")
        if allocated_busy(devices):
            raise ValueError("An allocated GPU has an existing compute process; refusing to interfere")
        identity, _ = process_identity(os.getpid())
        receipt = dict(guardian=identity, devices=devices, descriptors={str(g): fd for g, fd in locks})
        env = dict(os.environ)
        env.update(
            FUSION_GPUS=value,
            CUDA_VISIBLE_DEVICES=value,
            STUDENT_GPUS=str(len(devices)),
            RAY_ADDRESS="local",
            FUSION_RESOURCE_PARENT=str(os.getpid()),
            FUSION_ALLOCATION=json.dumps(receipt, separators=(",", ":")),
            VERL_ROLLOUT_IPC_NAMESPACE=f"fusion-{value.replace(',', '-')}-{os.getpid()}-{int(time.time())}",
            VLLM_USE_V2_MODEL_RUNNER="0",
            VLLM_PLUGINS="mirroropd_fusion",
            VLLM_WORKER_MULTIPROC_METHOD="spawn",
            OMP_NUM_THREADS="1",
            PYTHONPYCACHEPREFIX="/root/.cache/mirroropd-fusion/pycache",
            PYTHON_BIN=str(workspace / "envs/mirroropd-fusion/bin/python"),
            PYTHONPATH=str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
            VERL_USE_UV="0",
            SMOKE_RAY_TMPDIR=f"/tmp/mirroropd-fusion-{value.replace(',', '-')}-{os.getpid()}",
            LD_LIBRARY_PATH=str(workspace / "cache/fusion-runtime/cuda-compat-12-9/usr/local/cuda-12.9/compat")
            + (os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""),
        )
        env.pop("PYTHONDONTWRITEBYTECODE", None)
        # Only this guardian holds locks. Long-lived monitors or Ray children
        # cannot accidentally keep a released allocation locked through inheritance.
        child = None
        terminating = []
        forwarded = False
        previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}

        def forward_stop(signum, frame=None):
            nonlocal forwarded
            if not terminating:
                terminating.append((signum, time.monotonic()))
            if child is not None and not forwarded:
                forwarded = True
                try:
                    os.killpg(child.pid, terminating[0][0])
                except ProcessLookupError:
                    pass

        for sig in previous_handlers:
            signal.signal(sig, forward_stop)
        try:
            if not terminating:
                child = subprocess.Popen(command, cwd=repo, env=env, close_fds=True, start_new_session=True)
                if terminating:
                    forward_stop(terminating[0][0])
                killed = False
                while True:
                    try:
                        returncode = child.wait(timeout=0.5)
                        break
                    except subprocess.TimeoutExpired:
                        if terminating and not killed and time.monotonic() - terminating[0][1] >= 30:
                            # Escalation is restricted to this command's own session group.
                            try:
                                os.killpg(child.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                            killed = True
            else:
                returncode = 128 + terminating[0][0]
            status = returncode if returncode >= 0 else 128 - returncode
            for _ in range(120):
                if not allocated_busy(devices):
                    return 128 + terminating[0][0] if terminating else status
                time.sleep(0.5)
            print("Allocated GPUs remain occupied after driver exit; refusing the next stage", file=sys.stderr)
            return 128 + terminating[0][0] if terminating else status or 2
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)

    finally:
        for _, descriptor in reversed(locks):
            os.close(descriptor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "verify", "validate"])
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--gpus", required=True)
    argv = sys.argv[1:]
    command = []
    if "--" in argv:
        split = argv.index("--")
        command, argv = argv[split + 1 :], argv[:split]
    args = parser.parse_args(argv)
    try:
        if args.action == "validate":
            parse_gpus(args.gpus)
            return 0
        if args.action == "verify":
            verify_allocation(args.workspace.resolve(), args.gpus)
            return 0
        return run(args.repo, args.gpus, command)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        print(f"GPU allocation refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
