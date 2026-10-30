# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Exercise real flock ownership with fake NVML responses; never allocate a GPU."""

import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from examples.mirroropd.benchmarks.fusion import resource_guard as guard


@pytest.mark.parametrize("value", ["", "0,0", "7,7", "8", "-1", "0,", "0, 1", "0;1", "01"])
def test_reject_invalid_allocations(value):
    with pytest.raises(ValueError):
        guard.parse_gpus(value)


def test_preserve_requested_logical_order():
    assert guard.parse_gpus("5,4,0") == [5, 4, 0]


@pytest.fixture
def local(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    calls = []

    def nvidia(*args):
        calls.append(args)
        if args[0].startswith("--id="):
            return guard.GPU_UUIDS[int(args[0].split("=")[1])]
        return ""

    monkeypatch.setattr(guard, "nvidia_smi", nvidia)
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[2]))
    return repo, calls


def test_parent_owns_locks_and_children_verify_without_inheriting_them(local):
    repo, calls = local
    output = repo / "observed.json"
    code = """
import json, os
from pathlib import Path
from examples.mirroropd.benchmarks.fusion.resource_guard import verify_allocation
workspace = Path.cwd().parent
assert verify_allocation(workspace, os.environ['FUSION_GPUS']) == [4,5]
lock_descriptors = []
for descriptor in Path('/proc/self/fd').iterdir():
    try:
        path = descriptor.resolve(strict=True)
    except FileNotFoundError:
        continue
    if path.name in {'gpu4.lock','gpu5.lock'}:
        lock_descriptors.append(str(descriptor))
assert not lock_descriptors
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
assert verify_allocation(workspace, '4,5', visible='5') == [4,5]
try:
    verify_allocation(workspace, '4,5', visible='0')
except ValueError:
    pass
else:
    raise AssertionError('foreign device accepted')
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5'
Path('observed.json').write_text(json.dumps({k:os.environ[k] for k in
    ['CUDA_VISIBLE_DEVICES','STUDENT_GPUS','RAY_ADDRESS','FUSION_RESOURCE_PARENT','SMOKE_RAY_TMPDIR']}))
"""
    assert guard.run(repo, "4,5", [sys.executable, "-c", code]) == 0
    result = json.loads(output.read_text())
    assert result["CUDA_VISIBLE_DEVICES"] == "4,5"
    assert result["STUDENT_GPUS"] == "2" and result["RAY_ADDRESS"] == "local"
    assert result["FUSION_RESOURCE_PARENT"] == str(os.getpid())
    for gpu in [4, 5]:
        with guard.lock_path(repo.parent, gpu).open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert len(calls) == 4  # Both UUIDs, initial availability, and final cleanup.


def test_original_gpu6_lock_blocks_new_launcher(local):
    repo, _ = local
    path = guard.lock_path(repo.parent, 6)
    path.parent.mkdir(parents=True)
    with path.open("a") as original:
        fcntl.flock(original, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="already locked"):
            guard.run(repo, "6", [sys.executable, "-c", "raise AssertionError('must not launch')"])


@pytest.mark.parametrize("failure", ["uuid", "busy", "query"])
def test_refuses_wrong_uuid_busy_and_failed_nvml(local, monkeypatch, failure):
    repo, _ = local

    def nvidia(*args):
        if args[0].startswith("--id="):
            return "WRONG" if failure == "uuid" else guard.GPU_UUIDS[0]
        if failure == "query":
            raise subprocess.CalledProcessError(1, "nvidia-smi")
        return guard.GPU_UUIDS[0]

    monkeypatch.setattr(guard, "nvidia_smi", nvidia)
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        guard.run(repo, "0", [sys.executable, "-c", "raise AssertionError('must not launch')"])


def test_child_exit_status_propagates(local):
    repo, _ = local
    assert guard.run(repo, "1", [sys.executable, "-c", "raise SystemExit(17)"]) == 17


def test_unowned_internal_worker_is_rejected(local, monkeypatch):
    repo, _ = local
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.delenv("FUSION_ALLOCATION", raising=False)
    with pytest.raises(ValueError, match="receipt"):
        guard.verify_allocation(repo.parent, "0")


def test_cli_accepts_options_after_action_without_nvml():
    result = subprocess.run(
        [sys.executable, guard.__file__, "validate", "--gpus", "0,1,2,3,4,5,6,7"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("allocation", ["0,1", "4,5", "7,6"])
def test_topology_uses_allocated_pair_in_requested_order(allocation):
    from examples.mirroropd.benchmarks.fusion.probe_vllm_topology import assigned_devices

    first, second = allocation.split(",")
    assert assigned_devices(allocation, 2, 1, 0) == allocation
    assert assigned_devices(allocation, 1, 2, 0) == first
    assert assigned_devices(allocation, 1, 2, 1) == second
    with pytest.raises(ValueError):
        assigned_devices(allocation, 1, 2, 2)
    with pytest.raises(ValueError):
        assigned_devices(first, 1, 2)


@pytest.mark.parametrize("stop_signal", [signal.SIGTERM, signal.SIGINT])
def test_termination_keeps_lock_until_cpu_initializing_child_exits(tmp_path, stop_signal):
    repo = tmp_path / "repo"
    repo.mkdir()
    binary = tmp_path / "bin"
    binary.mkdir()
    nvml = binary / "nvidia-smi"
    nvml.write_text(
        "#!"
        + sys.executable
        + "\nimport sys\n"
        + "if sys.argv[1] == '--id=0': print("
        + repr(guard.GPU_UUIDS[0])
        + ")\n"
    )
    nvml.chmod(0o755)
    ready, stopping = repo / "ready", repo / "stopping"
    code = """
import os, signal, time
from pathlib import Path

def stop(sig, frame):
    Path('stopping').write_text(str(sig))
    time.sleep(1.5)
    raise SystemExit(0)

signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
Path('ready').write_text(str(os.getpid()))
while True:
    time.sleep(0.1)
"""
    env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ["PATH"])
    command = [
        sys.executable,
        guard.__file__,
        "run",
        "--repo",
        str(repo),
        "--gpus",
        "0",
        "--",
        sys.executable,
        "-c",
        code,
    ]
    parent = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            assert parent.poll() is None
            time.sleep(0.02)
        assert ready.exists()
        os.kill(parent.pid, stop_signal)
        while not stopping.exists() and time.monotonic() < deadline:
            assert parent.poll() is None
            time.sleep(0.02)
        assert stopping.exists()
        with guard.lock_path(tmp_path, 0).open("a") as probe:
            with pytest.raises(BlockingIOError):
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        second = subprocess.run(command, env=env, capture_output=True, text=True, timeout=5)
        assert second.returncode == 2 and "already locked" in second.stderr
        _, errors = parent.communicate(timeout=10)
        assert parent.returncode == 128 + stop_signal, errors
        with guard.lock_path(tmp_path, 0).open("a") as probe:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        if parent.poll() is None:
            parent.terminate()
            parent.wait(timeout=10)
