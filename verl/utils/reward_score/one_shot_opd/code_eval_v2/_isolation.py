# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Resource policy and client for the disposable LCB evaluator (Linux only).

This module must remain independent of verl, torch and Ray. It is also imported
by the standalone supervisor through its directory, without package imports.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

MIB = 1024**2
RESULT_LIMIT = 64 * 1024
STATUSES = {"PASS", "WRONG_ANSWER", "RUNTIME_ERROR", "SYNTAX_ERROR", "TIME_LIMIT", "MEMORY_LIMIT", "OUTPUT_LIMIT"}


class EvaluationInfrastructureError(RuntimeError):
    """An evaluator failure, never an incorrect model answer."""


class OutputLimitError(BaseException):
    pass


@dataclass(frozen=True)
class Limits:
    backend: str = "auto"
    cgroup_parent: str = ""
    memory_mib: int = 4096
    address_space_mib: int = 8192
    output_mib: int = 8
    diagnostic_mib: int = 1
    input_mib: int = 64
    max_concurrent: int = 4
    queue_timeout_s: float = 600
    startup_timeout_s: float = 30
    poll_s: float = 0.01
    slot_dir: str = ""
    audit_dir: str = ""

    def __post_init__(self):
        if self.backend not in {"auto", "rss", "cgroup"}:
            raise ValueError("LCB_BACKEND must be auto, rss or cgroup")
        for key in ["memory_mib", "address_space_mib", "output_mib", "diagnostic_mib", "input_mib", "max_concurrent"]:
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise ValueError(f"LCB {key} must be a positive integer")
        for key in ["queue_timeout_s", "startup_timeout_s", "poll_s"]:
            value = getattr(self, key)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"LCB {key} must be finite and positive")
        if self.address_space_mib < self.memory_mib:
            raise ValueError("LCB address_space_mib must be >= memory_mib")
        if self.backend == "cgroup" and not self.cgroup_parent:
            raise ValueError("MIRROROPD_LCB_CGROUP_PARENT is required for cgroup mode")

    @classmethod
    def from_env(cls):
        values = {}
        for key in cls.__dataclass_fields__:
            raw = os.environ.get("MIRROROPD_LCB_" + key.upper())
            if raw is not None:
                default = getattr(cls(), key)
                values[key] = (
                    float(raw)
                    if isinstance(default, float) or key.endswith("_s")
                    else (int(raw) if isinstance(default, int) else raw)
                )
        values.setdefault("slot_dir", f"/tmp/mirroropd-lcb-{os.getuid()}")
        return cls(**values)

    @property
    def actual_backend(self):
        return "cgroup" if self.backend == "cgroup" or (self.backend == "auto" and self.cgroup_parent) else "rss"


def append_event(directory, event):
    """Node-local bounded audit, serialized across processes; no code or test data."""
    directory = Path(directory)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (directory / "events.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = directory / "events.jsonl"
        if path.exists() and path.stat().st_size > 16 * MIB:
            path.replace(directory / "events.jsonl.1")
        with path.open("a") as stream:
            stream.write(json.dumps({"time_unix": time.time(), **event}, ensure_ascii=True) + "\n")


def try_slot(limits):
    """Hold a flock for the whole task, shared by all callers using this directory."""
    directory = Path(limits.slot_dir)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (directory / "pool.lock").open("a") as pool:
        fcntl.flock(pool, fcntl.LOCK_EX)
        capacity = directory / "capacity"
        previous = int(capacity.read_text()) if capacity.exists() else limits.max_concurrent
        if previous != limits.max_concurrent:
            held = []
            try:
                for i in range(previous):
                    handle = (directory / f"slot-{i}").open("a")
                    held.append(handle)
                    try:
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError as exc:
                        raise ValueError("Cannot change LCB node concurrency while tasks are active") from exc
            finally:
                for handle in held:
                    handle.close()
        capacity.write_text(str(limits.max_concurrent))
        for i in range(limits.max_concurrent):
            handle = (directory / f"slot-{i}").open("a")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return handle
            except BlockingIOError:
                handle.close()
    return None


class TaskCgroup:
    def __init__(self, limits, job_id):
        self.path = Path(limits.cgroup_parent) / ("mirroropd-lcb-" + job_id)
        self.path.mkdir()
        try:
            if not (self.path / "memory.max").exists() or not (self.path / "cgroup.kill").exists():
                raise OSError("Delegated cgroup v2 memory controller and cgroup.kill are required")
            for key, value in [
                ("memory.max", limits.memory_mib * MIB),
                ("memory.swap.max", 0),
                ("memory.oom.group", 1),
                ("pids.max", 64),
            ]:
                (self.path / key).write_text(str(value))
        except BaseException:
            self.path.rmdir()
            raise

    def events(self):
        return {k: int(v) for k, v in (line.split() for line in (self.path / "memory.events").read_text().splitlines())}

    def close(self):
        (self.path / "cgroup.kill").write_text("1")
        for _ in range(200):
            try:
                self.path.rmdir()
                return
            except OSError:
                time.sleep(0.01)
        raise RuntimeError(f"LCB cgroup cleanup failed: {self.path}")


def kill_group(pid):
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pid, signal.SIGKILL)


def evaluate(sample, generation, timeout=6, context=None):
    """Run the standalone supervisor and keep its stdin open as a liveness lease."""
    if sys.platform != "linux":
        raise EvaluationInfrastructureError("Isolated LCB evaluation requires Linux")
    try:
        limits = Limits.from_env()
        if type(timeout) is not int or timeout <= 0:
            raise ValueError("LCB case timeout must be a positive integer")
        if len(sample["input_output"]) > limits.input_mib * MIB:
            raise ValueError("LCB input exceeds configured request budget")
        parsed = json.loads(sample["input_output"])
        n_cases = len(parsed["inputs"])
        if not n_cases or n_cases != len(parsed["outputs"]):
            raise ValueError("LCB requires non-empty, matching inputs and outputs")
        del parsed
    except (ValueError, TypeError, KeyError) as exc:
        raise EvaluationInfrastructureError(str(exc)) from exc
    job_id = uuid.uuid4().hex
    allowed = {
        "index",
        "uid",
        "sample_id",
        "global_step",
        "step",
        "run_id",
        "experiment_name",
        "is_validation",
        "data_source",
    }
    context = {k: str(v)[:200] for k, v in (context or {}).items() if k in allowed}
    if os.environ.get("MIRROROPD_LCB_RUN_ID"):
        context.setdefault("run_id", os.environ["MIRROROPD_LCB_RUN_ID"][:200])
    with tempfile.TemporaryDirectory(prefix="mirroropd-lcb-task-") as td:
        directory = Path(td)
        request = directory / "request.json"
        with request.open("w") as stream:
            json.dump({"sample": sample, "generation": generation, "timeout": timeout}, stream)
        if request.stat().st_size > limits.input_mib * MIB:
            raise EvaluationInfrastructureError("Serialized LCB request exceeds budget")
        control = dict(limits=asdict(limits), job_id=job_id, context=context, n_cases=n_cases, timeout=timeout)
        (directory / "control.json").write_text(json.dumps(control))
        env = os.environ.copy()
        env.update(
            CUDA_VISIBLE_DEVICES="",
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            NUMEXPR_NUM_THREADS="1",
        )
        supervisor = Path(__file__).with_name("_lcb_supervisor.py")
        with subprocess.Popen(
            [sys.executable, str(supervisor), str(directory)],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            env=env,
        ) as process:
            try:
                process.wait(timeout=limits.queue_timeout_s + limits.startup_timeout_s + (timeout + 1) * n_cases + 20)
            finally:
                process.stdin.close()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # The worker has PDEATHSIG; closing the lease normally lets
                    # the supervisor reclaim the entire cgroup/process group.
                    process.kill()
                    process.wait(timeout=5)
            output = directory / "result.json"
            if process.returncode or not output.exists() or output.stat().st_size > RESULT_LIMIT:
                raise EvaluationInfrastructureError(f"LCB supervisor failed (exit={process.returncode}, job={job_id})")
            try:
                result = json.loads(output.read_text())
                if result["job_id"] != job_id or result["status"] not in STATUSES:
                    raise ValueError(result.get("message", "Invalid LCB result status"))
            except (ValueError, KeyError, TypeError) as exc:
                raise EvaluationInfrastructureError(f"LCB {job_id}: {exc}") from exc
            return result["status"] == "PASS", result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CPU-only preflight for the isolated LCB evaluator")
    parser.add_argument("--self-test", action="store_true", required=True)
    parser.parse_args()
    ok, info = evaluate(
        {"input_output": json.dumps({"inputs": ["2 3"], "outputs": ["5"]})},
        "a,b=map(int,input().split());print(a+b)",
        context={"data_source": "lcb-preflight"},
    )
    print(json.dumps(info, indent=2))
    if not ok:
        raise SystemExit("LCB preflight did not pass")
