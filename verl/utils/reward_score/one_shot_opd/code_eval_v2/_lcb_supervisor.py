# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Standalone task supervisor. Do not import the verl package here."""

import contextlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil
from _isolation import MIB, RESULT_LIMIT, STATUSES, Limits, TaskCgroup, append_event, kill_group, try_slot


def lease_alive():
    try:
        return os.read(0, 1) != b""
    except BlockingIOError:
        return True


def supervise(directory):
    control = json.loads((directory / "control.json").read_text())
    limits = Limits(**control["limits"])
    job_id = control["job_id"]
    started = time.monotonic()
    result = dict(
        job_id=job_id,
        status="INFRASTRUCTURE_ERROR",
        backend=limits.actual_backend,
        memory_limit_mib=limits.memory_mib,
        address_space_mib=limits.address_space_mib,
        context=control["context"],
        supervisor_pid=os.getpid(),
    )
    slot = None
    cgroup = None
    worker = None
    peak_rss = 0
    peak_charge = 0
    os.set_blocking(0, False)
    try:
        while slot is None:
            if not lease_alive():
                raise RuntimeError("Caller exited while queued")
            slot = try_slot(limits)
            if time.monotonic() - started > limits.queue_timeout_s:
                raise RuntimeError("LCB execution slot queue timeout")
            if slot is None:
                time.sleep(min(limits.poll_s, 0.05))
        result["queue_s"] = time.monotonic() - started
        if limits.actual_backend == "cgroup":
            cgroup = TaskCgroup(limits, job_id)
        result["cgroup"] = str(cgroup.path) if cgroup else None
        command = [
            sys.executable,
            str(Path(__file__).with_name("_lcb_worker.py")),
            str(directory),
            str(os.getpid()),
            str(cgroup.path) if cgroup else "",
        ]
        with (directory / "diagnostic.txt").open("wb") as diagnostic:
            worker = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=diagnostic, stderr=diagnostic, start_new_session=True
            )
            result["worker_pid"] = worker.pid
            append_event(limits.audit_dir or limits.slot_dir, {"event": "started", **result, "status": "RUNNING"})
            launch = time.monotonic()
            ready_at = None
            reason = None
            execution_budget = (control["timeout"] + 1) * control["n_cases"] + 5
            while worker.poll() is None:
                if ready_at is None and (directory / "ready").exists():
                    ready_at = time.monotonic()
                rss = 0
                with contextlib.suppress(psutil.NoSuchProcess):
                    process = psutil.Process(worker.pid)
                    for child in [process, *process.children(recursive=True)]:
                        with contextlib.suppress(psutil.NoSuchProcess):
                            rss += child.memory_info().rss
                peak_rss = max(peak_rss, rss)
                if cgroup:
                    peak_charge = max(peak_charge, int((cgroup.path / "memory.current").read_text()))
                if not lease_alive():
                    reason = "CALLER_EXITED"
                elif ready_at is None and time.monotonic() - launch > limits.startup_timeout_s:
                    reason = "STARTUP_TIMEOUT"
                elif ready_at is not None and time.monotonic() - ready_at > execution_budget:
                    reason = "TIME_LIMIT"
                elif limits.actual_backend == "rss" and rss > limits.memory_mib * MIB:
                    reason = "MEMORY_LIMIT"
                if reason:
                    if cgroup:
                        (cgroup.path / "cgroup.kill").write_text("1")
                    kill_group(worker.pid)
                    break
                time.sleep(limits.poll_s)
            worker.wait(timeout=5)
            ready = (directory / "ready").exists()
            if cgroup:
                result["memory_events"] = cgroup.events()
                if result["memory_events"].get("oom_kill", 0):
                    reason = "MEMORY_LIMIT"
            if worker.returncode == -signal.SIGXFSZ:
                reason = "OUTPUT_LIMIT"
            elif worker.returncode == -signal.SIGXCPU:
                reason = "TIME_LIMIT"
            result["returncode"] = worker.returncode
            if reason:
                if not ready or reason not in STATUSES:
                    raise RuntimeError(f"Evaluator stopped outside answer execution: {reason}")
                result["status"] = reason
            elif worker.returncode == 0:
                output = directory / "worker-result.json"
                if not output.exists() or output.stat().st_size > RESULT_LIMIT:
                    raise RuntimeError("Missing or oversized worker result")
                details = json.loads(output.read_text())
                if details.get("job_id") != job_id or details.get("status") not in STATUSES:
                    raise RuntimeError(details.get("message", "Invalid worker result"))
                result.update(
                    {
                        k: details[k]
                        for k in ["status", "results", "error_type", "message", "startup_s", "judge_s"]
                        if k in details
                    }
                )
            elif ready and worker.returncode in {-signal.SIGSEGV, -signal.SIGABRT, -signal.SIGILL, -signal.SIGFPE}:
                result["status"] = "RUNTIME_ERROR"
            else:
                raise RuntimeError(f"Unexpected evaluator exit: {worker.returncode}")
    except Exception as exc:
        result.update(status="INFRASTRUCTURE_ERROR", message=f"{type(exc).__name__}: {str(exc)[:500]}")
    finally:
        try:
            if worker is not None:
                kill_group(worker.pid)
                worker.wait(timeout=5)
            if cgroup is not None:
                cgroup.close()
        except Exception as exc:
            result.update(status="INFRASTRUCTURE_ERROR", message=f"Cleanup failed: {exc}")
        result.update(
            wall_s=time.monotonic() - started,
            peak_rss_mib=peak_rss / MIB,
            sampled_cgroup_peak_mib=peak_charge / MIB if cgroup else None,
        )
        try:
            append_event(limits.audit_dir or limits.slot_dir, {"event": "finished", **result})
        except Exception as exc:
            result.update(status="INFRASTRUCTURE_ERROR", message=f"Audit write failed: {exc}")
        finally:
            if slot is not None:
                slot.close()
        (directory / "result.json").write_text(json.dumps(result))
        if not lease_alive():
            shutil.rmtree(directory)


if __name__ == "__main__":
    supervise(Path(sys.argv[1]))
