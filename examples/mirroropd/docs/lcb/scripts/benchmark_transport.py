#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Compare the existing spawn+Manager wrapper with lightweight fresh workers.

Only a hand-written addition program is executed. A dedicated, owned cgroup
bounds the entire legacy process tree to 3 GiB and two CPU cores. No GPU work.
"""

import argparse
import contextlib
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
MIB = 1024**2


def memory():
    import psutil

    m = psutil.Process().memory_info()
    return {"rss_mib": m.rss / MIB, "vms_mib": m.vms / MIB}


def worker(args):
    (Path(args.group) / "cgroup.procs").write_text(str(os.getpid()))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (60, 65))
    sys.path.insert(0, str(ROOT))
    started = time.monotonic()
    from verl.utils.reward_score.one_shot_opd.code_eval_v2 import _lcb_eval_single, _lcb_runner

    report = {"import_s": time.monotonic() - started, "after_import": memory(), "runs": []}
    sample = {"input_output": json.dumps({"inputs": ["2 3", "-2 4"], "outputs": ["5", "2"]})}
    if args.mode == "legacy":
        for _ in range(3):
            start = time.monotonic()
            result = _lcb_eval_single((sample, "a,b=map(int,input().split());print(a+b)", 6))
            report["runs"].append({"wall_s": time.monotonic() - start, "result": result})
    else:
        # Diagnostic only: full verl import then the proposed one-line 4 GiB guard.
        original = _lcb_runner.reliability_guard
        _lcb_runner.reliability_guard = lambda: original(4 * 1024 * MIB)
        handle = open(args.result, "w")
        try:
            result = _lcb_runner.run_test(
                sample, test="a=bytearray(32*1024**2);x,y=map(int,input().split());print(x+y)", timeout=6
            )
            report["result"] = result
        except BaseException as exc:
            report["exception"] = type(exc).__name__
        json.dump(report, handle, default=str)
        handle.close()
        return
    Path(args.result).write_text(json.dumps(report, default=str))


def run(mode, parent):
    import psutil

    group = Path(parent) / ("mirroropd-lcb-transport-" + uuid.uuid4().hex)
    group.mkdir()
    try:
        if not (group / "memory.max").exists() or not (group / "cgroup.kill").exists():
            raise OSError("A delegated cgroup v2 memory controller and cgroup.kill are required")
        for key, value in [
            ("memory.max", 3 * 1024 * MIB),
            ("memory.swap.max", 0),
            ("memory.oom.group", 1),
            ("pids.max", 128),
            ("cpu.max", "200000 100000"),
        ]:
            (group / key).write_text(str(value))
    except BaseException:
        group.rmdir()
        raise
    peak = 0
    process_count = 0
    started = time.monotonic()
    p = None
    with tempfile.TemporaryDirectory(prefix="lcb-transport-") as td:
        result = Path(td) / "result.json"
        log = Path(td) / "log.txt"
        try:
            env = os.environ.copy()
            env.update(
                OPENBLAS_NUM_THREADS="1",
                OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
                NUMEXPR_NUM_THREADS="1",
                CUDA_VISIBLE_DEVICES="",
                VERL_USE_EXTERNAL_PLUGINS="none",
            )
            with log.open("w") as f:
                p = subprocess.Popen(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        "--group",
                        str(group),
                        "--mode",
                        mode,
                        "--result",
                        str(result),
                    ],
                    stdout=f,
                    stderr=f,
                    start_new_session=True,
                    env=env,
                )
                while p.poll() is None:
                    peak = max(peak, int((group / "memory.current").read_text()) / MIB)
                    with contextlib.suppress(psutil.Error):
                        process_count = max(process_count, len(psutil.Process(p.pid).children(recursive=True)) + 1)
                    if time.monotonic() - started > 90:
                        (group / "cgroup.kill").write_text("1")
                        p.kill()
                    time.sleep(0.01)
                p.wait(timeout=3)
            return {
                "mode": mode,
                "wall_s": time.monotonic() - started,
                "returncode": p.returncode,
                "sampled_cgroup_peak_mib": peak,
                "max_process_count": process_count,
                "worker": json.loads(result.read_text()) if result.exists() and result.stat().st_size else None,
                "diagnostic": log.read_text()[-3000:],
            }
        finally:
            (group / "cgroup.kill").write_text("1")
            if p is not None and p.poll() is None:
                p.wait(timeout=3)
            for _ in range(100):
                try:
                    group.rmdir()
                    break
                except OSError:
                    time.sleep(0.01)
            else:
                raise RuntimeError("owned cgroup not removed")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--group")
    p.add_argument("--mode")
    p.add_argument("--result")
    p.add_argument("--out", type=Path)
    p.add_argument("--cgroup-parent", default="/sys/fs/cgroup")
    a = p.parse_args()
    if a.worker:
        worker(a)
        return
    if a.out is None:
        p.error("--out required")
    rs = []
    for mode in ["legacy", "full_import_as4g"]:
        r = run(mode, a.cgroup_parent)
        rs.append(r)
        print(json.dumps(r), flush=True)
    a.out.write_text(json.dumps({"python": sys.version, "results": rs}, indent=2))


if __name__ == "__main__":
    main()
