#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded Linux CPU experiments, not a production sandbox or a training patch.

The controller only executes the synthetic and hand-written fixtures below.
Each worker has a 1 GiB emergency address-space bound, CPU limit, process-group
watchdog and bounded diagnostic files. Optional cgroups contain only our workers.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import importlib
import io
import json
import mmap
import os
import platform
import resource
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import types
import uuid
from pathlib import Path

MIB = 1024**2
ROOT = Path(__file__).resolve().parents[4]
JUDGE = ROOT / "verl/utils/reward_score/one_shot_opd/code_eval_v2"
POLICIES = ("baseline", "as256", "rss256_output", "cgroup256_output")


def import_judge():
    # Import the exact runner without importing the trainer or torch.
    package = types.ModuleType("_lcb_isolation_audit")
    package.__path__ = [str(JUDGE)]
    sys.modules[package.__name__] = package
    return importlib.import_module(package.__name__ + "._lcb_runner")


def stdio(name, code, inputs, outputs, **extra):
    return dict(
        name=name, code=code, sample={"input_output": json.dumps({"inputs": inputs, "outputs": outputs})}, **extra
    )


def functional(name, code, inputs=None, outputs=None, **extra):
    return dict(
        name=name,
        code=code,
        sample={
            "input_output": json.dumps({"inputs": inputs or ["0"], "outputs": outputs or ["1"], "fn_name": "solve"})
        },
        **extra,
    )


def fixtures():
    return [
        stdio("sum", "a,b=map(int,input().split());print(a+b)", ["2 3", "-2 4"], ["5", "2"]),
        functional(
            "numpy", "def solve(x):\n import numpy as np\n return int(np.arange(10000).sum())", outputs=["49995000"]
        ),
        functional(
            "pandas", "def solve(x):\n import pandas as pd\n return int(pd.Series([1,2,3]).sum())", outputs=["6"]
        ),
        stdio("wrong_answer", "print(0)", ["1"], ["1"]),
        stdio("runtime_error", "print(1/0)", ["1"], ["1"]),
        stdio("syntax_error", "def broken(", ["1"], ["1"]),
        functional(
            "allocation_fast", "def solve(x):\n a=bytearray(384*1024**2)\n import time\n time.sleep(0.15)\n return 1"
        ),
        functional(
            "allocation_slow",
            "def solve(x):\n import time\n a=[]\n for _ in range(24):\n"
            "  a.append(bytearray(16*1024**2))\n  time.sleep(0.02)\n return 1",
        ),
        stdio("output_flood", 'for _ in range(1024): print("x"*65536)', ["1"], ["1"]),
        functional("spin", "def solve(x):\n while True: pass"),
        functional(
            "ignore_alarm",
            "def solve(x):\n import signal\n signal.signal(signal.SIGALRM,signal.SIG_IGN)\n while True: pass",
            deadline=3,
        ),
        functional("virtual_reservation", "def solve(x):\n a=bytearray(8*1024**2)\n return 1", reserve_mib=512),
        stdio("binary_stdio", "import sys\na,b=map(int,sys.stdin.buffer.read().split());print(a+b)", ["2 3"], ["5"]),
    ]


def real_cases(path):
    import pyarrow.parquet as pq

    rows = pq.read_table(path).to_pylist()
    codes = {
        0: "x=int(input());print(sum(i*j for i in range(1,10) for j in range(1,10) if i*j!=x))",
        2: "a,b=map(int,input().split());print((a+b)**2)",
        3: """def count(n):
 if n<10:return 0
 s=str(n);h=int(s[0]);L=len(s)
 ans=sum(d**(k-1) for k in range(2,L) for d in range(1,10))
 ans+=sum(d**(L-1) for d in range(1,h))
 for i in range(1,L):
  d=int(s[i]);ans+=min(d,h)*h**(L-i-1)
  if d>=h:return ans
 return ans+1
l,r=map(int,input().split());print(count(r)-count(l-1))""",
        4: """n=int(input());a=list(map(int,input().split()))
d=[0]*(n+1);active=0;ans=[]
for i,v in enumerate(a):
 active+=d[i];v+=active;ans.append(max(0,v-(n-1-i)))
 if v>0:
  active+=1
  if i+v+1<n:d[i+v+1]-=1
print(*ans)""",
    }
    cases = []
    counts, inp, out, payloads = [], [], [], []
    for row in rows:
        ts = json.loads(row["reward_model"]["ground_truth"])
        counts.append(len(ts))
        inp.extend(len(str(t["input"]).encode()) for t in ts)
        out.extend(len(str(t["output"]).encode()) for t in ts)
        payloads.append(len(row["reward_model"]["ground_truth"].encode()))
    for i, code in codes.items():
        ts = json.loads(rows[i]["reward_model"]["ground_truth"])
        cases.append(
            stdio(
                "lcb_v6_" + str(i),
                code,
                [t["input"] for t in ts],
                [t["output"] for t in ts],
                deadline=20,
                case_timeout=6,
            )
        )
    stats = dict(
        rows=len(rows),
        cases=sum(counts),
        case_count_min=min(counts),
        case_count_median=statistics.median(counts),
        case_count_max=max(counts),
        max_input_bytes=max(inp),
        max_output_bytes=max(out),
        max_problem_payload_bytes=max(payloads),
        total_payload_bytes=sum(payloads),
        outer_timeout_max_s=7 * max(counts) + 5,
        hand_written_reference_indices=list(codes),
    )
    return cases, stats


class OutputLimit(BaseException):
    pass


class BoundedStringIO(io.StringIO):
    limit = 8 * MIB

    def __init__(self):
        super().__init__()
        self.written_bytes = 0

    def write(self, value):
        self.written_bytes += len(value.encode("utf-8"))
        if self.written_bytes > self.limit:
            raise OutputLimit("output budget exceeded")
        return super().write(value)


def worker(args):
    # Set bounds inside the fresh interpreter, never preexec_fn in a threaded parent.
    if args.cgroup:
        (Path(args.cgroup) / "cgroup.procs").write_text(str(os.getpid()))
    resource.setrlimit(resource.RLIMIT_AS, (1024 * MIB, 1024 * MIB))
    resource.setrlimit(resource.RLIMIT_CPU, (25, 26))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MIB, MIB))
    result_file = open(args.result, "w")
    original_stdout = sys.stdout
    proc_status = Path("/proc/self/status")
    # Read fixture and import prelude under the external experimental bound.
    case = json.loads(Path(args.worker).read_text())
    runner = import_judge()
    start = time.monotonic()
    exec(runner.import_string, {})
    import_s = time.monotonic() - start
    reservation = mmap.mmap(-1, case.get("reserve_mib", 0) * MIB) if case.get("reserve_mib") else None
    before = proc_status.read_text()
    if args.policy == "as256":
        original_guard = runner.reliability_guard
        runner.reliability_guard = lambda: original_guard(256 * MIB)
    if args.policy.endswith("_output"):
        expected = json.loads(case["sample"]["input_output"])["outputs"]
        BoundedStringIO.limit = max(8 * MIB, 4 * max(len(str(x).encode()) for x in expected) + 65536)
        original_capture = runner.Capturing

        class BoundedCapture(original_capture):
            def __enter__(self):
                self._stdout = sys.stdout
                sys.stdout = self._stringio = BoundedStringIO()
                self._stringio.close = lambda: None
                return self

        runner.Capturing = BoundedCapture
    started = time.monotonic()
    try:
        scores, metadata = runner.run_test(case["sample"], test=case["code"], timeout=case.get("case_timeout", 1))
        status = "PASS" if scores and all(x is True for x in scores) else "FAIL"
        # Keep upstream reason codes; do not reinterpret every failure as OOM.
        info = dict(scores=scores, metadata=metadata)
    except OutputLimit:
        status, info = "OUTPUT_LIMIT", {}
    except BaseException as exc:
        status, info = "RUNNER_EXCEPTION", {"exception": type(exc).__name__, "message": str(exc)[:300]}
    sys.stdout = original_stdout
    after = proc_status.read_text()
    hwm = next(int(line.split()[1]) / 1024 for line in after.splitlines() if line.startswith("VmHWM:"))
    result = dict(
        status=status,
        details=info,
        import_s=import_s,
        judge_s=time.monotonic() - started,
        proc_hwm_mib=hwm,
        rusage_maxrss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        pre_guard_status={
            k: v.strip()
            for line in before.splitlines()
            if ":" in line
            for k, v in [line.split(":", 1)]
            if k in ["VmRSS", "VmSize"]
        },
        as_limit_bytes=resource.getrlimit(resource.RLIMIT_AS)[0],
    )
    # The generated code only runs in this disposable child.
    json.dump(result, result_file, default=str)
    result_file.close()
    if reservation is not None:
        reservation.close()


def make_cgroup(parent):
    group = Path(parent) / ("mirroropd-lcb-audit-" + uuid.uuid4().hex)
    group.mkdir()
    try:
        if not (group / "memory.max").exists() or not (group / "cgroup.kill").exists():
            raise OSError("A delegated cgroup v2 memory controller and cgroup.kill are required")
        for name, value in [
            ("memory.max", 256 * MIB),
            ("memory.swap.max", 0),
            ("pids.max", 32),
            ("cpu.max", "100000 100000"),
            ("memory.oom.group", 1),
        ]:
            (group / name).write_text(str(value))
    except BaseException:
        group.rmdir()
        raise
    return group


def rss_tree(pid):
    import psutil

    try:
        p = psutil.Process(pid)
        ps = [p] + p.children(recursive=True)
    except psutil.Error:
        return 0, 0
    rss = 0
    for p in ps:
        with contextlib.suppress(psutil.Error):
            rss += p.memory_info().rss
    return rss, len(ps)


def kill_group(pid):
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pid, signal.SIGKILL)


def run_one(case, policy, cgroup_parent, poll_s=0.005):
    full_started = time.monotonic()
    record = None
    group = make_cgroup(cgroup_parent) if policy.startswith("cgroup") else None
    with tempfile.TemporaryDirectory(prefix="lcb-isolation-") as td:
        work = Path(td)
        (work / "case.json").write_text(json.dumps(case))
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            str(work / "case.json"),
            "--result",
            str(work / "result.json"),
            "--policy",
            policy,
        ]
        if group:
            command += ["--cgroup", str(group)]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
            env[key] = "1"
        started = time.monotonic()
        peak = 0
        processes = 0
        reason = None
        cg = {}
        cg_peak = 0
        try:
            with (work / "diagnostic.txt").open("w") as diagnostic:
                p = subprocess.Popen(command, stdout=diagnostic, stderr=diagnostic, start_new_session=True, env=env)
                try:
                    while p.poll() is None:
                        rss, n = rss_tree(p.pid)
                        peak = max(peak, rss)
                        processes = max(processes, n)
                        if group:
                            cg_peak = max(cg_peak, int((group / "memory.current").read_text()))
                        if policy == "rss256_output" and rss > 256 * MIB:
                            reason = "RSS_LIMIT"
                            kill_group(p.pid)
                        if time.monotonic() - started > case.get("deadline", 4):
                            reason = "WALL_TIMEOUT"
                            kill_group(p.pid)
                        time.sleep(poll_s)
                    p.wait(timeout=2)
                finally:
                    kill_group(p.pid)
                    if p.poll() is None:
                        p.wait(timeout=2)
            if group:
                cg["sampled_memory_current_peak_mib"] = cg_peak / MIB
                cg["memory_peak_mib"] = (
                    int((group / "memory.peak").read_text()) / MIB if (group / "memory.peak").exists() else None
                )
                cg["events"] = {
                    k: int(v) for k, v in (line.split() for line in (group / "memory.events").read_text().splitlines())
                }
                if cg["events"].get("oom_kill", 0):
                    reason = "CGROUP_OOM"
            detail = (
                json.loads((work / "result.json").read_text())
                if (work / "result.json").exists() and (work / "result.json").stat().st_size
                else {}
            )
            record = dict(
                case=case["name"],
                policy=policy,
                wall_s=time.monotonic() - started,
                status=reason or detail.get("status", "PROCESS_EXIT"),
                returncode=p.returncode,
                sampled_tree_peak_mib=peak / MIB,
                max_observed_processes=processes,
                cgroup=cg,
                worker=detail,
                diagnostic=(work / "diagnostic.txt").read_text()[-1500:],
            )
            return record
        finally:
            if group:
                (group / "cgroup.kill").write_text("1")
                for _ in range(100):
                    try:
                        group.rmdir()
                        break
                    except OSError:
                        time.sleep(0.01)
                else:
                    raise RuntimeError(f"Owned cgroup cleanup failed: {group}")
            if record is not None:
                record["end_to_end_s"] = time.monotonic() - full_started


def group_cleanup_probe():
    """Controlled helper only: demonstrate direct-child vs process-group termination."""
    import psutil

    result = []
    for mode in ["parent_only", "process_group"]:
        with tempfile.TemporaryDirectory(prefix="lcb-tree-") as td:
            pidfile = Path(td) / "child.pid"
            helper = (
                'import subprocess,time,sys; p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(15)"]); '
                'open(sys.argv[1],"w").write(str(p.pid)); time.sleep(15)'
            )
            p = subprocess.Popen(
                [sys.executable, "-c", helper, str(pidfile)],
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                deadline = time.monotonic() + 3
                while not pidfile.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                child = int(pidfile.read_text())
                if mode == "parent_only":
                    p.kill()
                else:
                    kill_group(p.pid)
                p.wait(timeout=2)
                time.sleep(0.05)
                try:
                    alive = psutil.Process(child).status() != psutil.STATUS_ZOMBIE
                except psutil.NoSuchProcess:
                    alive = False
                result.append(dict(mode=mode, descendant_running_after_kill=alive))
            finally:
                kill_group(p.pid)
                if p.poll() is None:
                    p.wait(timeout=2)
    return result


def concurrency_probe(case, parent):
    import psutil

    report = []
    # Fixed 12 jobs per condition; concurrent workers are fresh processes.
    for width in [1, 2, 4]:
        for repeat in range(2):
            stop = threading.Event()
            peaks = []

            def monitor(stop=stop, peaks=peaks):
                while not stop.is_set():
                    children = psutil.Process().children(recursive=True)
                    total = 0
                    for child in children:
                        with contextlib.suppress(psutil.Error):
                            total += child.memory_info().rss
                    peaks.append(total / MIB)
                    stop.wait(0.01)

            thread = threading.Thread(target=monitor)
            thread.start()
            start = time.monotonic()
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=width) as pool:
                    jobs = list(pool.map(lambda _: run_one(case, "rss256_output", parent), range(12)))
            finally:
                stop.set()
                thread.join()
            report.append(
                dict(
                    width=width,
                    repeat=repeat,
                    jobs=12,
                    wall_s=time.monotonic() - start,
                    peak_sum_child_rss_mib=max(peaks, default=0),
                    statuses=[j["status"] for j in jobs],
                )
            )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--cgroup-parent", default="/sys/fs/cgroup")
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--policy", choices=POLICIES)
    parser.add_argument("--cgroup")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    if args.out is None:
        parser.error("--out is required")
    args.out.mkdir(parents=True, exist_ok=False)
    cases = fixtures()
    stats = None
    if args.dataset:
        extra, stats = real_cases(args.dataset)
        cases += extra
    report = dict(
        environment={
            "python": sys.version,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "emergency_as_mib": 1024,
            "policy_memory_mib": 256,
            "rss_poll_seconds": 0.005,
            "cgroup_parent": args.cgroup_parent,
            "fixture_scope": "synthetic and four hand-written reference programs; no model generations",
        },
        dataset=stats,
        runs=[],
        cleanup=[],
        concurrency=[],
    )
    import importlib.metadata

    report["environment"]["packages"] = {
        name: importlib.metadata.version(name) for name in ["numpy", "pandas", "psutil", "pyarrow"]
    }
    # Only an owned temporary child is written; never enable parent controllers.
    try:
        probe = make_cgroup(args.cgroup_parent)
        probe.rmdir()
        available = True
    except OSError as exc:
        available = False
        report["environment"]["cgroup_unavailable"] = str(exc)
    for policy in POLICIES:
        if policy.startswith("cgroup") and not available:
            continue
        for case in cases:
            if args.quick and case["name"] not in ["sum", "allocation_fast", "virtual_reservation"]:
                continue
            r = run_one(case, policy, args.cgroup_parent)
            report["runs"].append(r)
            print(policy, case["name"], r["status"], round(r["wall_s"], 3), flush=True)
            (args.out / "results.json").write_text(json.dumps(report, indent=2))
    report["cleanup"] = group_cleanup_probe()
    if not args.quick:
        report["concurrency"] = concurrency_probe(cases[0], args.cgroup_parent)
    (args.out / "results.json").write_text(json.dumps(report, indent=2))
    print("WROTE", args.out / "results.json", flush=True)


if __name__ == "__main__":
    main()
