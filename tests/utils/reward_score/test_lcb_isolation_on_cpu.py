# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Real bounded evaluator processes; no Ray cluster, models or GPU allocation."""

import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from verl.utils.reward_score.one_shot_opd.code_eval_v2 import _isolation as isolation
from verl.utils.reward_score.one_shot_opd.code_eval_v2 import compute_score, lcb_check_correctness


@pytest.fixture(autouse=True)
def bounded_policy(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("MIRROROPD_LCB_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("MIRROROPD_LCB_SLOT_DIR", str(tmp_path / "slots"))
    monkeypatch.setenv("MIRROROPD_LCB_MEMORY_MIB", "256")
    monkeypatch.setenv("MIRROROPD_LCB_ADDRESS_SPACE_MIB", "1024")
    monkeypatch.setenv("MIRROROPD_LCB_BACKEND", "rss")


def sample():
    return [{"input": "2 3", "output": "5", "testtype": "stdin", "metadata": {}}]


def run(code, timeout=1):
    return lcb_check_correctness(sample(), code, timeout)


@pytest.mark.parametrize(
    "code,status",
    [
        ("a,b=map(int,input().split());print(a+b)", "PASS"),
        ("print(0)", "WRONG_ANSWER"),
        ("print(1/0)", "RUNTIME_ERROR"),
        ("def broken(", "SYNTAX_ERROR"),
        ("while True: pass", "TIME_LIMIT"),
        ("a=bytearray(384*1024**2);print(5)", "MEMORY_LIMIT"),
    ],
)
def test_statuses(code, status):
    passed, detail = run(code)
    assert passed == (status == "PASS")
    assert detail["status"] == status
    assert detail["backend"] == "rss"
    assert detail["worker_pid"] != os.getpid()


def test_output_limit_and_original_binary_io_semantics(monkeypatch):
    monkeypatch.setenv("MIRROROPD_LCB_OUTPUT_MIB", "1")
    assert run('for _ in range(256): print("x"*65536)')[1]["status"] == "OUTPUT_LIMIT"
    assert run("import sys; print(sys.stdin.buffer.read())")[1]["status"] == "RUNTIME_ERROR"


def test_ignoring_alarm_is_stopped_by_supervisor():
    start = time.monotonic()
    _, detail = run("import signal\nsignal.signal(signal.SIGALRM,signal.SIG_IGN)\nwhile True: pass")
    assert detail["status"] == "TIME_LIMIT"
    assert time.monotonic() - start < 20


def test_failure_to_start_is_not_wrong_answer(monkeypatch):
    monkeypatch.setenv("MIRROROPD_LCB_STARTUP_TIMEOUT_S", "0.001")
    with pytest.raises(isolation.EvaluationInfrastructureError):
        compute_score("taco", "```python\nprint(5)\n```", sample())


@pytest.mark.parametrize(
    "key,value",
    [
        ("BACKEND", "invalid"),
        ("POLL_S", "nan"),
        ("MAX_CONCURRENT", "0"),
        ("MEMORY_MIB", "0"),
        ("ADDRESS_SPACE_MIB", "16"),
    ],
)
def test_bad_policy_fails_before_execution(monkeypatch, key, value):
    monkeypatch.setenv("MIRROROPD_LCB_" + key, value)
    with pytest.raises(isolation.EvaluationInfrastructureError):
        run("print(5)")


def test_missing_cgroup_controller_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setenv("MIRROROPD_LCB_BACKEND", "cgroup")
    monkeypatch.setenv("MIRROROPD_LCB_CGROUP_PARENT", str(tmp_path))
    with pytest.raises(isolation.EvaluationInfrastructureError):
        run("print(5)")
    assert not list(tmp_path.glob("mirroropd-lcb-*"))


def test_node_slots_are_shared_by_supervisor_processes(monkeypatch, tmp_path):
    monkeypatch.setenv("MIRROROPD_LCB_MAX_CONCURRENT", "1")
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda _: run("import time;time.sleep(0.15);print(5)"), range(3)))
    assert all(passed for passed, _ in results)
    events = [json.loads(line) for line in (tmp_path / "slots/events.jsonl").read_text().splitlines()]
    active = 0
    for event in events:
        active += 1 if event["event"] == "started" else -1
        assert 0 <= active <= 1
    assert active == 0
    assert max(detail["queue_s"] for _, detail in results) > 1
    assert len({detail["supervisor_pid"] for _, detail in results}) == 3


def test_active_capacity_cannot_change(tmp_path):
    a = isolation.Limits(slot_dir=str(tmp_path), max_concurrent=1)
    slot = isolation.try_slot(a)
    try:
        with pytest.raises(ValueError, match="active"):
            isolation.try_slot(isolation.Limits(slot_dir=str(tmp_path), max_concurrent=2))
    finally:
        slot.close()
    resized = isolation.try_slot(isolation.Limits(slot_dir=str(tmp_path), max_concurrent=2))
    resized.close()


def test_audit_context_is_bounded_and_contains_no_program(tmp_path):
    _, detail = lcb_check_correctness(
        sample(), "print(5)", 1, {"index": "question-1", "solution": "secret-code", "sample_id": "3"}
    )
    assert detail["context"] == {"index": "question-1", "sample_id": "3"}
    events = (tmp_path / "slots/events.jsonl").read_text()
    assert "secret-code" not in events and "print(5)" not in events
    assert "question-1" in events


def test_caller_death_reclaims_worker(tmp_path):
    helper = """import sys,json
sys.path.insert(0,sys.argv[1])
from _isolation import evaluate
evaluate({'input_output':json.dumps({'inputs':['0'],'outputs':['0']})},
         'import signal;signal.signal(signal.SIGALRM,signal.SIG_IGN)\\nwhile True: pass',1)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", helper, str(Path(isolation.__file__).parent)],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    audit = tmp_path / "slots/events.jsonl"
    try:
        deadline = time.monotonic() + 15
        started = None
        while time.monotonic() < deadline:
            if audit.exists():
                events = [json.loads(line) for line in audit.read_text().splitlines()]
                started = next((e for e in events if e["event"] == "started"), None)
                if started:
                    break
            time.sleep(0.05)
        assert started is not None
        process.kill()
        process.wait(timeout=3)
        deadline = time.monotonic() + 5
        finished = None
        while time.monotonic() < deadline:
            events = [json.loads(line) for line in audit.read_text().splitlines()]
            finished = next((e for e in events if e["event"] == "finished"), None)
            if finished:
                break
            time.sleep(0.05)
        assert finished is not None and finished["status"] == "INFRASTRUCTURE_ERROR"
        assert not psutil.pid_exists(started["worker_pid"])
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)


def test_cgroup_actual_memory_enforcement(monkeypatch):
    root = Path("/sys/fs/cgroup")
    probe = root / f"mirroropd-test-probe-{os.getpid()}"
    try:
        probe.mkdir()
    except OSError:
        pytest.skip("No cgroup delegation in this test environment")
    try:
        available = (probe / "memory.max").exists() and (probe / "cgroup.kill").exists()
    finally:
        probe.rmdir()
    if not available:
        pytest.skip("Required cgroup controllers unavailable")
    monkeypatch.setenv("MIRROROPD_LCB_BACKEND", "cgroup")
    monkeypatch.setenv("MIRROROPD_LCB_CGROUP_PARENT", str(root))
    _, detail = run("a=bytearray(384*1024**2);print(5)")
    assert detail["status"] == "MEMORY_LIMIT"
    assert detail["memory_events"]["oom_kill"] > 0
    assert not Path(detail["cgroup"]).exists()


def test_batch_preserves_order_and_does_not_swallow_infra(monkeypatch):
    from verl.utils.reward_score.one_shot_opd.code_eval_v2 import lcb_check_correctness_batch

    assert lcb_check_correctness_batch([sample(), sample()], ["print(5)", "print(0)"], num_workers=2) == [True, False]
    monkeypatch.setenv("MIRROROPD_LCB_BACKEND", "invalid")
    with pytest.raises(isolation.EvaluationInfrastructureError):
        lcb_check_correctness_batch([sample()], ["print(5)"])


def test_input_budget_and_queue_timeout(monkeypatch):
    monkeypatch.setenv("MIRROROPD_LCB_INPUT_MIB", "1")
    with pytest.raises(isolation.EvaluationInfrastructureError, match="budget"):
        isolation.evaluate({"input_output": "x" * (2 * 1024**2)}, "print(5)")
    monkeypatch.setenv("MIRROROPD_LCB_MAX_CONCURRENT", "1")
    monkeypatch.setenv("MIRROROPD_LCB_QUEUE_TIMEOUT_S", "0.1")
    slot = isolation.try_slot(isolation.Limits.from_env())
    try:
        with pytest.raises(isolation.EvaluationInfrastructureError, match="queue timeout"):
            run("print(5)")
    finally:
        slot.close()


def test_audit_directory_and_run_id(monkeypatch, tmp_path):
    monkeypatch.setenv("MIRROROPD_LCB_AUDIT_DIR", str(tmp_path / "audit"))
    monkeypatch.setenv("MIRROROPD_LCB_RUN_ID", "test-run")
    _, detail = run("print(5)")
    assert detail["context"]["run_id"] == "test-run"
    assert (tmp_path / "audit/events.jsonl").exists()
    assert not (tmp_path / "slots/events.jsonl").exists()


def test_functional_output_is_bounded(monkeypatch):
    monkeypatch.setenv("MIRROROPD_LCB_DIAGNOSTIC_MIB", "1")
    tests = [{"input": "0", "output": "1", "testtype": "functional", "metadata": {"func_name": "solve"}}]
    _, detail = lcb_check_correctness(tests, 'def solve(x):\n for _ in range(256): print("x"*65536)\n return 1', 1)
    assert detail["status"] == "OUTPUT_LIMIT"
