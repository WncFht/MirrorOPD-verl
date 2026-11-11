# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Fault injection for interrupted queues, corrupted groups and broker rollback."""

import os
import subprocess
from unittest.mock import Mock

import pytest

from examples.mirroropd.research.core.io import dump, read
from examples.mirroropd.research.runtime.processes import identity
from examples.mirroropd.research.studies.spectral.absorption.queue import reconcile_running


def test_recovery_refuses_live_worker_or_child_and_pid_reuse(tmp_path):
    current = identity(os.getpid())
    dump(
        tmp_path / "queue.json",
        dict(
            jobs=[
                dict(task_id=0, status="running", worker_identity=current),
                dict(task_id=1, status="running", worker_identity=[current[0], "old-start"], child_identity=current),
                dict(task_id=2, status="running", worker_identity=[current[0], "old-start"]),
                dict(task_id=3, status="running"),
            ]
        ),
    )
    result = reconcile_running(tmp_path)
    assert result == dict(live=[0, 1], interrupted=[2], unknown=[3])
    assert read(tmp_path / "queue.json")["jobs"][2]["status"] == "interrupted"
    assert read(tmp_path / "queue.json")["jobs"][3]["status"] == "running"


def test_same_size_group_corruption_is_rejected(tmp_path):
    from examples.mirroropd.research.studies.spectral.absorption.journal import GroupJournal

    tensor = tmp_path / "domain-gradient-sums/g.pt"
    tensor.parent.mkdir()
    tensor.write_bytes(b"original-bytes")
    journal = GroupJournal(tmp_path, {"source": "fixed"})
    journal.save("g", [], [])
    tensor.write_bytes(b"modified-bytes")
    with pytest.raises(ValueError, match="checksum mismatch"):
        GroupJournal(tmp_path, {"source": "fixed"})


def test_performance_exclusive_phases_and_wall_gaps(tmp_path):
    from examples.mirroropd.research.studies.spectral.absorption.performance import PerformanceLedger

    now = [0.0]
    ledger = PerformanceLedger(tmp_path, clock=lambda: now[0])
    with ledger.phase("backward"):
        now[0] = 2
        with pytest.raises(RuntimeError, match="overlap"):
            with ledger.phase("invalid"):
                pass
    now[0] = 3
    with ledger.phase("write"):
        now[0] = 5
    report = ledger.snapshot(sites=2, complete=True)
    assert report["session_wall_seconds"] == 5
    assert report["phases_seconds"] == {"backward": 2, "write": 2}
    assert report["unattributed_seconds"] == 1
    assert report["sites_per_minute"] == 24


@pytest.mark.parametrize("failure", ["launch", "attach", "reserve"])
def test_broker_setup_failure_rolls_back_only_own_reservation(tmp_path, monkeypatch, failure):
    from examples.mirroropd.research.runtime import broker

    commands = []
    child = Mock(pid=123)
    child.poll.return_value = None

    def wait(timeout):
        child.poll.return_value = -15
        return -15

    child.wait.side_effect = wait

    def run(command, **kwargs):
        action = command[2]
        commands.append(command)
        if action == failure:
            raise subprocess.CalledProcessError(1, command)
        return Mock(returncode=0)

    monkeypatch.setattr(broker, "occupied", lambda *args: True)
    monkeypatch.setattr(broker.subprocess, "run", run)

    def popen(*args, **kwargs):
        if failure == "launch":
            raise OSError("injected launch failure")
        return child

    monkeypatch.setattr(broker.subprocess, "Popen", popen)
    with pytest.raises((OSError, subprocess.CalledProcessError)):
        broker.run_leased(
            resource=tmp_path,
            repo=tmp_path,
            gpu=2,
            owner="only-this-owner",
            output=tmp_path / "attempt",
            command=["unused"],
        )
    actions = [cmd[2] for cmd in commands]
    assert (
        actions
        == {"reserve": ["reserve"], "launch": ["reserve", "release"], "attach": ["reserve", "attach", "release"]}[
            failure
        ]
    )
    assert all(cmd[cmd.index("--owner") + 1] == "only-this-owner" for cmd in commands)
    assert not (tmp_path / "attempt/ATTACHED").exists()
    if failure != "reserve":
        assert read(tmp_path / "attempt/occupancy.json")["verified"]


def test_external_stop_prevents_claim(tmp_path):
    from examples.mirroropd.research.studies.spectral.absorption.queue import claim

    external = tmp_path / "predecessor-STOP"
    dump(tmp_path / "settings.json", dict(stop_paths=[str(external)]))
    dump(tmp_path / "queue.json", dict(jobs=[dict(task_id=0, status="pending")]))
    external.touch()
    assert claim(tmp_path, 0) is None
    assert read(tmp_path / "queue.json")["jobs"][0]["status"] == "pending"


def test_stop_during_attach_never_opens_gate(tmp_path, monkeypatch):
    from examples.mirroropd.research.runtime import broker

    stop = tmp_path / "STOP"
    child = Mock(pid=123)
    child.poll.return_value = None

    def wait(timeout):
        child.poll.return_value = -15
        return -15

    child.wait.side_effect = wait
    commands = []

    def run(command, **kwargs):
        commands.append(command[2])
        if command[2] == "attach":
            stop.touch()
        return Mock(returncode=0)

    monkeypatch.setattr(broker, "occupied", lambda *args: True)
    monkeypatch.setattr(broker.subprocess, "run", run)
    monkeypatch.setattr(broker.subprocess, "Popen", lambda *args, **kwargs: child)
    with pytest.raises(InterruptedError, match="during broker attach"):
        broker.run_leased(
            resource=tmp_path,
            repo=tmp_path,
            gpu=1,
            owner="this-owner",
            output=tmp_path / "attempt",
            command=["unused"],
            stop_paths=[stop],
        )
    assert commands == ["reserve", "attach"]
    child.terminate.assert_called_once()
    assert not (tmp_path / "attempt/ATTACHED").exists()
    assert read(tmp_path / "attempt/occupancy.json")["verified"]


def test_gradient_pipeline_keeps_results_and_resumes_committed_group(tmp_path, monkeypatch):
    import torch

    from examples.mirroropd.research.studies.spectral.absorption import gradients as module
    from examples.mirroropd.research.studies.spectral.absorption.protocol import RunContext

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.tensor([[0.2, -0.1], [0.3, 0.4]]))

    calls = []

    def logits(model, row, position):
        calls.append(position)
        return model.w.sum(0) * (position + 1)

    monkeypatch.setattr(module, "load_model", lambda *args, **kwargs: TinyModel())
    monkeypatch.setattr(module, "checkpoint_identity", lambda path: {"path": path})
    monkeypatch.setattr(module, "prefix_logits", logits)
    monkeypatch.setattr(module, "collect_memory", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    (tmp_path / "directions").mkdir()
    torch.save(
        dict(learned=torch.ones(2, 2), off=torch.eye(2), null=torch.zeros(2, 2)), tmp_path / "directions/0000.pt"
    )
    row = dict(row_id="q1", domain="math", split="heldout", response_ids=[1, 2])
    banks = {"initial-student": [row]}
    reference = {("initial-student", "q1"): torch.tensor([[0.4], [0.8]])}
    head = torch.tensor([[1.0], [-1.0]])
    records = [dict(key="w", index=0)]
    job = {"gradient_stages": {"endpoint": "checkpoint-a"}}
    context = RunContext(tmp_path)
    module.gradients(context, job, tmp_path, banks, reference, head, records)
    folder = tmp_path / "gradients/endpoint"
    original = read(folder / "per-site-projections.json")
    assert len(original) == 2
    assert original[0]["directions"]["null"]["unit_derivative"] is None
    assert all(row["gradient_norm"] > 0 for row in original)
    report = read(next(folder.glob("performance-*.json")))
    assert report["complete"] and report["session_sites"] == 2
    for phase in [
        "forward_backward",
        "direction_read_transfer_projection",
        "gradient_d2h",
        "cpu_accumulate",
        "group_norms",
        "group_tensor_write",
        "group_commit",
    ]:
        assert report["phase_calls"][phase] >= 1
    (folder / "complete.json").unlink()
    calls.clear()
    module.gradients(context, job, tmp_path, banks, reference, head, records)
    assert not calls
    assert read(folder / "per-site-projections.json") == original
    assert read(folder / "complete.json")["sites"] == 2


def test_worker_timeout_reaps_child_and_preserves_failed_attempt(tmp_path, monkeypatch):
    from examples.mirroropd.research.studies.spectral.absorption.queue import worker

    out = tmp_path / "attempt-001"
    dump(tmp_path / "source-hashes.json", {})
    dump(tmp_path / "settings.json", {})
    dump(
        tmp_path / "queue.json",
        dict(jobs=[dict(task_id=0, job_id=0, status="pending", phase="gradients", output_attempt=str(out))]),
    )
    child = Mock(pid=os.getpid())
    child.poll.return_value = None
    waits = []

    def wait(timeout):
        waits.append(timeout)
        if len(waits) == 1:
            raise subprocess.TimeoutExpired("child", timeout)
        child.poll.return_value = -15
        return -15

    child.wait.side_effect = wait

    def popen(*args, **kwargs):
        (tmp_path / "STOP").touch()
        return child

    monkeypatch.setattr(subprocess, "Popen", popen)
    worker(tmp_path, 0)
    assert waits == [90, 90]
    assert child.poll() == -15
    assert read(tmp_path / "queue.json")["jobs"][0]["status"] == "failed"
    assert (out / "command.json").exists()
    assert (tmp_path / "errors/task-0.json").exists()


def test_unrestored_gpu_cannot_report_success(tmp_path, monkeypatch):
    from examples.mirroropd.research.runtime import broker

    checks = iter([True, False])
    monkeypatch.setattr(broker, "occupied", lambda *args: next(checks))
    child = Mock(pid=123)
    child.poll.return_value = 0
    child.wait.return_value = 0
    monkeypatch.setattr(broker.subprocess, "Popen", lambda *args, **kwargs: child)
    monkeypatch.setattr(broker.subprocess, "run", lambda *args, **kwargs: Mock(returncode=0))
    ticks = iter([0, 241])
    monkeypatch.setattr(broker.time, "monotonic", lambda: next(ticks))
    with pytest.raises(RuntimeError, match="occupancy not restored"):
        broker.run_leased(
            resource=tmp_path, repo=tmp_path, gpu=1, owner="this-owner", output=tmp_path / "attempt", command=["unused"]
        )
    assert not read(tmp_path / "attempt/occupancy.json")["verified"]


def test_stage_scope_releases_parameters_before_next_model(tmp_path, monkeypatch):
    import weakref

    import torch

    from examples.mirroropd.research.studies.spectral.absorption import gradients as module
    from examples.mirroropd.research.studies.spectral.absorption.protocol import RunContext

    previous_parameters = []

    def load(*args, **kwargs):
        assert all(ref() is None for ref in previous_parameters), "Previous stage retained model parameters"
        model = torch.nn.Linear(2, 2, bias=False)
        previous_parameters.extend(weakref.ref(p) for p in model.parameters())
        return model

    monkeypatch.setattr(module, "load_model", load)
    monkeypatch.setattr(module, "checkpoint_identity", lambda path: {"path": path})
    monkeypatch.setattr(module, "prefix_logits", lambda model, row, pos: model.weight.sum(0))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    (tmp_path / "directions").mkdir()
    torch.save(
        dict(learned=torch.ones(2, 2), off=torch.eye(2), null=torch.zeros(2, 2)), tmp_path / "directions/0000.pt"
    )
    row = dict(row_id="q", domain="math", split="heldout", response_ids=[1])
    module.gradients(
        RunContext(tmp_path),
        {"gradient_stages": {"initial": "a", "endpoint": "b"}},
        tmp_path,
        {"initial-student": [row]},
        {("initial-student", "q"): torch.ones(1, 1)},
        torch.tensor([[1.0], [-1.0]]),
        [dict(key="weight", index=0)],
    )
    assert all(ref() is None for ref in previous_parameters)
