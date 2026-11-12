# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Do not publish a partly written log line while following a live writer."""

import json
import sys
from types import SimpleNamespace

from examples.mirroropd.research.studies.absorption_dynamics.readouts import collect, records


def test_live_rows_wait_for_newline_and_preserve_same_step_records(tmp_path):
    path = tmp_path / "absorption-front4/training/E94/metrics.jsonl"
    path.parent.mkdir(parents=True)
    first = {"step": 16, "data": {"optim/actor/loss": 0.1}}
    second = {"step": 16, "data": {"eval/rkl/public_initial/all/question_mean": 0.2}}
    path.write_text(json.dumps(first) + "\n" + json.dumps(second))
    assert records(path) == [first]
    with path.open("a") as stream:
        stream.write("\n")
    assert collect(tmp_path)["E94"] == (path, [first, second])


def test_mirror_commits_last_record_without_waiting_for_next_outer(tmp_path, monkeypatch):
    from examples.mirroropd.research.studies.absorption_dynamics import readouts

    lane = tmp_path / "absorption-front4"
    path = lane / "training/E94/metrics.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"step": 40, "data": {"step/outer": 40}}) + "\n")
    (lane / "status.json").write_text(json.dumps({"state": "complete"}))
    committed = []

    class Run:
        summary = {}
        url = "https://wandb.ai/test/test/runs/test"
        finished = False

        def define_metric(self, *args, **kwargs):
            pass

        def log(self, data, step, commit=False):
            if commit:
                committed.append((step, data))

        def finish(self):
            self.finished = True

    run = Run()
    monkeypatch.setitem(
        sys.modules, "wandb", SimpleNamespace(init=lambda **kwargs: run, Settings=lambda **kwargs: None)
    )
    monkeypatch.setattr(
        readouts.netrc, "netrc", lambda path: SimpleNamespace(authenticators=lambda host: ("test", None, "test"))
    )
    monkeypatch.setenv("WANDB_API_KEY", "test")
    monkeypatch.setenv("WANDB_MODE", "offline")
    readouts.mirror(tmp_path, tmp_path / "netrc")
    assert len(committed) == 1
    assert committed[0][1]["mirror/source_lines"] == 1
    assert committed[0][1]["step/outer"] == 40
    assert run.finished


def test_completed_pair_mirror_finishes_before_rest_of_lane(tmp_path, monkeypatch):
    import pytest

    from examples.mirroropd.research.studies.absorption_dynamics import readouts

    lane = tmp_path / "absorption-front4"
    path = lane / "training/E94/metrics.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"step": 40, "data": {"step/outer": 40}}) + "\n")
    (lane / "status.json").write_text(
        json.dumps({"state": "running", "stages": [{"label": "E94-train", "state": "complete"}]})
    )
    finished = []
    run = SimpleNamespace(
        summary={},
        url="https://wandb.ai/test/test/runs/test",
        define_metric=lambda *args, **kwargs: None,
        log=lambda *args, **kwargs: None,
        finish=lambda: finished.append(True),
    )
    monkeypatch.setitem(
        sys.modules, "wandb", SimpleNamespace(init=lambda **kwargs: run, Settings=lambda **kwargs: None)
    )
    monkeypatch.setattr(
        readouts.netrc, "netrc", lambda path: SimpleNamespace(authenticators=lambda host: ("test", None, "test"))
    )
    monkeypatch.setenv("WANDB_API_KEY", "test")
    monkeypatch.setenv("WANDB_MODE", "offline")

    def stop_after_one_cycle(seconds):
        assert finished == [True]
        raise RuntimeError("end test")

    monkeypatch.setattr(readouts.time, "sleep", stop_after_one_cycle)
    with pytest.raises(RuntimeError, match="end test"):
        readouts.mirror(tmp_path, tmp_path / "netrc")
    assert finished == [True]


def test_watch_handles_empty_metrics_during_pair_handoff(tmp_path, monkeypatch):
    import pytest

    from examples.mirroropd.research.studies.absorption_dynamics import readouts

    lane = tmp_path / "absorption-front4"
    path = lane / "training/E100/metrics.jsonl"
    path.parent.mkdir(parents=True)
    path.touch()
    (lane / "status.json").write_text(json.dumps({"state": "running", "current": "E100-train"}))
    monkeypatch.setattr(readouts, "render", lambda *args: None)
    monkeypatch.setattr(readouts.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=""))

    def stop_after_one_cycle(seconds):
        report = json.loads((tmp_path / "readout-watch/latest.json").read_text())
        assert report["performance"]["E100"]["latest_optimizer_step"] == 0
        assert report["performance"]["E100"]["completed_outer"] == 0
        raise RuntimeError("end test")

    monkeypatch.setattr(readouts.time, "sleep", stop_after_one_cycle)
    with pytest.raises(RuntimeError, match="end test"):
        readouts.watch(tmp_path, tmp_path / "figures")
