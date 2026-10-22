# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def monitor(monkeypatch, root, pending=False):
    path = Path(__file__).resolve().parents[3] / "examples/mirroropd/docs/infra/profile/scripts/hourly_monitor.py"
    spec = importlib.util.spec_from_file_location("hourly_monitor_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr("sys.argv", [str(path), "--root", str(root), "--thread", "test-thread", "--codex", "fake"])
    (root / "hourly-monitor.json").write_text(
        json.dumps({"tick": 1 if pending else 0, "pending": pending, "next_at_unix": 0})
    )
    return mod


def test_queue_and_explicit_stop(monkeypatch, tmp_path):
    mod = monitor(monkeypatch, tmp_path)
    calls = []

    def queue(command, **kwargs):
        calls.append(command)
        (tmp_path / "MONITOR_STOP").touch()
        return SimpleNamespace(returncode=0, stdout="Queued message test-id for thread test-thread.", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", queue)
    mod.main()
    state = json.loads((tmp_path / "hourly-monitor.json").read_text())
    assert state["state"] == "stopped" and state["tick"] == 1 and state["pending"]
    assert calls[0][:5] == ["fake", "queue", "--thread", "test-thread", "--message"]


def test_unacknowledged_review_does_not_accumulate(monkeypatch, tmp_path):
    mod = monitor(monkeypatch, tmp_path, True)
    monkeypatch.setattr(mod.time, "sleep", lambda _: (tmp_path / "MONITOR_STOP").touch())

    def unexpected(*args, **kwargs):
        raise AssertionError("must not queue a second unacknowledged review")

    monkeypatch.setattr(mod.subprocess, "run", unexpected)
    mod.main()
    history = json.loads((tmp_path / "hourly-history.jsonl").read_text().splitlines()[0])
    assert history["state"] == "awaiting_previous_review" and history["tick"] == 1


def test_review_done_finishes_without_queuing(monkeypatch, tmp_path):
    mod = monitor(monkeypatch, tmp_path)
    (tmp_path / "REVIEW_DONE.json").write_text('{"reason":"reviewed"}')
    mod.main()
    assert json.loads((tmp_path / "hourly-monitor.json").read_text())["state"] == "finished"
