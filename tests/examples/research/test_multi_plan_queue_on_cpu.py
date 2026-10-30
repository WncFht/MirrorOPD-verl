# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Shared queues retain distinct frozen plans, sources and dependency namespaces."""

import pytest

from examples.mirroropd.research.core.artifacts import file_hash
from examples.mirroropd.research.core.io import dump, read
from examples.mirroropd.research.studies.fixed_transfer.batch import jobs


def test_batch_keeps_each_source_and_plan_and_rejects_changed_source(tmp_path):
    old = tmp_path / "old-source"
    old.mkdir()
    (old / "worker.py").write_text("original code")
    task = dict(name="train", args=[], dependencies=[], marker="done.json", retries=0)
    spec = dict(
        entries=[
            dict(
                name="old",
                plan={"version": 1},
                source=str(old),
                source_files=[dict(path="worker.py", sha256=file_hash(old / "worker.py"))],
                jobs=[dict(task, preferred_gpus=[1])],
            ),
            dict(name="new", plan={"version": 2}, jobs=[task]),
        ]
    )
    plan = tmp_path / "batch.json"
    dump(plan, spec)
    result = jobs(plan)
    assert [r["name"] for r in result] == ["old:train", "new:train"]
    assert result[0]["source"] == str(old) and result[0]["preferred_gpus"] == [1]
    assert "source" not in result[1]
    assert [read(r["plan"])["version"] for r in result] == [1, 2]
    (old / "worker.py").write_text("changed code")
    with pytest.raises(ValueError, match="source changed"):
        jobs(plan)


def test_queue_honors_worker_gpu_source_and_plan(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from examples.mirroropd.research.studies.fixed_transfer import runner

    captured = []
    tasks = [
        dict(
            name=name,
            args=["work", name],
            marker=str(tmp_path / (name + ".json")),
            dependencies=[],
            retries=0,
            plan=str(tmp_path / (name + "-plan.json")),
            **({"source": str(tmp_path / "old-source"), "preferred_gpus": [1]} if name == "eval" else {}),
        )
        for name in ["eval", "train"]
    ]

    def launch(command, **kwargs):
        name = command[4]
        captured.append((name, kwargs["env"]["CUDA_VISIBLE_DEVICES"], kwargs["env"]["PYTHONPATH"], command[-1]))
        dump(tmp_path / (name + ".json"), {})
        return SimpleNamespace(pid=999999999, poll=lambda: 0)

    monkeypatch.setattr(runner.subprocess, "Popen", launch)
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    runner.queue(SimpleNamespace(gpus=(0, 1)), tmp_path / "batch.json", "batch", tmp_path, "test", custom_jobs=tasks)
    assert captured == [
        ("eval", "1", str(tmp_path / "old-source"), str(tmp_path / "eval-plan.json")),
        ("train", "0", str(tmp_path / "source"), str(tmp_path / "train-plan.json")),
    ]
