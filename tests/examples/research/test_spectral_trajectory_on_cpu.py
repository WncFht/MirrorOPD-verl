# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Check streamed dynamics against direct linear algebra and historical metrics."""

import gzip
import json

import pytest
import torch

from examples.mirroropd.research.studies.spectral.right_dynamics import measure as historical_measure
from examples.mirroropd.research.studies.spectral.trajectory import measure, run
from tests.examples.research.test_spectral_on_cpu import make_checkpoint


@pytest.mark.parametrize("shape", [(7, 5), (5, 9)])
def test_streamed_geometry_matches_direct_and_historical(tmp_path, shape):
    torch.manual_seed(82)
    key = "model.layers.0.self_attn.k_proj.weight"
    base = torch.randn(*shape, dtype=torch.float64)
    a, b = torch.randn_like(base) * 0.01, torch.randn_like(base) * 0.02
    paths = {
        name: str(make_checkpoint(tmp_path / name, {key: tensor}))
        for name, tensor in {"base": base, "teacher": base + b, "early": base + a, "final": base + a + b}.items()
    }
    task = dict(
        condition="synthetic",
        base=paths["base"],
        teacher_path=paths["teacher"],
        tensor=key,
        points=[[16, paths["early"]], [64, paths["final"]]],
        ranks=[1, 3, 16],
        threads=1,
    )
    actual = measure(task)
    old = historical_measure(task)
    assert actual["dtype"] == "float64"
    assert actual["sha256"] == {**old["sha256"], "teacher": actual["sha256"]["teacher"]}
    assert actual["increments"][0]["optimizer_interval"] == 48
    for i, d in enumerate((a, a + b)):
        s = torch.linalg.svdvals(d)
        torch.testing.assert_close(torch.tensor(actual["stages"][i]["singular_values"]), s.float())
        c = actual["stages"][i]["teacher_comparison"]
        assert c["residual_energy_ratio"] == pytest.approx(1 + c["norm_ratio"] ** 2 - 2 * c["norm_ratio"] * c["cosine"])
    for row in actual["pairs"]:
        reference = next(
            x
            for x in old["pairs"]
            if (x["step_a"], x["step_b"], x["rank"]) == (row["step_a"], row["step_b"], row["rank"])
        )
        assert row["overlap"] == pytest.approx(reference["overlap"], abs=1e-12)
        assert row["a_captures_b"] == pytest.approx(reference["a_captures_b"], abs=1e-12)
    for row in actual["increments"][0]["captures"]:
        assert row["capture"] <= row["optimal_capture"] + 1e-12


def test_invalid_plan_and_failed_task_are_not_success(tmp_path):
    with pytest.raises(ValueError, match="unique devices"):
        run({"tasks": [{}]}, tmp_path / "duplicate", devices=["cuda:0", "cuda:0"])
    with pytest.raises(ValueError, match="positive, unique"):
        measure(dict(ranks=[0], points=[[1, "missing"]]))
    with pytest.raises(RuntimeError, match="matrix tasks failed"):
        run(
            {"tasks": [dict(ranks=[1], points=[[1, "missing"]], base="missing", tensor="missing")]}, tmp_path / "failed"
        )
    assert (tmp_path / "failed/error-00000.json").exists()


def test_resume_preserves_records_and_rejects_changed_protocol(tmp_path):
    from examples.mirroropd.research.studies.spectral.trajectory import resume_records

    task = dict(condition="test", points=[[16, "checkpoint"]])
    plan = dict(tasks=[task])
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    (source / "protocol.json").write_text(json.dumps(plan))
    raw = gzip.compress(
        json.dumps(dict(schema="absorption-geometry-v1", dtype="float64", task=task, stages=[dict(step=16)])).encode()
    )
    (source / "matrix-00000.json.gz").write_bytes(raw)
    assert resume_records(plan, output, [source]) == {0}
    assert (output / "matrix-00000.json.gz").read_bytes() == raw
    with pytest.raises(ValueError, match="protocol differs"):
        resume_records(dict(tasks=[dict(task, points=[[32, "other"]])]), output, [source])


def test_stop_does_not_dispatch_queued_tasks(tmp_path, monkeypatch):
    from examples.mirroropd.research.studies.spectral import trajectory

    original = trajectory.resume_records

    def stopped(plan, output, sources):
        reused = original(plan, output, sources)
        (output / "STOP").touch()
        return reused

    monkeypatch.setattr(trajectory, "resume_records", stopped)
    output = tmp_path / "stopped"
    with pytest.raises(InterruptedError, match="drained active matrices"):
        run(dict(tasks=[dict(ranks=[1], points=[[1, "missing"]])]), output)
    assert not list(output.glob("error-*.json"))
    receipt = json.loads((output / "exit.json").read_text())
    assert receipt["stopped"] and receipt["complete"] == 0 and receipt["exit_code"] == 2


def test_cross_gram_and_shared_projection_match_direct():
    from examples.mirroropd.research.studies.spectral.right_dynamics import angle_metrics

    torch.manual_seed(123)
    a = torch.linalg.qr(torch.randn(19, 7, dtype=torch.float64))[0]
    b = torch.linalg.qr(torch.randn(19, 7, dtype=torch.float64))[0]
    delta = torch.randn(11, 19, dtype=torch.float64)
    cross = a.T @ b
    captured = (delta @ a).square().sum(0).cumsum(0)
    for k in (1, 3, 7):
        direct = torch.linalg.svdvals(a[:, :k].T @ b[:, :k]).square().mean()
        actual = angle_metrics(a[:, :k], b[:, :k], cross=cross[:k, :k])
        assert actual["overlap"] == pytest.approx(float(direct), abs=1e-12)
        assert float(captured[k - 1]) == pytest.approx(float((delta @ a[:, :k]).square().sum()), abs=1e-12)
