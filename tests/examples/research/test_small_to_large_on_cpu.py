# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Conditional-support gradients must match autograd without trimming the model head."""

import torch

from examples.mirroropd.research.studies.boundary import small_to_large as study


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(11, 3)
        self.lm_head = torch.nn.Linear(3, 7, bias=False)


def test_conditional_gradient_matches_autograd_and_keeps_tail(monkeypatch):
    torch.manual_seed(42)
    model = Toy()
    monkeypatch.setattr(study, "SUPPORT", 4)
    monkeypatch.setattr(study.native, "hidden", lambda model, row: model.embed(torch.tensor(row["response_ids"])))
    rows = [dict(id=0, response_ids=[1, 2, 3, 4, 5]), dict(id=1, response_ids=[6, 7, 8])]
    states = {r["id"]: torch.randn(len(r["response_ids"]), 2) for r in rows}
    head = torch.randn(4, 2)
    ref = study.backward(model, head, states, rows, reference=True, chunk=2)
    gradients = {n: p.grad.clone() for n, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    candidate = study.backward(model, head, states, rows, chunk=3)
    assert abs(ref - candidate) < 1e-9
    for name, p in model.named_parameters():
        torch.testing.assert_close(p.grad, gradients[name], rtol=1e-5, atol=1e-9)
    assert model.lm_head.weight.shape == (7, 3)
    assert torch.count_nonzero(model.lm_head.weight.grad[4:]) == 0


def test_two_gpu_jobs_leave_one_gpu_and_advance_dependencies(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from examples.mirroropd.research.core.io import dump
    from examples.mirroropd.research.studies.fixed_transfer import runner

    jobs = [
        dict(
            name=name,
            args=["step", name],
            marker=str(tmp_path / f"{name}.json"),
            width=width,
            dependencies=deps,
            retries=0,
        )
        for name, width, deps in [("first", 2, []), ("side", 1, []), ("next", 2, ["first", "side"])]
    ]
    devices = []

    def launch(command, **kw):
        name = command[4]
        devices.append((name, kw["env"]["CUDA_VISIBLE_DEVICES"]))
        dump(tmp_path / f"{name}.json", {})
        return SimpleNamespace(pid=999999999, poll=lambda: 0)

    monkeypatch.setattr(runner.subprocess, "Popen", launch)
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    runner.queue(SimpleNamespace(gpus=(4, 5, 6)), tmp_path / "plan.json", "custom", tmp_path, "test", custom_jobs=jobs)
    assert devices == [("first", "4,5"), ("side", "6"), ("next", "4,5")]


def test_long_replay_preserves_fixed_pool_and_protects_old_outputs(tmp_path, monkeypatch):
    import json

    import pytest

    for key in ["OUT", "CACHE", "UPDATES", "TRAIN_GROUPS", "SCORE_GROUPS", "SAVE_RESUME", "RUN_AUXILIARY"]:
        monkeypatch.setattr(study, key, getattr(study, key))
    path = study.REPO / "examples/mirroropd/research/studies/boundary/plans/small-to-large-stage320.json"
    plan = json.loads(path.read_text())
    candidate = tmp_path / "plan.json"
    plan["output"] = str(study.BASE_OUT)
    candidate.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="new output"):
        study.configure(candidate)
    plan["output"] = str(tmp_path / "science")
    candidate.write_text(json.dumps(plan))
    study.configure(candidate)
    assert [study.training_group(s) for s in [1, 64, 65, 320, 321, 640]] == ["A", "D", "E", "T", "A", "T"]
    assert study.SAVE_RESUME and not study.RUN_AUXILIARY
    assert set("ABCDEFGHIJKLMNOPQRST") <= set(study.SCORE_GROUPS)
    plan["training"]["optimizer"]["lr"] = 1e-5
    candidate.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="Optimizer"):
        study.configure(candidate)


def test_diagnostic_replay_requires_measured_deterministic_gradients(tmp_path, monkeypatch):
    import json

    import pytest

    for key in [
        "OUT",
        "CACHE",
        "UPDATES",
        "TRAIN_GROUPS",
        "SCORE_GROUPS",
        "SAVE_RESUME",
        "RUN_AUXILIARY",
        "FA_DETERMINISTIC",
        "REPLAY_REFERENCE_POLICY",
        "REPEATABILITY_EVIDENCE",
    ]:
        monkeypatch.setattr(study, key, getattr(study, key))
    monkeypatch.setenv("FLASH_ATTENTION_DETERMINISTIC", "0")
    plan = json.loads(
        (study.REPO / "examples/mirroropd/research/studies/boundary/plans/small-to-large-stage320.json").read_text()
    )
    plan.update(output=str(tmp_path / "science"), cache=str(tmp_path / "cache"), replay_reference_policy="diagnostic")
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="deterministic new replica"):
        study.configure(path)
    proof = dict(
        student=list(study.MODELS["r7"]),
        teacher=list(study.MODELS["r1"]),
        scientific_updates=0,
        support=study.SUPPORT,
        results=[
            dict(deterministic=False, unequal_gradient_tensors=1),
            dict(deterministic=True, unequal_gradient_tensors=0, relative_l2=0, first_loss=1, second_loss=1),
        ],
    )
    evidence = tmp_path / "probe.json"
    evidence.write_text(json.dumps(proof))
    plan["training"]["flash_attention_deterministic"] = True
    plan["repeatability_probe"] = str(evidence)
    path.write_text(json.dumps(plan))
    study.configure(path)
    assert study.FA_DETERMINISTIC and study.REPEATABILITY_EVIDENCE
    proof["results"][1]["unequal_gradient_tensors"] = 1
    evidence.write_text(json.dumps(proof))
    with pytest.raises(AssertionError):
        study.configure(path)
