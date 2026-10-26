# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Validate LR overrides and optimizer snapshots used by the solver audit."""

import importlib
from pathlib import Path

import pytest
import torch


@pytest.fixture
def modules(monkeypatch):
    scripts = Path(__file__).resolve().parents[3] / "examples/mirroropd/docs/mirror-descent/scripts"
    monkeypatch.syspath_prepend(str(scripts))
    return importlib.import_module("solver_optimizers"), importlib.import_module("audit_mirror_solvers")


def test_constant_schedule_keeps_restored_group_lr_override(modules):
    optimizers, _ = modules
    p = torch.nn.Parameter(torch.ones(2))
    opt = torch.optim.AdamW([p], lr=1e-6)
    torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
    assert opt.param_groups[0]["initial_lr"] == 1e-6
    opt.param_groups[0]["lr"] = 1e-7
    scheduler = optimizers.constant_scheduler(opt)
    assert opt.param_groups[0]["lr"] == 1e-7
    p.grad = torch.ones_like(p)
    opt.step()
    scheduler.step()
    assert opt.param_groups[0]["lr"] == 1e-7


def test_snapshot_is_independent_and_restores_actual_optimizer_state(modules):
    _, audit = modules
    p = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    opt = torch.optim.AdamW([p], lr=0.01)
    p.grad = torch.tensor([0.2, -0.3])
    opt.step()
    saved = audit.cpu_tree(opt.state_dict())
    p.grad = torch.tensor([0.4, 0.1])
    opt.step()
    assert not audit.equal_tree(saved, opt.state_dict())
    opt.load_state_dict(audit.cpu_tree(saved))
    assert audit.equal_tree(saved, opt.state_dict())
    opt.state[p]["exp_avg"].add_(1)
    assert not audit.equal_tree(saved, opt.state_dict())


@pytest.mark.skipif(not hasattr(torch.optim, "Muon"), reason="Native Muon unavailable")
def test_muon_partition_excludes_embeddings_heads_and_norms(modules):
    optimizers, _ = modules
    named = [
        ("model.layers.0.proj.weight", torch.nn.Parameter(torch.zeros(4, 3))),
        ("model.embed_tokens.weight", torch.nn.Parameter(torch.zeros(8, 3))),
        ("lm_head.weight", torch.nn.Parameter(torch.zeros(8, 3))),
        ("model.layers.0.norm.weight", torch.nn.Parameter(torch.zeros(3))),
    ]
    opt = optimizers.FullMatrixMuonAdamW(named, lr=0.01, aux_lr=0.001)
    assert opt.partition == {"muon_tensors": 1, "aux_tensors": 3, "muon_parameters": 12, "aux_parameters": 51}
    optimizers.constant_scheduler(opt)
    assert [g["lr"] for g in opt.param_groups] == [0.01, 0.001]
    assert opt.auxiliary.state is opt.state


def test_forward_error_decomposition_is_zero_for_unchanged_policy(monkeypatch):
    scripts = Path(__file__).resolve().parents[3] / "examples/mirroropd/docs/mirror-descent/scripts"
    monkeypatch.syspath_prepend(str(scripts))
    probe = importlib.import_module("probe_mirror_forward")
    z = torch.tensor([[2.0, 0.0, -1.0], [0.0, 1.0, 2.0]], dtype=torch.float64)
    ref = z.log_softmax(-1)
    observed = (z + torch.tensor([0.01, -0.02, 0.01])).log_softmax(-1)
    r = probe.forward_error_decomposition(observed, observed, ref, ref)
    assert r["quadratic_motion"] == 0
    assert r["quadratic_error_change"] == 0
    assert r["actual_policy_kl"] == 0
    assert r["error_correlation"] == pytest.approx(1.0)


def test_fixed_logit_error_does_not_invent_error_motion(monkeypatch):
    scripts = Path(__file__).resolve().parents[3] / "examples/mirroropd/docs/mirror-descent/scripts"
    monkeypatch.syspath_prepend(str(scripts))
    probe = importlib.import_module("probe_mirror_forward")
    z = torch.tensor([[2.0, 0.0, -1.0]], dtype=torch.float64)
    dz = torch.tensor([[0.001, -0.002, 0.003]], dtype=torch.float64)
    error = torch.tensor([[0.2, -0.1, 0.1]], dtype=torch.float64)
    r = probe.forward_error_decomposition(
        (z + error).log_softmax(-1), (z + dz + error).log_softmax(-1), z.log_softmax(-1), (z + dz).log_softmax(-1)
    )
    assert r["quadratic_error_change"] < 1e-25
    assert abs(r["decomposition_residual"]) < 1e-20


def test_changing_arithmetic_error_is_separated_from_reference_motion(monkeypatch):
    scripts = Path(__file__).resolve().parents[3] / "examples/mirroropd/docs/mirror-descent/scripts"
    monkeypatch.syspath_prepend(str(scripts))
    probe = importlib.import_module("probe_mirror_forward")
    z = torch.tensor([[2.0, 0.0, -1.0]], dtype=torch.float64)
    e = torch.tensor([[0.01, -0.01, 0.0]], dtype=torch.float64)
    r = probe.forward_error_decomposition(
        (z + e).log_softmax(-1), (z - e).log_softmax(-1), z.log_softmax(-1), z.log_softmax(-1)
    )
    assert r["quadratic_reference_motion"] == 0
    assert r["quadratic_error_change"] == pytest.approx(r["quadratic_motion"])
    assert abs(r["decomposition_residual"]) < 1e-20
    assert r["error_correlation"] == pytest.approx(-1.0)


@pytest.mark.parametrize(
    "arm",
    [
        {"solver": "sgd", "lr": 0.001, "eta": 0.5},
        {"solver": "sgd", "lr": 0.001, "momentum": 0.9},
        {"solver": "adam_saved", "lr": 1e-6, "precision": "float16"},
    ],
)
def test_solver_audit_rejects_silently_ignored_settings(modules, arm):
    _, audit = modules
    with pytest.raises(ValueError):
        audit.validate_arm(arm)
