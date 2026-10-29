# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent teacher-only scoring for the precision-transfer experiments."""

import importlib
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def modules(monkeypatch):
    return importlib.import_module("examples.mirroropd.research.backends.fsdp"), importlib.import_module(
        "examples.mirroropd.research.studies.precision.audit_mirror_solvers"
    )


def test_eta_one_reference_uses_complete_teacher_normalizer_without_anchor(modules):
    replay, _ = modules
    hs = torch.tensor([[0.2, -0.3], [0.1, 0.5]], dtype=torch.float64)
    ws = torch.tensor([[1.0, 0.0], [-1.0, 0.5], [0.2, 0.3]], dtype=torch.float64)
    ht = hs + 0.1
    wt = torch.cat([ws, torch.ones(1, 2, dtype=torch.float64)], 0)
    lp, lt = (hs @ ws.T).log_softmax(-1), (ht @ wt.T).log_softmax(-1)
    expected = (lp.exp() * (lp - lt[:, :3])).sum(-1)
    result = replay.reference(hs, ws, ht, wt, spec=SimpleNamespace(teacher_only_kl=True))
    torch.testing.assert_close(result.token_objective, expected, rtol=0, atol=0)
    assert result.token_diagnostics == {}


def test_opd_score_does_not_invent_an_anchor_measurement(modules):
    _, audit = modules
    score = audit.canonical_score({"distill/full_kl": 0.012}, SimpleNamespace(teacher_only_kl=True))
    assert score["distill/teacher_kl"] == score["distill/target_kl"] == score["distill/objective"] == 0.012
    assert score["distill/target_log_z"] == 0
    assert "distill/anchor_kl" not in score and "distill/mdpo/objective" not in score


def test_checkpoint_option_rejects_incompatible_optimizer(modules):
    _, audit = modules
    with pytest.raises(ValueError, match="native saved Adam"):
        audit.validate_arm({"solver": "muon", "lr": 1e-6, "save_checkpoint": True})
