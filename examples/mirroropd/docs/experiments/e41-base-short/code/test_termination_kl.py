# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
import os
import sys

import pytest
import torch

if os.environ.get("E41_PATCHED_SOURCE_ROOT"):
    sys.path.insert(0, os.environ["E41_PATCHED_SOURCE_ROOT"])
import verl.trainer.distillation.full_vocab as f

pytestmark = pytest.mark.skipif(
    not os.environ.get("E41_PATCHED_SOURCE_ROOT"),
    reason="Archived pilot tests require the original patched source snapshot",
)


def dense_coarse(hs, ws, ht, wt, ids):
    p = (hs @ ws.T).softmax(-1)
    q = (ht.detach() @ wt.detach().T).softmax(-1)
    rest = [i for i in range(p.shape[1]) if i not in ids]
    pc = torch.cat((p[:, rest], p[:, ids].sum(-1, keepdim=True)), dim=-1)
    qc = torch.cat((q[:, rest], q[:, ids].sum(-1, keepdim=True)), dim=-1)
    return (pc * (pc.log() - qc.log())).sum()


@pytest.mark.parametrize("chunk", [1, 3, 10])
def test_objective_gradient_and_raw_diagnostics_match_dense_reference(monkeypatch, chunk):
    monkeypatch.setattr(f, "termination_equivalence_ids", lambda: (1, 4))
    torch.manual_seed(42)
    hs = torch.randn(7, 5, dtype=torch.float64, requires_grad=True)
    ws = torch.randn(9, 5, dtype=torch.float64, requires_grad=True)
    ht = torch.randn(7, 6, dtype=torch.float64, requires_grad=True)
    wt = torch.randn(9, 6, dtype=torch.float64, requires_grad=True)
    expected = dense_coarse(hs, ws, ht, wt, [1, 4])
    actual = f.full_reverse_kl_result(hs, ws, ht, wt, backend="precompute", token_chunk=chunk)
    torch.testing.assert_close(actual.loss_sum, expected, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(actual.token_kl, f.dense_rows(hs, ws, ht, wt), atol=1e-10, rtol=1e-10)
    assert not actual.token_kl.requires_grad
    eg = torch.autograd.grad(expected, (hs, ws))
    ag = torch.autograd.grad(actual.loss_sum, (hs, ws, ht, wt), allow_unused=True)
    for x, y in zip(ag[:2], eg, strict=True):
        torch.testing.assert_close(x, y, atol=1e-10, rtol=1e-10)
    assert ag[2:] == (None, None)


def test_equivalent_stop_actions_do_not_incentivize_continuation(monkeypatch):
    monkeypatch.setattr(f, "termination_equivalence_ids", lambda: (0, 1))
    p = torch.tensor([[0.8, 1e-6, 0.1, 0.099999]], dtype=torch.float64)
    q = p[:, [1, 0, 2, 3]]
    eye = torch.eye(4, dtype=torch.float64)
    with torch.no_grad():
        coarse, gradient, raw = f.termination_equivalent_kl_rows(p.log(), eye, q.log(), eye)
    assert raw.item() > 10 and abs(coarse.item()) < 1e-12
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-12, rtol=0)
    with torch.no_grad():
        _, ordinary = f.precompute_kl_rows(p.log(), eye, q.log(), eye)
    assert ordinary[:, [0, 1]].sum() > 0


def test_disabled_flag_preserves_original_kl(monkeypatch):
    monkeypatch.delenv("VERL_E41_TERMINATION_EQUIV", raising=False)
    torch.manual_seed(1)
    hs = torch.randn(5, 3, dtype=torch.float64)
    ws = torch.randn(8, 3, dtype=torch.float64)
    ht = torch.randn(5, 4, dtype=torch.float64)
    wt = torch.randn(8, 4, dtype=torch.float64)
    result = f.full_reverse_kl_result(hs, ws, ht, wt, backend="precompute", token_chunk=2)
    torch.testing.assert_close(result.loss_sum, f.dense_rows(hs, ws, ht, wt).sum())


def test_unimplemented_backend_fails_instead_of_silently_using_wrong_objective(monkeypatch):
    monkeypatch.setattr(f, "termination_equivalence_ids", lambda: (1, 4))
    with pytest.raises(ValueError, match="precompute"):
        f.full_reverse_kl_result(
            torch.zeros(2, 3), torch.zeros(8, 3), torch.zeros(2, 3), torch.zeros(8, 3), backend="dense"
        )
