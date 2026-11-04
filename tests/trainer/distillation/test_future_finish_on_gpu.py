# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Fused future finish against independent native action-score composition."""

import pytest
import torch

from verl.trainer.distillation.full_vocab_triton import finish_future
from verl.trainer.distillation.termination import TerminationActionMap
from verl.trainer.distillation.trajectory import action_score

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("stop_ids", [(), (1, 4)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("vocab", [31, 8193])
def test_future_finish_matches_native_score_and_row_weights(stop_ids, dtype, vocab):
    torch.manual_seed(51)
    logits = torch.randn(3, vocab, device="cuda")
    sampled = torch.tensor([0, 1, 4], device="cuda")
    actions = TerminationActionMap.create(vocab, stop_ids, logits.device) if stop_ids else None
    local = (torch.randn_like(logits) * 0.01).to(dtype)
    future = logits.new_tensor([0.0, 0.8, -0.3])
    weights = logits.new_tensor([0.2, 0.5, 1.0])
    lp, score = action_score(logits, sampled, actions)
    expected = ((local + future[:, None] * score) * weights[:, None]).to(dtype)
    value, actual = finish_future(logits, local.clone(), sampled, future, weights, actions, None)
    torch.testing.assert_close(value, lp, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(actual, expected, atol=2e-7 if dtype == torch.float32 else 2e-5, rtol=0.01)


def test_future_finish_preserves_near_certain_action_pivot():
    logits = torch.tensor([[0.0, -30.0, -35.0]], device="cuda")
    sampled = torch.zeros(1, dtype=torch.long, device="cuda")
    lp, score = action_score(logits, sampled, None)
    value, actual = finish_future(logits, torch.zeros_like(logits), sampled, logits.new_ones(1), 1, None, None)
    torch.testing.assert_close(value, lp, atol=1e-19, rtol=1e-5)
    torch.testing.assert_close(actual, score, atol=1e-19, rtol=1e-5)
    assert actual[0, 0] > 0


def test_fused_future_casts_before_parameter_gradient_gemms(monkeypatch):
    from verl.trainer.distillation import vjp
    from verl.trainer.distillation.full_vocab import full_vocab_objective_result
    from verl.trainer.distillation.reduction import PackedReduction
    from verl.workers.config.distillation_spec import FutureCreditConfig, resolve_distillation_spec

    observed = []
    original = vjp.add_product

    def accumulate(out, left, right):
        observed.append((out.dtype, left.dtype, right.dtype))
        return original(out, left, right)

    monkeypatch.setattr(vjp, "add_product", accumulate)
    h = torch.randn(3, 8, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(31, 8, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    layout = PackedReduction(
        (2, 1),
        torch.tensor([0, 1, 0], device="cuda"),
        torch.zeros(3, device="cuda", dtype=torch.long),
        future=FutureCreditConfig(mode="mean"),
    )
    spec = resolve_distillation_spec({"loss_mode": "mirror", "mirror": {"eta": 1}})
    result = full_vocab_objective_result(
        h, w, h.detach() * 0.9, w.detach(), spec=spec, reduction=layout, backend="precompute_fused", token_chunk=2
    )
    result.loss_sum.backward()
    assert observed == [(torch.float32, torch.bfloat16, torch.bfloat16)] * 4
