# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Actual BF16 CUDA kernels, including multi-part vocabulary reductions."""

import pytest
import torch

from verl.trainer.distillation.full_vocab import full_vocab_objective_result
from verl.trainer.distillation.full_vocab_triton import fused_bpo_rows
from verl.workers.config.distillation_spec import resolve_distillation_spec

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("objective", ["mdpo", "bpo"])
@pytest.mark.parametrize("vocab", [31, 8193, 151936])
def test_real_bf16_fused_value_and_head_hidden_vjp(objective, vocab):
    torch.manual_seed(17)
    hs, ws = [
        (torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * 0.2).requires_grad_()
        for shape in [(5, 16), (vocab, 16)]
    ]
    ht, wt, hq, wq = [
        torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * 0.2
        for shape in [(5, 12), (vocab + 3, 12), (5, 16), (vocab, 16)]
    ]
    spec = resolve_distillation_spec({"loss_mode": "mirror", "mirror": {"objective": objective, "eta": 0.3}})
    reference = full_vocab_objective_result(hs, ws, ht, wt, spec=spec, hq=hq, wq=wq, backend="dense")
    gradients = torch.autograd.grad(reference.loss_sum / spec.eta, (hs, ws))
    for backend in ("precompute", "precompute_fused"):
        actual = full_vocab_objective_result(hs, ws, ht, wt, spec=spec, hq=hq, wq=wq, backend=backend, token_chunk=2)
        torch.testing.assert_close(actual.loss_sum, reference.loss_sum, atol=2e-5, rtol=2e-4)
        actual_grad = torch.autograd.grad(actual.loss_sum / spec.eta, (hs, ws))
        for got, wanted in zip(actual_grad, gradients, strict=True):
            relative = (got.float() - wanted.float()).norm() / wanted.float().norm().clamp_min(1e-12)
            assert relative < 0.04, (objective, vocab, backend, relative.item())


def test_fused_bpo_small_variance_does_not_subtract_second_moments():
    torch.manual_seed(25)
    z = torch.randn(3, 8193, device="cuda")
    b = z + 1e-4 * torch.randn_like(z)
    lq = torch.randn_like(z).log_softmax(-1)
    value, dz = fused_bpo_rows(z, b, lq, torch.float32)
    q, d = lq.exp(), z - b
    residual = d - (q * d).sum(-1, keepdim=True)
    expected = (q * residual.square()).sum(-1) / 2
    assert (value >= 0).all()
    torch.testing.assert_close(value, expected, atol=1e-12, rtol=1e-4)
    torch.testing.assert_close(dz, q * residual, atol=1e-10, rtol=1e-4)
