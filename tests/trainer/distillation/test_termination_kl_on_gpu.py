# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""CUDA BF16 checks for the deployed precompute termination-equivalence path."""

import pytest
import torch

from verl.trainer.distillation.full_vocab import full_reverse_kl_result, project


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("chunk", [1, 5])
def test_bfloat16_projection_and_vjp_match_autograd(chunk):
    torch.manual_seed(36)
    hs = (torch.randn(9, 32, device="cuda", dtype=torch.bfloat16) * 0.2).requires_grad_()
    ws = (torch.randn(128, 32, device="cuda", dtype=torch.bfloat16) * 0.2).requires_grad_()
    ht = torch.randn(9, 48, device="cuda", dtype=torch.bfloat16) * 0.2
    wt = torch.randn(128, 48, device="cuda", dtype=torch.bfloat16) * 0.2
    p = project(hs, ws).softmax(-1)
    q = project(ht, wt).softmax(-1)
    ids = [1, 4]
    rest = [i for i in range(128) if i not in ids]
    pc = torch.cat((p[:, rest], p[:, ids].sum(-1, keepdim=True)), dim=-1)
    qc = torch.cat((q[:, rest], q[:, ids].sum(-1, keepdim=True)), dim=-1)
    expected = (pc * (pc.log() - qc.log())).sum()
    grads = torch.autograd.grad(expected, (hs, ws))
    actual = full_reverse_kl_result(hs, ws, ht, wt, backend="precompute", token_chunk=chunk, termination_token_ids=ids)
    torch.testing.assert_close(actual.loss_sum, expected, atol=1e-5, rtol=1e-5)
    raw = (p.detach() * (p.detach().log() - q.log())).sum(-1)
    torch.testing.assert_close(actual.token_kl, raw, atol=1e-5, rtol=1e-5)
    for actual_grad, expected_grad in zip(torch.autograd.grad(actual.loss_sum, (hs, ws)), grads, strict=True):
        assert torch.isfinite(actual_grad).all()
        torch.testing.assert_close(actual_grad, expected_grad, atol=5e-4, rtol=2e-2)
