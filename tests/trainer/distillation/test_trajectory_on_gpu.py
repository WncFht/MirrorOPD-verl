# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Deferred CUDA acceptance for native/fused trajectory score and weighted VJPs."""

import pytest
import torch
from test_trajectory_vjp_on_cpu import independent_reference, spec

from verl.trainer.distillation.full_vocab import full_vocab_objective_result
from verl.trainer.distillation.reduction import PackedReduction
from verl.workers.config.distillation_spec import FutureCreditConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("backend", ["precompute", "precompute_fused"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("mode", ["none", "sum", "mean", "discounted_sum", "discounted_mean", "discounted_sequence"])
@pytest.mark.parametrize(
    "aggregate",
    ["token-mean", "token-sum", "seq-mean-token-mean", "seq-mean-token-sum", "seq-mean-token-sum-norm"],
)
@pytest.mark.parametrize("eta,stop_ids", [(1.0, []), (0.2, []), (0.2, [1, 4])])
def test_trajectory_cuda_value_and_gradients(backend, dtype, mode, aggregate, eta, stop_ids):
    torch.manual_seed(13)
    inputs = [
        (torch.randn(*s, device="cuda", dtype=dtype) * 0.2).requires_grad_()
        for s in [(7, 8), (257, 8), (7, 6), (257, 6), (7, 8), (257, 8)]
    ]
    hs, ws, ht, wt, hq, wq = inputs
    layout = PackedReduction(
        (4, 0, 1, 2),
        torch.tensor([0, 1, 3, 6, 0, 2, 5], device="cuda"),
        torch.tensor([0, 2, 1, 4, 3, 1, 4], device="cuda"),
        aggregate,
        FutureCreditConfig(mode=mode, gamma=0.87 if mode.startswith("discounted") else 1.0),
    )
    settings = spec(eta, 0.5)
    expected, _ = independent_reference(*(x.double() for x in inputs), settings, layout, stop_ids)
    actual = full_vocab_objective_result(
        hs,
        ws,
        ht,
        wt,
        hq=hq,
        wq=wq,
        spec=settings,
        reduction=layout,
        backend=backend,
        token_chunk=2,
        termination_token_ids=stop_ids,
    )
    torch.testing.assert_close(actual.loss_sum.double(), expected, atol=5e-5, rtol=2e-4)
    wanted = torch.autograd.grad(expected / eta, (hs, ws))
    got = torch.autograd.grad(actual.loss_sum / eta, inputs, allow_unused=True)
    assert got[2:] == (None,) * 4
    for a, b in zip(got[:2], wanted, strict=True):
        relative_error = (a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)
        assert relative_error < (1e-4 if dtype == torch.float32 else 0.03), relative_error.item()
