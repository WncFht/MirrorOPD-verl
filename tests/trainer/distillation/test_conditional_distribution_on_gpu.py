# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Fused conditional head values/VJP across strided supports and AMP scaling."""

import pytest
import torch

from verl.trainer.distillation.full_vocab import full_vocab_objective_result
from verl.workers.config.distillation_spec import resolve_distillation_spec

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("mapped", [False, True])
@pytest.mark.parametrize("termination", [(), (0, 1)])
def test_fused_conditional_matches_reference_for_gapped_and_permuted_rows(dtype, mapped, termination):
    torch.manual_seed(75)
    hs = torch.randn(5, 32, device="cuda", dtype=dtype) * 0.1
    ws = torch.randn(517, 32, device="cuda", dtype=dtype) * 0.1
    ht = torch.randn(5, 24, device="cuda", dtype=dtype) * 0.1
    wt = torch.randn(389, 24, device="cuda", dtype=dtype) * 0.1
    support = dict(mode="common_conditioned", common_vocab_size=385)
    if mapped:
        support = dict(mode="common_conditioned", student_ids=list(range(384, -1, -1)), teacher_ids=list(range(385)))
    results = []
    for backend in ["precompute", "precompute_fused"]:
        h, w = hs.detach().clone().requires_grad_(), ws.detach().clone().requires_grad_()
        output = full_vocab_objective_result(
            h,
            w,
            ht,
            wt,
            spec=resolve_distillation_spec(
                {"loss_mode": "mirror", "mirror": {"objective": "mdpo", "estimator": "full_vocab", "eta": 1}}
            ),
            backend=backend,
            token_chunk=2,
            distribution=support,
            termination_token_ids=termination,
        )
        (output.loss_sum * 1024).backward()
        results.append((output, h.grad, w.grad))
    ref, actual = results
    torch.testing.assert_close(actual[0].loss_sum, ref[0].loss_sum, rtol=2e-4, atol=1e-7)
    torch.testing.assert_close(actual[0].objective_rows, ref[0].objective_rows, rtol=2e-4, atol=1e-7)
    for a, b in zip(actual[1:], ref[1:], strict=True):
        torch.testing.assert_close(a, b, rtol=0.01 if dtype == torch.bfloat16 else 0.002, atol=1e-5)
    assert torch.count_nonzero(actual[2][385:]) == 0
    for key in ("student_excluded_mass", "teacher_excluded_mass"):
        torch.testing.assert_close(
            actual[0].token_diagnostics[key], ref[0].token_diagnostics[key], rtol=1e-6, atol=1e-7
        )
