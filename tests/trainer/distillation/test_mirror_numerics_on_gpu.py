# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Actual CUDA regression: saturated distributions and logit-gauge shifts."""

import pytest
import torch

from verl.trainer.distillation.full_vocab_triton import fused_mirror_rows

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("vocab", [31, 8193, 151936])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_near_target_fused_gradient_matches_fp64(vocab, dtype):
    base = torch.full((2, vocab), -24.0, device="cuda")
    base[:, 0] = 0
    base[:, 1] = -10
    base[:, 2] = -12
    teacher = base.clone()
    teacher[:, 1] += 0.03
    teacher[:, 2] -= 0.02
    lq = base.log_softmax(-1)
    # Preserve a non-contiguous teacher view and its full-support normalizer.
    lt = torch.cat([teacher, teacher.new_full((2, 3), -40)], dim=-1).log_softmax(-1)[:, :vocab]
    b = 0.7 * lq + 0.3 * lt
    initial_gradient = None
    for offset in [0.0, 32.0, 128.0]:
        z = base + offset
        oracle_z = z.double().requires_grad_()
        lp = oracle_z.log_softmax(-1)
        oracle = torch.autograd.grad((lp.exp() * (lp - b.double())).sum(), oracle_z)[0]
        for diagnostics in [False, True]:
            _, actual, _ = fused_mirror_rows(z, b, lq, lt, dtype, objective="mdpo", diagnostics=diagnostics)
            relative = (actual.double() - oracle).norm() / oracle.norm()
            assert relative < (0.006 if dtype == torch.bfloat16 else 0.001), relative.item()
            if initial_gradient is None:
                initial_gradient = actual.clone()
            torch.testing.assert_close(actual, initial_gradient, atol=0, rtol=0)
