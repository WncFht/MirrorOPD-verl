# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Near-target MDPO VJPs must preserve logit-gauge invariance in FP32."""

import pytest
import torch

from verl.trainer.distillation.objectives import full_vocab_rows


@pytest.mark.parametrize("offset", [0.0, 16.0, 32.0, 128.0])
def test_mdpo_gradient_ignores_large_row_constant(offset):
    base = torch.tensor([[0.0, -10.0, -12.0, -20.0]])
    teacher = base + torch.tensor([[0.0, 0.03, -0.02, 0.01]])
    b = 0.7 * base.log_softmax(-1) + 0.3 * teacher.log_softmax(-1)
    z = base + offset
    _, actual, _ = full_vocab_rows(z, b, objective="mdpo")
    reference_z = z.double().requires_grad_()
    lp = reference_z.log_softmax(-1)
    reference = torch.autograd.grad((lp.exp() * (lp - b.double())).sum(), reference_z)[0]
    relative = (actual.double() - reference).norm() / reference.norm()
    assert relative < 1e-3, relative.item()
    _, unshifted, _ = full_vocab_rows(base, b, objective="mdpo")
    torch.testing.assert_close(actual, unshifted, atol=0, rtol=0)


def test_self_target_has_no_analytic_logit_gradient():
    torch.manual_seed(12)
    z = torch.randn(4, 8193) * 4 + 32
    target = z.log_softmax(-1).detach()
    rows, dz, _ = full_vocab_rows(z, target, objective="mdpo")
    torch.testing.assert_close(rows, torch.zeros_like(rows), atol=0, rtol=0)
    torch.testing.assert_close(dz, torch.zeros_like(dz), atol=0, rtol=0)
