# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Actual CUDA checks of the fused kernels, not the CPU fallback."""

from types import SimpleNamespace

import pytest
import torch
from test_stable_rows_on_cpu import oracle

from verl.trainer.distillation.full_vocab_triton import fused_mirror_logits, fused_rows

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_endpoint_preserves_dominant_gradient_and_small_positive_value(dtype):
    z = torch.tensor([[0.0, -24.0, -26.0]], device="cuda")
    t = torch.tensor([[0.0, -23.97, -26.02]], device="cuda")
    expected, gradient, _, _ = oracle(z[0].cpu(), z[0].cpu(), t[0].cpu(), 0, 1, "mdpo")
    first = None
    for offset in [0.0, 32.0, 128.0]:
        value, g = fused_rows(z + offset, t, dtype)
        assert value.item() == pytest.approx(expected, rel=2e-5)
        assert g[0, 0] != 0
        torch.testing.assert_close(
            g[0].double().cpu(), gradient, atol=1e-18, rtol=0.005 if dtype == torch.bfloat16 else 2e-5
        )
        if first is not None:
            torch.testing.assert_close(g, first, atol=0, rtol=0)
        first = g


@pytest.mark.parametrize("objective", ["mdpo", "bpo"])
@pytest.mark.parametrize("kappa", [0.0, 1e-7, 0.5, 1.0])
@pytest.mark.parametrize("eta", [1e-6, 0.3, 1.0])
def test_fused_split_increment_and_self_target(objective, kappa, eta):
    z = torch.tensor([[32.0, 22.0, 20.0]], device="cuda")
    t = z + z.new_tensor([[0.0, 0.00003, -0.00002]])
    spec = SimpleNamespace(objective=objective, kappa=kappa, eta=eta, diagnostics=True)
    value, g, diag = fused_mirror_logits(z, z, t, torch.float32, spec=spec)
    rv, rg, rk, rz = oracle(z[0].cpu(), z[0].cpu(), t[0].cpu(), kappa, eta, objective)
    torch.testing.assert_close(g[0].double().cpu(), rg, atol=1e-24, rtol=4e-5)
    assert value.item() == pytest.approx(rv, rel=2e-4, abs=1e-27)
    assert diag[0, 0].item() == pytest.approx(rk, rel=2e-4, abs=1e-27)
    assert diag[0, 1].item() == pytest.approx(rz, rel=2e-4, abs=1e-27)
    self_value, self_gradient, self_diag = fused_mirror_logits(z, z, z, torch.bfloat16, spec=spec)
    assert torch.count_nonzero(self_gradient) == torch.count_nonzero(self_value) == torch.count_nonzero(self_diag) == 0


@pytest.mark.parametrize(
    "kappa,eta", [(0.0, 0.999999), (0.5, 0.999999), (0.99999999, 0.3), (0.99999999, 1e-6), (0.99999999, 0.999999)]
)
def test_fused_normalizer_and_teacher_endpoint_boundary(kappa, eta):
    anchor = torch.tensor([[32.0, 22.0, 20.0]], device="cuda")
    teacher = anchor + anchor.new_tensor([[0.0, 0.00003, -0.00002]])
    spec = SimpleNamespace(objective="mdpo", kappa=kappa, eta=eta, diagnostics=True)
    value, g, diag = fused_mirror_logits(teacher, anchor, teacher, torch.float32, spec=spec)
    rv, rg, rk, rz = oracle(teacher[0].cpu(), anchor[0].cpu(), teacher[0].cpu(), kappa, eta, "mdpo")
    assert diag[0, 1].item() < 0
    torch.testing.assert_close(g[0].double().cpu(), rg, atol=1e-25, rtol=3e-4)
    assert value.item() == pytest.approx(rv, rel=3e-4, abs=1e-29)
    assert diag[0, 0].item() == pytest.approx(rk, rel=3e-4, abs=1e-29)
    assert diag[0, 1].item() == pytest.approx(rz, rel=3e-4, abs=1e-29)


def test_fused_near_linear_power_mean_with_nonlocal_ratios():
    z = torch.tensor([[0.0, -2.0, -4.0]], device="cuda")
    t = z + z.new_tensor([[0.0, 1.0, -0.75]])
    spec = SimpleNamespace(objective="mdpo", kappa=0.999999, eta=1e-6, diagnostics=True)
    value, g, diag = fused_mirror_logits(z, z, t, torch.float32, spec=spec)
    rv, rg, rk, rz = oracle(z[0].cpu(), z[0].cpu(), t[0].cpu(), spec.kappa, spec.eta, "mdpo")
    torch.testing.assert_close(g[0].double().cpu(), rg, atol=1e-15, rtol=3e-4)
    assert value.item() == pytest.approx(rv, rel=3e-4)
    assert diag[0, 0].item() == pytest.approx(rk, rel=3e-4)
    assert diag[0, 1].item() == pytest.approx(rz, rel=3e-4)
