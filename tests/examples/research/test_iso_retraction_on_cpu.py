# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from examples.mirroropd.research.studies.spectral.iso.retraction import retract


@pytest.mark.parametrize("shape", [(17, 7), (7, 17), (11, 11)])
@pytest.mark.parametrize("perturbation", [0.0, 0.001, 0.02])
def test_newton_matches_the_polar_factor_not_just_an_orthogonal_frame(shape, perturbation):
    torch.manual_seed(24)
    base = retract(torch.randn(*shape, dtype=torch.float64), method="svd")
    value = base + perturbation * torch.randn_like(base)
    actual, info = retract(value, return_info=True)
    expected = retract(value, method="svd")
    assert info["used"] == "newton"
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    assert torch.linalg.norm(actual - value) <= torch.linalg.norm(base - value) + 1e-10


@pytest.mark.parametrize("shape", [(17, 7), (7, 17), (9, 9)])
def test_far_or_rank_deficient_inputs_fall_back_without_changing_svd_semantics(shape):
    torch.manual_seed(8)
    value = torch.randn(*shape, dtype=torch.float64)
    value[:, 0] = 0
    actual, info = retract(value, return_info=True)
    assert info["used"] == "svd" and info["fallback"]
    torch.testing.assert_close(actual, retract(value, method="svd"))


def test_low_iteration_budget_falls_back_and_float_dtype_is_preserved():
    torch.manual_seed(90)
    q = torch.linalg.qr(torch.randn(17, 7, dtype=torch.float64))[0]
    value = q + 0.03 * torch.randn_like(q)
    actual, info = retract(value, max_iterations=1, return_info=True)
    assert info["used"] == "svd"
    torch.testing.assert_close(actual, retract(value, method="svd"))
    assert retract(q.float()).dtype == torch.float32


@pytest.mark.parametrize("shape", [(17, 9), (9, 17), (11, 11)])
def test_spectrum_gradient_matches_chain_rule(shape):
    from examples.mirroropd.research.studies.spectral.iso.retraction import spectrum_gradient

    torch.manual_seed(19)
    m, n = shape
    u, _, vh = torch.linalg.svd(torch.randn(m, n, dtype=torch.float64), full_matrices=False)
    s = torch.randn(min(shape), dtype=torch.float64, requires_grad=True)
    gradient = torch.randn(m, n, dtype=torch.float64)
    (((u * s) @ vh) * gradient).sum().backward()
    torch.testing.assert_close(spectrum_gradient(gradient, u, vh.T), s.grad)


def test_factor_optimizer_records_newton_calls_and_fallback():
    from examples.mirroropd.research.studies.spectral.iso.training import FactorAdam

    torch.manual_seed(31)
    weight = torch.nn.Parameter(torch.randn(9, 7))
    optimizer = FactorAdam([("matrix", weight)], 1e-5, "frames", retraction="newton")
    weight.grad = torch.randn_like(weight)
    optimizer.step()
    assert optimizer.retraction_stats == dict(calls=2, newton=2, fallback=0)
    actual = optimizer._retract(torch.randn(17, 7, dtype=torch.float64))
    assert optimizer.retraction_stats == dict(calls=3, newton=2, fallback=1)
    torch.testing.assert_close(actual.T @ actual, torch.eye(7, dtype=torch.float64))
