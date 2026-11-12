# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Actual window updates split by early right frames, without norm rescaling."""

import pytest
import torch

from examples.mirroropd.research.studies.spectral.decomposition import right_frame_components


@pytest.mark.parametrize("shape", [(5, 3), (3, 5), (4, 4)])
def test_window_split_is_orthogonal_and_invariant_to_frame_coordinates(shape):
    torch.manual_seed(42)
    delta = torch.randn(shape, dtype=torch.float64)
    basis = torch.linalg.qr(torch.randn(shape[1], 2, dtype=delta.dtype)).Q
    parts = right_frame_components(delta, basis)
    torch.testing.assert_close(parts["in"] + parts["out"], delta)
    torch.testing.assert_close((parts["in"] * parts["out"]).sum(), delta.new_zeros(()), atol=1e-12, rtol=0)
    torch.testing.assert_close(parts["out"] @ basis, delta.new_zeros((shape[0], 2)), atol=1e-12, rtol=0)
    torch.testing.assert_close(delta.square().sum(), parts["in"].square().sum() + parts["out"].square().sum())
    rotation = torch.linalg.qr(torch.randn(2, 2, dtype=delta.dtype)).Q
    changed = right_frame_components(delta, basis @ rotation)
    torch.testing.assert_close(parts["in"], changed["in"])


def test_small_outside_energy_can_have_large_function_effect():
    delta = torch.diag(torch.tensor([1.0, 0.01], dtype=torch.float64))
    basis = torch.tensor([[1.0], [0.0]], dtype=delta.dtype)
    parts = right_frame_components(delta, basis)
    x = torch.tensor([0.0, 100.0], dtype=delta.dtype)
    assert parts["out"].square().sum() / delta.square().sum() < 0.001
    torch.testing.assert_close(parts["out"] @ x, delta @ x)
    torch.testing.assert_close(parts["in"] @ x, torch.zeros_like(x))


def test_window_split_rejects_nonorthogonal_basis_and_low_precision():
    delta = torch.ones(3, 4)
    with pytest.raises(ValueError, match="orthonormal"):
        right_frame_components(delta, torch.ones(4, 2))
    with pytest.raises(ValueError, match="m by n"):
        right_frame_components(delta, torch.ones(3, 2))
    with pytest.raises(ValueError, match="FP32 or FP64"):
        right_frame_components(delta.half(), torch.eye(4).half())
