# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Geometric checks independent of the recorded experiment matrices."""

import numpy as np
import pytest

from examples.mirroropd.research.studies.spectral.analyze_gradient_spectrum import eigensystem, projection_curve


def test_projection_matches_explicit_parameter_space():
    rng = np.random.default_rng(42)
    gradients = rng.normal(size=(7, 19))
    held = rng.normal(size=19)
    curve, _, _ = projection_curve(gradients @ gradients.T, gradients @ held, held @ held)
    orthogonal, _ = np.linalg.qr(gradients.T)
    expected = np.linalg.norm(orthogonal.T @ held) ** 2 / (held @ held)
    np.testing.assert_allclose(curve[-1], expected, rtol=1e-12)
    assert np.all(np.diff(curve) >= -1e-12)


def test_rescaling_and_rotation_preserve_full_span():
    rng = np.random.default_rng(12)
    gradients = rng.normal(size=(5, 11))
    held = rng.normal(size=11)
    rotation, _ = np.linalg.qr(rng.normal(size=(11, 11)))
    scaled = (gradients * np.arange(1, 6)[:, None]) @ rotation
    rotated_held = held @ rotation
    a, _, _ = projection_curve(gradients @ gradients.T, gradients @ held, held @ held)
    b, _, _ = projection_curve(scaled @ scaled.T, scaled @ rotated_held, held @ held)
    np.testing.assert_allclose(a[-1], b[-1], rtol=1e-12)


def test_rank_deficiency_and_orthogonal_holdout():
    gradients = np.array([[1, 0, 0], [2, 0, 0]], dtype=float)
    held = np.array([0, 0, 1], dtype=float)
    curve, _, _ = projection_curve(gradients @ gradients.T, gradients @ held, held @ held)
    np.testing.assert_array_equal(curve, 0)


def test_indefinite_matrix_is_rejected():
    with pytest.raises(ValueError, match="negative eigenvalue"):
        eigensystem(np.array([[1, 2], [2, 1]], dtype=float))
