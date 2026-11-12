# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Exact matrix diagnostics; callers control device, precision and matrix scope."""

import math

import torch


def ratio(numerator, denominator):
    """Undefined normalized quantities are JSON null, never an invented zero."""
    return float(numerator / denominator) if denominator > 0 else None


def energy(x):
    return x.double().square().sum()


def frame_components(base, delta, *, frame=None):
    """Orthogonal diag/off/null in the base thin-SVD frame (not finite rotations)."""
    if base.ndim != 2 or base.shape != delta.shape:
        raise ValueError("base and delta must be matrices of identical shape")
    u, _, vh = torch.linalg.svd(base, full_matrices=False) if frame is None else frame
    coordinates = u.T @ delta @ vh.T
    diagonal = (u * coordinates.diagonal()) @ vh
    in_frame = u @ coordinates @ vh
    return {"diag": diagonal, "off": in_frame - diagonal, "null": delta - in_frame}


def right_frame_components(delta, basis):
    """Split a window update by an orthonormal input-space basis (n by k).

    basis comes from an earlier cumulative update, not the full weight matrix
    or a teacher residual. Products avoid materializing an n by n projector.
    No norm matching is applied: the two outputs sum to the actual update.
    """
    if delta.ndim != 2 or basis.ndim != 2 or delta.shape[1] != basis.shape[0]:
        raise ValueError("delta must be m by n and basis n by k")
    if basis.shape[1] == 0 or basis.shape[1] > basis.shape[0]:
        raise ValueError("basis must have between 1 and n columns")
    if delta.dtype != basis.dtype or delta.device != basis.device:
        raise ValueError("delta and basis must share dtype and device")
    if delta.dtype not in (torch.float32, torch.float64):
        raise ValueError("frame decomposition requires FP32 or FP64")
    gram = basis.T @ basis
    tolerance = 1e-5 if basis.dtype == torch.float32 else 1e-10
    if not torch.allclose(gram, torch.eye(len(gram), dtype=gram.dtype, device=gram.device), atol=tolerance, rtol=0):
        raise ValueError("right basis must be orthonormal")
    inside = (delta @ basis) @ basis.T
    return {"in": inside, "out": delta - inside}


def spectrum_summary(matrix, *, threshold=1e-5, singular_values=None, driver=None):
    """Full-spectrum rank diagnostics; threshold is relative to the largest value."""
    if not 0 < threshold < 1:
        raise ValueError("threshold must lie in (0, 1)")
    s = torch.linalg.svdvals(matrix, driver=driver) if singular_values is None else singular_values
    squared = s.double().square()
    total = squared.sum()
    return dict(
        shape=list(matrix.shape),
        frobenius=float(total.sqrt()),
        stable_rank=ratio(total, squared[0]),
        numerical_rank=int((s > threshold * s[0]).sum()),
        relative_threshold=threshold,
        top_energy={str(k): ratio(squared[:k].sum(), total) for k in (1, 4, 16, 64)},
    )


def weight_summary(base, target, *, frame=None, singular_values=None, driver=None):
    """ISO scalars plus update rank; zero denominators remain undefined."""
    delta = target - base
    u, s0, vh = torch.linalg.svd(base, full_matrices=False) if frame is None else frame
    spectral_distance = (torch.linalg.svdvals(target, driver=driver) - s0).double().norm()
    coordinates = u.T @ delta @ vh.T
    diagonal_energy = energy(coordinates.diagonal())
    delta_energy = energy(delta)
    return dict(
        **spectrum_summary(delta, singular_values=singular_values, driver=driver),
        delta_sigma=ratio(spectral_distance, energy(base).sqrt()),
        rho_sigma=ratio(spectral_distance, delta_energy.sqrt()),
        kappa_spec=ratio(base.numel() / min(base.shape) * diagonal_energy, delta_energy),
        frame_energy=dict(
            diag=float(diagonal_energy),
            off=float(energy(coordinates) - diagonal_energy),
            null=float((delta - u @ coordinates @ vh).double().square().sum()),
        ),
    )


def top_fraction(delta, fraction, *, restore_norm=False):
    """Keep ceil(fraction * min(shape)) components; fraction is rank, not energy."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must lie in (0, 1]")
    u, s, vh = torch.linalg.svd(delta, full_matrices=False)
    k = math.ceil(fraction * len(s))
    result = (u[:, :k] * s[:k]) @ vh[:k]
    if restore_norm and energy(result) > 0:
        result = result * (energy(delta) / energy(result)).sqrt()
    return result


def random_controls(delta, *, seed=0, singular_values=None, driver=None):
    """Same-shape/norm Gaussian and real-spectrum/random-frame controls."""
    generator = torch.Generator(device=delta.device).manual_seed(seed)

    def normal(shape):
        return torch.randn(shape, dtype=delta.dtype, device=delta.device, generator=generator)

    gaussian = normal(delta.shape)
    gaussian *= (energy(delta) / energy(gaussian)).sqrt()
    q = min(delta.shape)
    u = torch.linalg.qr(normal((delta.shape[0], q)), mode="reduced").Q
    v = torch.linalg.qr(normal((delta.shape[1], q)), mode="reduced").Q
    singular = torch.linalg.svdvals(delta, driver=driver) if singular_values is None else singular_values
    return {"gaussian": gaussian, "random_frames": (u * singular) @ v.T}


def head_terms(h0, ht, w0, wt):
    """Exact bias-free head identity; states are post-final-norm, rows are positions."""
    return {"representation": (ht - h0) @ w0.T, "head": ht @ (wt - w0).T}
