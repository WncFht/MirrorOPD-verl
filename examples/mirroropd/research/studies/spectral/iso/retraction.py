# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Checked polar retraction near the Stiefel manifold, with exact SVD fallback."""

import torch


def retract(matrix, *, method="newton", tolerance=1e-10, max_iterations=6, return_info=False):
    """Return the thin polar factor; preserve SVD semantics for wide matrices.

    Coupled Newton-Schulz computes the inverse square root of the small Gram
    matrix in FP64. A Frobenius bound <= .5 certifies its convergence basin.
    The full Gram residual of the resulting frame is checked before rounding
    back to the input dtype. Outside the basin, or after failed convergence,
    use the original exact SVD. No QR replacement or singular-value clipping.
    """
    if method not in {"newton", "svd"}:
        raise ValueError(f"Unknown polar method: {method}")
    if matrix.ndim != 2 or not matrix.is_floating_point() or min(matrix.shape) == 0:
        raise ValueError("Polar retraction requires a nonempty real floating matrix")
    if tolerance <= 0 or max_iterations < 1:
        raise ValueError("Positive tolerance and iteration budget required")
    transposed = matrix.shape[0] < matrix.shape[1]
    x = (matrix.T if transposed else matrix).double()
    info = dict(requested=method, used="svd", iterations=0, fallback=None)
    result = None
    if method == "newton":
        identity = torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
        y = x.T @ x
        deviation = float((y - identity).norm())
        info["initial_gram_frobenius"] = deviation
        if deviation <= 0.5:
            z = identity.clone()
            for iteration in range(max_iterations):
                correction = (3 * identity - z @ y) * 0.5
                y, z = y @ correction, correction @ z
                info["iterations"] = iteration + 1
                # Check the inverse square root identity before the tall multiply.
                if float((z @ y - identity).abs().max()) <= tolerance * 0.1:
                    candidate = x @ z
                    residual = float((candidate.T @ candidate - identity).abs().max())
                    info["orthogonality_max"] = residual
                    if residual <= tolerance:
                        result = candidate
                        info["used"] = "newton"
                    break
            if result is None:
                info["fallback"] = "convergence gate"
        else:
            info["fallback"] = "outside certified near-orthogonal basin"
    if result is None:
        u, _, vh = torch.linalg.svd(x, full_matrices=False, driver="gesvd" if x.is_cuda else None)
        result = u @ vh
    if transposed:
        result = result.T
    result = result.to(matrix.dtype)
    return (result, info) if return_info else result


def spectrum_gradient(gradient, u, v):
    """Compute diag(U.T @ gradient @ V) without forming a square product.

    Choose the contraction with the shorter outer dimension. This is exactly
    the chain rule for W = U diag(s) V.T; accumulation keeps input precision.
    """
    if gradient.shape[0] <= gradient.shape[1]:
        return ((gradient @ v) * u).sum(0)
    return ((gradient.T @ u) * v).sum(0)
