# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Finite gradient-span bounds, distinguished from neural tangent-kernel spectra."""

import numpy as np
from scipy.optimize import nnls


def span_diagnostics(gram, train, target, *, rtol=1e-10):
    """Bound alignment with a fixed target from any combination of train vectors.

    Target is a full-length vector of weights on the Gram's records/batches.
    The oracle uses target gradients and is an upper bound, not a training rule.
    This only characterizes the measured finite span at a frozen checkpoint.
    """
    gram = np.asarray(gram, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    train = np.asarray(train, dtype=int)
    kernel = gram[np.ix_(train, train)]
    cross = gram[train] @ target
    target_norm = float(target @ gram @ target)
    values, vectors = np.linalg.eigh(kernel)
    if values.min() < -max(values.max(), 1e-30) * 1e-7:
        raise ValueError("Gradient Gram is not positive semidefinite")
    keep = values > max(values.max(), 1e-30) * rtol
    values, vectors = values[keep], vectors[:, keep]
    coordinates = (vectors.T @ cross) / np.sqrt(values)
    projection = float(coordinates @ coordinates)
    if target_norm <= 0 or not len(values):
        raise ValueError("Nonzero target and training span required")
    fraction = projection / target_norm
    if fraction > 1 + 1e-5:
        raise ValueError("Projection exceeds target norm; inconsistent Gram")
    oracle = vectors @ ((vectors.T @ cross) / values)
    factor = np.sqrt(values)[:, None] * vectors.T
    positive = nnls(factor, coordinates)[0]

    def cosine(weights):
        norm = float(weights @ kernel @ weights)
        return float(weights @ cross / np.sqrt(norm * target_norm)) if norm > 0 else 0.0

    return dict(
        train_batches=len(train),
        numerical_rank=len(values),
        effective_rank=float(values.sum() ** 2 / (values @ values)),
        uniform_cosine=cosine(np.ones(len(train)) / len(train)),
        span_cosine_upper=float(np.sqrt(min(fraction, 1))),
        outside_span_energy=max(0.0, 1 - fraction),
        oracle_positive_cosine=cosine(positive),
        oracle_coefficients=oracle.tolist(),
        positive_coefficients=positive.tolist(),
        eigenvalues=values.tolist(),
        target_projection_energy_by_eigenvector=(coordinates**2).tolist(),
    )


def calibrated_reweighting(gram, train, calibration, test, *, ridge=0.01):
    """Fit positive/unconstrained coefficients on calibration gradients, evaluate disjoint test gradients."""
    gram = np.asarray(gram, dtype=np.float64)
    train = np.asarray(train, dtype=int)
    kernel = gram[np.ix_(train, train)]
    calibration = np.asarray(calibration)
    test = np.asarray(test)
    cross = gram[train] @ calibration
    values, vectors = np.linalg.eigh(kernel)
    values = np.maximum(values, 0)
    strength = ridge * np.trace(kernel) / len(kernel)
    regularized = values + max(strength, 1e-30)
    root = np.sqrt(regularized)[:, None] * vectors.T
    coordinates = (vectors.T @ cross) / np.sqrt(regularized)
    positive = nnls(root, coordinates)[0]
    signed = np.linalg.solve(kernel + max(strength, 1e-30) * np.eye(len(kernel)), cross)
    weights = dict(
        uniform=np.ones(len(train)) / len(train),
        norm_equalized=1 / np.sqrt(np.maximum(kernel.diagonal(), 1e-30)),
        positive_calibrated=positive,
        signed_calibrated=signed,
    )
    target_norm = float(test @ gram @ test)
    result = {}
    for name, w in weights.items():
        norm = w @ kernel @ w
        result[name] = float(w @ gram[train] @ test / np.sqrt(norm * target_norm)) if norm > 0 else 0.0
    return result


def cross_batch_jackknife(gram):
    """Delete-one-batch uncertainty for off-diagonal energy, not training-seed variance."""
    gram = np.asarray(gram, dtype=np.float64)
    n = len(gram)
    if n < 3:
        return {}

    def estimate(kernel):
        m = len(kernel)
        second = np.trace(kernel) / m
        cross = (kernel.sum() - np.trace(kernel)) / (m * (m - 1))
        return cross / second if second > 0 else float("nan")

    value = estimate(gram)
    leaves = np.array([estimate(np.delete(np.delete(gram, i, axis=0), i, axis=1)) for i in range(n)])
    se = float(np.sqrt((n - 1) / n * np.square(leaves - leaves.mean()).sum()))
    return dict(
        estimate=value,
        standard_error=se,
        low=value - 1.96 * se,
        high=value + 1.96 * se,
        units="delete-one-fixed-batch jackknife; approximate normal interval; no training-seed replication",
    )


def cross_receiver_projection(gram, train, receivers, *, rtol=1e-10):
    """Remove receiver self-energy from projected and full shared-gradient estimates.

    Off-diagonal receiver products distinguish finite held-out sampling noise
    from a systematic component outside the measured training span. The ratio
    is descriptive and may leave [0,1]; it is not a finite-sample cosine bound.
    """
    gram = np.asarray(gram, dtype=np.float64)
    train, receivers = np.asarray(train, dtype=int), np.asarray(receivers, dtype=int)
    if not len(train) or len(receivers) < 2:
        raise ValueError("Receiver correction requires a nonempty source span and at least two receivers")
    kernel = gram[np.ix_(train, train)]
    cross = gram[np.ix_(train, receivers)]
    held = gram[np.ix_(receivers, receivers)]
    values, vectors = np.linalg.eigh(kernel)
    keep = values > max(values.max(), 1e-30) * rtol
    coordinates = (vectors[:, keep].T @ cross) / np.sqrt(values[keep])[:, None]
    projected = coordinates.T @ coordinates

    def offdiag(matrix):
        n = len(matrix)
        return float((matrix.sum() - np.trace(matrix)) / (n * (n - 1)))

    signal = offdiag(held)
    captured = offdiag(projected)
    fraction = captured / signal if signal > 0 else float("nan")
    n = len(held)
    leaves = []
    for i in range(n) if n >= 3 else []:
        index = [j for j in range(n) if j != i]
        denominator = offdiag(held[np.ix_(index, index)])
        numerator = offdiag(projected[np.ix_(index, index)])
        leaves.append(numerator / denominator if denominator > 0 else float("nan"))
    leaves = np.array(leaves)
    se = float(np.sqrt((n - 1) / n * np.square(leaves - np.mean(leaves)).sum())) if n >= 3 else float("nan")
    return dict(
        receiver_shared_inner=signal,
        projected_receiver_shared_inner=captured,
        outside_receiver_shared_inner=signal - captured,
        projected_shared_fraction=fraction,
        fraction_jackknife_se=se,
        empirical_receiver_mean_norm_sq=float(held.mean()),
        empirical_projected_mean_norm_sq=float(projected.mean()),
        receiver_batches=n,
        note="Distinct-receiver U-statistics; ratio is not unbiased and is not clipped to [0,1].",
    )
