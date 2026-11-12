# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU-only summaries with explicit objects, intervals and aggregation units."""

import numpy as np

MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
RANKS = (16, 32, 64, 128, 256)


def angle_distribution(angles):
    """Exact ECDF at fixed angles; interpolated quantiles retain curve shape compactly."""
    values = np.asarray(angles, dtype=float)
    if not len(values) or not np.isfinite(values).all() or (values < 0).any() or (values > 90 + 1e-8).any():
        raise ValueError("Principal angles must be finite and between zero and 90 degrees")
    probabilities = np.linspace(0, 1, 33)
    degrees = np.arange(0, 91, 5)
    return dict(
        count=len(values),
        probabilities=probabilities.tolist(),
        quantiles_deg=np.quantile(values, probabilities).tolist(),
        ecdf_degrees=degrees.tolist(),
        ecdf_fraction=[float(np.mean(values <= d)) for d in degrees],
    )


def spectrum_metrics(point, initial):
    """Separate uniform singular scaling from residual shape; no rank truncation."""
    s = np.asarray(point["probabilities"]) * point["nuclear_norm"]
    b = np.asarray(initial["probabilities"]) * initial["nuclear_norm"]
    alpha = float(np.dot(b, s) / np.dot(b, b))
    energy = s * s
    p = energy / energy.sum()
    return dict(
        step=point["step"],
        scale=alpha,
        shape_residual=float(np.linalg.norm(s - alpha * b) / np.linalg.norm(s)),
        entropy_rank=float(np.exp(-np.sum(p[p > 0] * np.log(p[p > 0])))),
        maximum_perturbation=float(np.max(np.abs(s - b)) / b[0]),
    )


def derive(sigma):
    return {
        kind: [spectrum_metrics(p, trajectory["points"][0]) for p in trajectory["points"]]
        for kind, trajectory in sigma.items()
        if kind in ("weight", "update")
    }


def relation(matrix, a, b, rank):
    a, b = sorted((a, b))
    return next(p for p in matrix["pairs"] if p["step_a"] == a and p["step_b"] == b and p["requested_rank"] == rank)


def rank_record(stage, rank):
    return next(p for p in stage["ranks"] if p["requested_rank"] == rank)
