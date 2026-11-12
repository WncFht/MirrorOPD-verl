# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Trace-normalized spectral distances (arXiv:2605.26489), without new SVDs."""

import numpy as np


def distribution(values):
    """Return descending singular probabilities and scale; zero spectra are undefined."""
    s = np.asarray(values, dtype=np.float64)
    if s.ndim != 1 or not len(s) or not np.isfinite(s).all() or (s < 0).any():
        raise ValueError("Expected a nonempty finite nonnegative singular spectrum")
    if (np.diff(s) > 1e-12 * max(1.0, float(s[0]))).any():
        raise ValueError("Singular spectrum must be descending")
    total = float(s.sum())
    return dict(
        probabilities=(s / total).tolist() if total else None,
        nuclear_norm=total,
        frobenius_norm=float(np.linalg.norm(s)),
    )


def distance(a, b):
    """Compare equal-dimensional spectra; never truncate or pad cross-size teachers."""
    if a["probabilities"] is None or b["probabilities"] is None:
        return None
    p, q = np.asarray(a["probabilities"]), np.asarray(b["probabilities"])
    if p.shape != q.shape:
        raise ValueError("Spectral distances require matching singular dimensions")
    d = float(np.linalg.norm(p - q))
    return dict(sd_variation=d, hellinger=float(np.linalg.norm(np.sqrt(p) - np.sqrt(q)) / np.sqrt(2)))


def trajectory(points):
    """All checkpoint pairs plus interval net speed; not an optimizer-step path length."""
    steps = [p["step"] for p in points]
    if steps != sorted(set(steps)):
        raise ValueError("Checkpoint steps must be unique and increasing")
    comparisons = [
        dict(step_a=a["step"], step_b=b["step"], **distance(a, b))
        for a in points
        for b in points
        if a["step"] <= b["step"]
    ]
    intervals = []
    for a, b in zip(points, points[1:], strict=False):
        gap = b["step"] - a["step"]
        d = distance(a, b)
        intervals.append(
            dict(start=a["step"], end=b["step"], optimizer_interval=gap, **d, net_speed=d["sd_variation"] / gap)
        )
    return dict(points=points, pairs=comparisons, intervals=intervals)


def measure(record):
    """Preserve exact normalized vectors for CPU-only independent recomputation."""
    initial = dict(step=0, **distribution(record["initial_singular_values"]))
    weights = [initial] + [
        dict(step=s["step"], **distribution(s["weight"]["singular_values"])) for s in record["stages"]
    ]
    updates = [dict(step=s["step"], **distribution(s["singular_values"])) for s in record["stages"]]
    # The production geometry engine requires nonzero cumulative updates.
    if any(p["probabilities"] is None for p in weights + updates):
        raise ValueError("Zero checkpoint spectrum cannot define a singular distribution")
    result = dict(schema="trace-normalized-spectrum-v1", weight=trajectory(weights), update=trajectory(updates))
    if record.get("teacher"):
        teacher = record["teacher"]
        result["teacher_weight"] = distribution(teacher["weight"]["singular_values"])
        result["teacher_update"] = distribution(teacher["singular_values"]) if "singular_values" in teacher else None
        baseline = distance(initial, result["teacher_weight"])["sd_variation"]
        result["teacher_distances"] = [
            dict(
                step=p["step"],
                **distance(p, result["teacher_weight"]),
                remaining_fraction=distance(p, result["teacher_weight"])["sd_variation"] / baseline
                if baseline > 1e-12
                else None,
            )
            for p in weights
        ]
        if result["teacher_update"] and result["teacher_update"]["probabilities"] is not None:
            result["teacher_update_distances"] = [
                dict(step=p["step"], **distance(p, result["teacher_update"])) for p in updates
            ]
    return result
