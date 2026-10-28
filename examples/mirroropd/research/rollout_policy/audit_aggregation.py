# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy<2"]
# ///
"""FP64 checks for mean/max aggregation identities; no model experiments."""

import argparse
import json
from pathlib import Path

import numpy as np


def center(p, q, alpha):
    z = alpha * np.log(p) + (1 - alpha) * np.log(q)
    z -= z.max()
    out = np.exp(z)
    return out / out.sum()


def kl(p, q):
    return float(np.dot(p, np.log(p / q)))


def solve(p, q):
    lo, hi = 0.0, 1.0
    delta = np.log(p / q)
    for _ in range(64):
        alpha = (lo + hi) / 2
        if np.dot(center(p, q, alpha), delta) < 0:
            lo = alpha
        else:
            hi = alpha
    alpha = (lo + hi) / 2
    return alpha, center(p, q, alpha)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261028)
    identity_errors = []
    max_ratio = 0.0
    rho_grid_errors = []
    for _ in range(100):
        p, q = rng.dirichlet(np.ones(7)), rng.dirichlet(np.ones(7))
        alpha, mm = solve(p, q)
        mean = center(p, q, 0.5)
        bhattacharyya = -float(np.log(np.sqrt(p * q).sum()))
        radius = max(kl(mm, p), kl(mm, q))
        identity_errors.append(abs(kl(mm, mean) - (radius - bhattacharyya)))
        max_ratio = max(max_ratio, max(kl(mean, p), kl(mean, q)) / radius)
        for rho in [0.1, 0.5, 1.0]:
            lo, hi = (1 - rho) / 2, (1 + rho) / 2
            clipped = np.clip(alpha, lo, hi)
            r = center(p, q, clipped)
            ds, dt = kl(r, p), kl(r, q)
            candidate = (ds + dt + rho * abs(ds - dt)) / 2
            grid_best = np.inf
            for a in np.linspace(0, 1, 201):
                g = center(p, q, a)
                gs, gt = kl(g, p), kl(g, q)
                grid_best = min(grid_best, (gs + gt + rho * abs(gs - gt)) / 2)
            rho_grid_errors.append(candidate - grid_best)

    p = np.array([0.8, 0.19, 0.01])
    q = np.array([0.1, 0.2, 0.7])
    a, mm = solve(p, q)
    example = []
    for name, alpha in [("mean", 0.5), ("rho=0.1", float(np.clip(a, 0.45, 0.55))), ("minimax", a)]:
        r = center(p, q, alpha)
        ds, dt = kl(r, p), kl(r, q)
        example.append(
            {
                "name": name,
                "student_weight": alpha,
                "Q": r.tolist(),
                "KL_student": ds,
                "KL_teacher": dt,
                "mean": (ds + dt) / 2,
                "maximum": max(ds, dt),
            }
        )

    result = {
        "random_cases": 100,
        "max_gap_identity_error": max(identity_errors),
        "largest_observed_mean_center_worst_radius_ratio": max_ratio,
        "max_clipped_alpha_objective_excess_vs_grid": max(rho_grid_errors),
        "example": example,
        "scope": "CPU FP64 synthetic distributions only",
    }
    assert max(identity_errors) < 1e-10
    assert max_ratio <= 2 + 1e-10
    assert max(rho_grid_errors) < 1e-10
    out = args.out
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
