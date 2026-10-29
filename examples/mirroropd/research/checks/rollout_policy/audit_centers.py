# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pyBregMan @ https://github.com/alexandersoen/pyBregMan/archive/6f375e4d38e8ff798801bf5d34572ed52b1fadbb.tar.gz",
# ]
# ///
"""Audit distribution centers and the pinned upstream Chernoff implementation.

CPU-only numerical examples; these are not LLM accuracy experiments.
Run with uv run examples/mirroropd/research/checks/rollout_policy/audit_centers.py.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def kl(p, q):
    return float(np.dot(p, np.log(p / q)))


def hellinger2(p, q):
    return float(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2) / 2)


def geometric(p, q, alpha):
    from scipy.special import logsumexp

    z = alpha * np.log(p) + (1 - alpha) * np.log(q)
    return np.exp(z - logsumexp(z))


def chernoff(p, q):
    lo, hi = 0.0, 1.0
    delta = np.log(p / q)
    for _ in range(64):
        alpha = (lo + hi) / 2
        r = geometric(p, q, alpha)
        if np.dot(r, delta) < 0:
            lo = alpha
        else:
            hi = alpha
    alpha = (lo + hi) / 2
    return alpha, geometric(p, q, alpha)


def forward_center(p, q):
    lo, hi = 0.0, 1.0
    for _ in range(64):
        weight = (lo + hi) / 2
        r = weight * p + (1 - weight) * q
        if kl(p, r) > kl(q, r):
            lo = weight
        else:
            hi = weight
    weight = (lo + hi) / 2
    return weight, weight * p + (1 - weight) * q


def hcenter(p, q):
    r = (np.sqrt(p) + np.sqrt(q)) ** 2
    return r / r.sum()


def numerical_minimax(p, q, div):
    from scipy.optimize import minimize

    def objective(x):
        return x[-1]

    def constraints(x):
        return np.array([x[-1] - div(x[:-1], p), x[-1] - div(x[:-1], q)])

    start = (p + q) / 2
    x0 = np.r_[start, max(div(start, p), div(start, q))]
    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=[(1e-12, 1.0)] * len(p) + [(0.0, None)],
        constraints=[{"type": "eq", "fun": lambda x: x[:-1].sum() - 1}, {"type": "ineq", "fun": constraints}],
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.x[:-1], result.fun


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from bregman.application.distribution.exponential_family.multinomial.multinomial import MultinomialManifold
    from bregman.base import LAMBDA_COORDS, Point
    from bregman.dissimilarity.bregman import ChernoffInformation
    from scipy.special import logsumexp

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261028)
    errors, herrors = [], []
    for _ in range(40):
        p, q = rng.dirichlet(np.ones(5)), rng.dirichlet(np.ones(5))
        manifold = MultinomialManifold(k=5, n=1)
        upstream = ChernoffInformation(manifold)
        a_upstream = upstream.chernoff_point(Point(LAMBDA_COORDS, p), Point(LAMBDA_COORDS, q), eps=1e-11)
        a_ref, r_ref = chernoff(p, q)
        errors.append(abs(a_ref - a_upstream))
        h = hcenter(p, q)
        h_opt, h_radius = numerical_minimax(p, q, hellinger2)
        herrors.append(abs(max(hellinger2(h, p), hellinger2(h, q)) - h_radius))

    p = np.array([0.8, 0.19, 0.01])
    q = np.array([0.1, 0.2, 0.7])
    a, r = chernoff(p, q)
    fweight, f = forward_center(p, q)
    h = hcenter(p, q)
    upstream = ChernoffInformation(MultinomialManifold(k=3, n=1))
    point_p, point_q = Point(LAMBDA_COORDS, p), Point(LAMBDA_COORDS, q)
    lib_a = upstream.chernoff_point(point_p, point_q, eps=1e-11)
    lib_value = float(upstream.dissimilarity(point_p, point_q, eps=1e-11))
    opposite_exponent_a = 1 - lib_a
    scaled_opposite_chernoff = -float(
        logsumexp(opposite_exponent_a * np.log(p) + (1 - opposite_exponent_a) * np.log(q))
    ) / (lib_a * (1 - lib_a))

    tiny = 1e-12
    tail_p = np.array([0.999, tiny, 0.001 - tiny])
    tail_q = np.array([tiny, 0.999, 0.001 - tiny])
    tail_a, tail_r = chernoff(tail_p, tail_q)

    result = {
        "upstream_commit": "6f375e4d38e8ff798801bf5d34572ed52b1fadbb",
        "random_cases": 40,
        "max_upstream_alpha_error": max(errors),
        "max_hellinger_closed_form_radius_error_vs_slsqp": max(herrors),
        "example": {
            "student": p.tolist(),
            "teacher": q.tolist(),
            "chernoff_alpha": a,
            "chernoff_Q": r.tolist(),
            "chernoff_radius": kl(r, p),
            "forward_minimax_weight": fweight,
            "forward_minimax_Q": f.tolist(),
            "hellinger_Q": h.tolist(),
            "hellinger_radius": hellinger2(h, p),
            "upstream_alpha": lib_a,
            "upstream_dissimilarity_value": lib_value,
            "scaled_opposite_exponent_value": scaled_opposite_chernoff,
        },
        "rare_shared_event_counterexample": {
            "student": tail_p.tolist(),
            "teacher": tail_q.tolist(),
            "chernoff_alpha": tail_a,
            "chernoff_Q": tail_r.tolist(),
            "chernoff_radius": kl(tail_r, tail_p),
            "hellinger_Q": hcenter(tail_p, tail_q).tolist(),
            "arithmetic_Q": ((tail_p + tail_q) / 2).tolist(),
        },
        "scope": "CPU FP64 probability examples; no model training or GPU performance claims",
    }
    assert max(errors) < 1e-8
    assert max(herrors) < 1e-8
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
