# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Prompt-level bootstrap; callers aggregate repeated rollouts before sampling."""

import random
import statistics


def mean_ci(values, n=2000, *, seed=42):
    """Historical percentile convention, preserving each study's draw count."""
    if not values or n < 40:
        raise ValueError("Bootstrap requires nonempty values and at least 40 draws")
    rng = random.Random(seed)
    draws = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(n))
    return [draws[int(n * 0.025) - 1], draws[int(n * 0.975) - 1]]


def ratio_ci(before, after, n=2000, *, seed=42):
    if len(before) != len(after) or not before or n < 40:
        raise ValueError("Ratio bootstrap requires paired, nonempty observations and at least 40 draws")
    rng = random.Random(seed)
    draws = []
    for _ in range(n):
        ids = rng.choices(range(len(before)), k=len(before))
        denominator = sum(before[i] for i in ids)
        if denominator <= 0:
            raise ValueError("Ratio bootstrap denominator must be positive")
        draws.append(sum(after[i] for i in ids) / denominator)
    draws.sort()
    return [draws[int(n * 0.025) - 1], draws[int(n * 0.975) - 1]]


def paired_mean_difference(before, after, n=5000, *, seed=42):
    """Percentile CI of paired sample differences; does not represent training-seed variation."""
    import numpy as np

    before, after = np.asarray(before, dtype=float), np.asarray(after, dtype=float)
    if before.ndim != 1 or before.shape != after.shape or before.size == 0 or n < 40:
        raise ValueError("Paired bootstrap requires equal nonempty vectors and at least 40 draws")
    if not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError("Paired bootstrap requires finite observations")
    delta = after - before
    rng = np.random.default_rng(seed)
    draws = delta[rng.integers(len(delta), size=(n, len(delta)))].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return dict(
        before=float(before.mean()),
        after=float(after.mean()),
        delta=float(delta.mean()),
        ci_low=float(low),
        ci_high=float(high),
        increased_prompts=int((delta > 0).sum()),
    )
