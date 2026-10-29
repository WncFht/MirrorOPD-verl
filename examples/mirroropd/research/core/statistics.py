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
