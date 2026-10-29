# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Paired prompt bootstrap and unclipped task GapRecovery, shared across study families."""

import numpy as np


def summarize_gap(before, teacher, after, rng, replicates=5000):
    keys = list(before)
    assert keys == list(teacher) == list(after)
    arrays = np.array([[before[k], teacher[k], after[k]] for k in keys])
    boots = arrays[rng.integers(0, len(keys), size=(replicates, len(keys)))].mean(1)
    return arrays.mean(0), boots


def gap_row(means, boots):
    b, t, a = means
    den = t - b
    delta = a - b
    bd = boots[:, 1] - boots[:, 0]
    ba = boots[:, 2] - boots[:, 0]
    dlo, dhi = np.quantile(bd, [0.025, 0.975])
    stable = bool(dlo > 0 or dhi < 0)
    zero_fraction = float(np.mean(np.abs(bd) <= 1e-12))
    ratio_defined = stable and zero_fraction == 0
    ratio_ci = np.quantile(100 * ba / bd, [0.025, 0.975]) if ratio_defined else [None, None]
    return dict(
        before_acc=100 * b,
        teacher_acc=100 * t,
        after_acc=100 * a,
        gain_pp=100 * delta,
        teacher_advantage_pp=100 * den,
        gap_recovery_percent=100 * delta / den if abs(den) > 1e-12 else None,
        denominator_ci_low_pp=100 * dlo,
        denominator_ci_high_pp=100 * dhi,
        gain_ci_low_pp=100 * np.quantile(ba, 0.025),
        gain_ci_high_pp=100 * np.quantile(ba, 0.975),
        ratio_ci_low=ratio_ci[0],
        ratio_ci_high=ratio_ci[1],
        interpretation="positive_teacher_gap"
        if den > 1e-12
        else ("teacher_below_student" if den < -1e-12 else "undefined_zero_gap"),
        denominator_ci_crosses_zero=not stable,
        ratio_ci_defined=ratio_defined,
        bootstrap_zero_denominator_fraction=zero_fraction,
    )
