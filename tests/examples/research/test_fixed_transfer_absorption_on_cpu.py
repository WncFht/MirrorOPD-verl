# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Regression checks for ratio interpretation at plateaus and short budgets."""

import math

import pytest

from examples.mirroropd.research.studies.fixed_transfer.quantify_absorption import diagnose


def make_run(train, held):
    def points(values):
        return [{"step": step, "log_absorption": a, "rkl": 10 ** (-a), "prompts": 320} for step, a in values.items()]

    return {"panels": {"Training scored": points(train), "Math": points(held)}}


def test_relative_residual_identity_and_efficiency_are_distinct():
    run = make_run({0: 0, 64: 0.5, 128: 1}, {0: 0, 64: 0.25, 128: 0.5})
    q = diagnose(run, "Math", 128)
    # Training residual is 0.1, held residual is sqrt(0.1): efficiency 0.5
    # means a power-law contraction, not removal of 50% of held risk.
    assert q["held_rkl"] == pytest.approx(math.sqrt(0.1))
    assert q["efficiency"] == pytest.approx(0.5)
    assert 10 ** q["gap"] == pytest.approx(q["held_rkl"] / q["train_rkl"])
    assert q["gap"] == pytest.approx(q["train_absorption"] * (1 - q["efficiency"]))


@pytest.mark.parametrize(
    ("end_train", "status"), [(0.99, "train_rebound"), (1.0, "train_rebound"), (1.005, "small_train_progress")]
)
def test_held_improvement_with_train_plateau_is_not_a_negative_efficiency(end_train, status):
    run = make_run({0: 0, 64: 0.5, 256: 1, 320: end_train}, {0: 0, 64: 0.4, 256: 1, 320: 1.1})
    q = diagnose(run, "Math", 320)
    assert q["late_held"] == pytest.approx(0.1)
    assert q["late_efficiency"] is None
    assert q["late_status"] == status
    assert q["gap"] < 0  # Held absorption still leads.


def test_64_step_run_does_not_reuse_early_window_as_late_window():
    q = diagnose(make_run({0: 0, 64: 1}, {0: 0, 64: 0.8}), "Math", 64)
    assert q["early_efficiency"] == pytest.approx(0.8)
    assert q["late_train"] is None
    assert q["late_efficiency"] is None
    assert q["late_status"] == "insufficient_budget"


def test_no_efficiency_at_initial_checkpoint():
    q = diagnose(make_run({0: 0}, {0: 0}), "Math", 0)
    assert q["efficiency"] is None
    assert q["gap"] == 0
    assert q["early_status"] == "insufficient_budget"
