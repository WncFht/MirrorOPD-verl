# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Evaluation denominators and canonical output schema."""

import csv
from pathlib import Path

import pytest

from verl.utils.metric.schema import CANONICAL_FIELDS, evaluation_metrics, metric_axis, outer_metrics


def test_every_observed_old_key_is_removed_from_public_projection():
    path = Path(__file__).parents[2] / "docs/metrics/data/metrics-system-20261021/current-metric-inventory.csv"
    old = {row["metric_key"]: 1.0 for row in csv.DictReader(path.open())}
    new = outer_metrics(old)
    assert not new.keys() & old.keys()
    assert set(new) <= CANONICAL_FIELDS
    assert new["perf/actor/mfu"] == 1
    assert "optim/actor/loss" not in new  # never rename an outer mean to an update observation


def test_failed_sessions_and_unequal_prompt_trial_counts():
    result = evaluation_metrics(
        ["MATH-500"] * 4,
        ["a", "a", "b", "b"],
        {"acc": [1, 1, 0, 1]},
        {("MATH-500", "a"): 2, ("MATH-500", "b"): 4, ("MATH-500", "c"): 2},
    )
    assert result == {
        "eval/math500/acc_mean": (1 + 0.25 + 0) / 3,
        "eval/math500/failed_count": 4,
        "eval/math500/response_count": 8,
    }


def test_wrong_answers_are_not_failed_sessions():
    result = evaluation_metrics(["AMC23"] * 2, ["a", "a"], {"acc": [0, 0]}, {("AMC23", "a"): 2})
    assert result["eval/amc23/failed_count"] == 0
    assert result["eval/amc23/acc_mean"] == 0


def test_all_failed_validation_keeps_the_denominator():
    result = evaluation_metrics([], [], {}, {("AIME25", "a"): 16})
    assert result == {"eval/aime25/acc_mean": 0, "eval/aime25/failed_count": 16, "eval/aime25/response_count": 16}


def test_dataset_name_collision_and_excess_trials_fail():
    with pytest.raises(ValueError, match="collision"):
        evaluation_metrics(["a-b", "a_b"], ["a", "a"], {"acc": [1, 0]})
    with pytest.raises(ValueError, match="More validation"):
        evaluation_metrics(["x", "x"], ["a", "a"], {"acc": [1, 0]}, {("x", "a"): 1})


def test_axes_follow_measurement_not_transport_step():
    assert metric_axis("policy/current_behavior/ratio_p99") == "step/update"
    assert metric_axis("policy/anchor_behavior/k3") == "step/outer"
    assert metric_axis("policy/entropy/pre_update") == "step/outer"
    assert metric_axis("eval/math500/acc_mean") == "step/optimizer"
