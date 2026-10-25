# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Evaluation denominators and canonical output schema."""

import csv
from pathlib import Path

import pytest

from verl.utils.metric.schema import evaluation_metrics, metric_axis, outer_metrics


def test_every_observed_old_key_is_removed_from_public_projection():
    path = (
        Path(__file__).parents[2]
        / "examples/mirroropd/docs/archive/metric-implementation/data/metrics-system-20261021"
        / "current-metric-inventory.csv"
    )
    old = {
        row["metric_key"]: 1.0
        for row in csv.DictReader(path.open())
        if not row["metric_key"].startswith(("val-core/", "val-aux/"))
    }
    new = outer_metrics(old)
    assert not new.keys() & old.keys()
    assert all(
        key.split("/")[0] in {"optim", "policy", "advantage", "correction", "distill", "rollout", "perf", "step"}
        for key in new
    )
    assert new["perf/actor/mfu"] == 1
    assert "optim/actor/loss" not in new  # never rename an outer mean to an update observation


def test_failed_sessions_and_unequal_prompt_trial_counts():
    result = evaluation_metrics(
        ["MATH-500"] * 4,
        ["a", "a", "b", "b"],
        {"acc": [1, 1, 0, 1]},
        {("MATH-500", "a"): 2, ("MATH-500", "b"): 4, ("MATH-500", "c"): 2},
    )
    assert {
        k: result[k] for k in ("eval/math500/acc/mean", "eval/math500/failed_count", "eval/math500/response_count")
    } == {
        "eval/math500/acc/mean": (1 + 0.25 + 0) / 3,
        "eval/math500/failed_count": 4,
        "eval/math500/response_count": 8,
    }


def test_wrong_answers_are_not_failed_sessions():
    result = evaluation_metrics(["AMC23"] * 2, ["a", "a"], {"acc": [0, 0]}, {("AMC23", "a"): 2})
    assert result["eval/amc23/failed_count"] == 0
    assert result["eval/amc23/acc/mean"] == 0


def test_all_failed_validation_keeps_the_denominator():
    result = evaluation_metrics([], [], {}, {("AIME25", "a"): 16})
    assert result["eval/aime25/acc/mean"] == 0
    assert result["eval/aime25/failed_count"] == result["eval/aime25/response_count"] == 16
    assert result["eval/aime25/acc/pass@16"] == 0


def test_dataset_name_collision_and_excess_trials_fail():
    with pytest.raises(ValueError, match="collision"):
        evaluation_metrics(["a-b", "a_b"], ["a", "a"], {"acc": [1, 0]})
    with pytest.raises(ValueError, match="More validation"):
        evaluation_metrics(["x", "x"], ["a", "a"], {"acc": [1, 0]}, {("x", "a"): 1})


def test_axes_follow_measurement_not_transport_step():
    assert metric_axis("policy/current_behavior/ratio_p99") == "step/update"
    assert metric_axis("policy/anchor_behavior/k3") == "step/outer"
    assert metric_axis("policy/entropy/pre_update") == "step/outer"
    assert metric_axis("eval/math500/acc/mean") == "step/optimizer"


def test_validation_mean_matches_the_existing_estimator():
    from verl.trainer.ppo.metric_utils import process_validation_metrics

    sources = ["math"] * 5
    uids = ["a", "a", "a", "b", "b"]
    infos = {"acc": [1, 0, 1, 1, 0]}
    expected = {("math", "a"): 4, ("math", "b"): 4, ("math", "c"): 4}
    reference = process_validation_metrics(sources, uids, infos, expected_acc_counts=expected)
    actual = evaluation_metrics(sources, uids, infos, expected)
    assert actual["eval/math/acc/mean"] == reference["math"]["acc"]["mean@4"]


def test_non_accuracy_reward_is_not_mislabeled_as_accuracy():
    actual = evaluation_metrics(["reward_task"], ["a"], {"reward": [2.5]})
    assert actual["eval/reward_task/reward/mean"] == 2.5
    assert "eval/reward_task/acc/mean" not in actual


def test_conditional_fields_are_preserved_in_new_namespaces():
    from verl.utils.metric.schema import canonical_name

    cases = {
        "actor/pg_loss": "optim/actor/pg_loss",
        "actor/distillation/student_mass": "distill/topk/student_mass",
        "critic/vf_loss": "optim/critic/value_loss",
        "critic/vf_explained_var": "policy/critic/explained_variance",
        "rollout_corr/rollout_rs_masked_fraction": "correction/rejection/batch/masked_fraction",
        "rollout_corr/rollout_is_eff_sample_size": "correction/is/batch/applied_ess_fraction",
        "timing_s/new_stage": "perf/timing/new_stage_seconds",
        "training/off_policy/trajectory_staleness/max": "rollout/staleness/trajectory_staleness/max",
        "policy/new_pair/any_stat": "policy/new_pair/any_stat",
    }
    for old, new in cases.items():
        assert canonical_name(old) == new
        assert outer_metrics({old: 2}) == {new: 2}
    assert outer_metrics({"extension/custom": 3}) == {"extension/custom": 3}


def test_invalid_metrics_have_visible_events():
    from verl.utils.metric.schema import finite_metrics

    assert finite_metrics({"optim/actor/loss": float("nan"), "policy/pair/m2": None}) == {
        "optim/actor/loss/nonfinite": 1,
        "policy/pair/m2/undefined": 1,
    }


def test_helper_pair_names_follow_measurement_location():
    from verl.utils.metric.schema import canonical_name

    assert canonical_name("rollout_corr/k3_kl") == "policy/anchor_behavior/k3"
    assert canonical_name("rollout_corr/k3_kl", role="actor") == "policy/current_behavior/local_summary/k3"
    assert canonical_name("actor/reward_kl_penalty") == "policy/anchor_reference/reward_kl"
    assert metric_axis("policy/anchor_reference/reward_kl") == "step/outer"
