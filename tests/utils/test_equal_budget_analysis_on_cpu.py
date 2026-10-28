# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Protect equal-update windows, nonoverlapping cost and pre-update semantics."""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "examples/mirroropd/research/analysis"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("equal_budget", SCRIPTS / "analyze_equal_budget.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_budget_caps_outer_and_does_not_double_count_actor_or_checkpoint():
    rows = [
        {
            "step/outer": i,
            "step/optimizer": 2 * i,
            "rollout/response_count": 4,
            "rollout/response_tokens": 100 * i,
            "rollout/score/mean": 0.5,
            "rollout/finish_reason/length_count": 1,
            "perf/timing/outer_step_seconds": 20,
            "perf/timing/update_actor_seconds": 10,
            "perf/timing/save_checkpoint_seconds": 3,
            "perf/timing/validation_seconds": 5,
        }
        for i in [1, 2]
    ]
    result = module.budget(rows, [{"step/optimizer": i} for i in range(1, 5)], 2, 2)
    assert (result["complete_outer"], result["updates"], result["responses"]) == (1, 2, 4)
    assert result["response_tokens"] == 100 and result["reused_response_tokens"] == 200
    assert result["time"]["accounted_hours"] == 25 / 3600
    assert result["actor_seconds_per_update"] == 5


def test_reuse_window_excludes_future_batch_and_averages_batch_ratios():
    batches = []
    for i, values in enumerate([[10, 5], [2, 0.2], [100, 50]], 1):
        batches.append(
            {
                "outer": i,
                "values": values,
                "first": values[0],
                "last": values[-1],
                "first_update": i * 2 - 1,
                "last_update": i * 2,
                "relative_drop": 1 - values[-1] / values[0],
                "increases": 0,
            }
        )
    updates = [{"step/update": i, "optim/actor/grad_norm_preclip": 1} for i in range(1, 7)]
    result = module.reuse_window(batches, updates, 1, 4)
    assert result["batches"] == 2
    assert result["next_batch_rebound"]["count"] == 1
    assert result["next_batch_rebound"]["mean"] == 2 / 5
    assert result["within_drop"]["mean"] == pytest.approx(0.7)
    assert result["normalized_profile"] == pytest.approx([1, 0.3])
    assert result["adjacent_transitions"] == 2  # One observed transition per two-update batch.


def test_update_audit_rejects_skipped_and_mislabelled_updates():
    config = {"actor_rollout_ref": {"actor": {"ppo_epochs": 2}}}
    updates = [
        {
            "step/update": i,
            "step/optimizer": i,
            "step/outer": 1,
            "rollout/data_reuse/update_index": i,
            "optim/actor/update_applied": 1,
            "optim/actor/update_skipped": 0,
            "distill/full_kl": 1 / i,
            "optim/actor/grad_norm_preclip": 1,
        }
        for i in [1, 2]
    ]
    outer = {"step/outer": 1, "step/optimizer": 2, "perf/timing/outer_step_seconds": 10}
    module.audit_run({"exit_code": None}, config, updates + [outer])
    updates[1]["optim/actor/update_skipped"] = 1
    with pytest.raises(AssertionError):
        module.audit_run({"exit_code": None}, config, updates + [outer])
    updates[1]["optim/actor/update_skipped"] = 0
    outer["step/optimizer"] = 1
    with pytest.raises(AssertionError):
        module.audit_run({"exit_code": None}, config, updates + [outer])
