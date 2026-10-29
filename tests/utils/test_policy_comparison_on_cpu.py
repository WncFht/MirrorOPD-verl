# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Guard comparison semantics: native axes, nested timers and reuse boundaries."""

import pytest

from examples.mirroropd.research.studies.production import compare_policy_runs as module


def test_recorded_axis_preserves_within_batch_updates():
    rows = [{"step/outer": 1, "step/update": 1, "loss": 3.0}, {"step/outer": 1, "step/update": 2, "loss": 2.0}]
    axes = module.recorded_axes(rows, [{"name": "loss", "step_metric": "step/update"}])
    assert module.normalize(rows, axes)["loss"]["points"] == [(1, 3.0), (2, 2.0)]
    with pytest.raises(ValueError, match="lack recorded axes"):
        module.recorded_axes(rows, [])


def test_cost_does_not_add_nested_checkpoint_or_rollout_timers():
    rows = [
        {"step/outer": 1, "perf/timing/outer_step_seconds": 100},
        {"step/outer": 2, "perf/timing/outer_step_seconds": 60, "perf/timing/rollout_batch_seconds": 40},
        {
            "step/outer": 3,
            "perf/timing/outer_step_seconds": 80,
            "perf/timing/save_checkpoint_seconds": 20,
            "perf/timing/validation_seconds": 200,
            "perf/timing/validation_rollout_seconds": 190,
        },
    ]
    result = module.cost(rows, 3)
    assert result["accounted_hours"] == pytest.approx(440 / 3600)
    assert result["regular_records"] == 1
    assert result["regular"]["perf/timing/outer_step_seconds"]["mean"] == 60


def test_reuse_keeps_complete_update_group_even_without_outer_record():
    rows = [
        {"step/outer": 178, "step/update": 2832 + i, "rollout/data_reuse/update_index": i, "distill/full_kl": 1 / i}
        for i in range(1, 17)
    ]
    groups = module.reuse(rows, 16)
    assert len(groups) == 1 and groups[0]["last_update"] == 2848
    assert groups[0]["relative_drop"] == 15 / 16
    assert module.reuse(rows[:-1], 16) == []


def test_missing_gpu_is_not_zero_utilization():
    system = [{"_timestamp": 20, "system.gpu.0.gpu": 50}]
    outer = [{"_timestamp": 40, "step/outer": 1, "perf/timing/outer_step_seconds": 40}]
    result = module.telemetry(system, outer, 2)
    assert result["phases"]["train_trim10"]["gpu"] == {"count": 0}
