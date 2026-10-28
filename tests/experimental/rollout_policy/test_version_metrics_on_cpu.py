# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
import pytest

from verl.experimental.rollout_policy.version_metrics import rollout_version_metrics


def test_frozen_source_has_no_fabricated_actor_age():
    frozen = {"min_global_steps": None, "max_global_steps": None, "rollout_has_current_component": False}
    current = {"min_global_steps": 2, "max_global_steps": 2, "rollout_has_current_component": True}
    mixed = rollout_version_metrics([frozen, current], 3)
    assert mixed["rollout/policy/current_component_version_fraction"] == 0.5
    assert mixed["training/off_policy/trajectory_staleness/mean"] == 0
    assert rollout_version_metrics([frozen], 3) == {"rollout/policy/current_component_version_fraction": 0}


def test_default_version_metrics_and_missing_current_version():
    metrics = rollout_version_metrics([{"min_global_steps": 1, "max_global_steps": 2}], 4)
    assert metrics["training/off_policy/trajectory_spans/mean"] == 2
    assert metrics["training/off_policy/trajectory_staleness/mean"] == 1
    assert metrics["training/off_policy/trajectory_staleness_worst/mean"] == 2
    assert not any(key.startswith("rollout/policy") for key in metrics)
    with pytest.raises(ValueError, match="missing"):
        rollout_version_metrics([{"min_global_steps": None, "max_global_steps": None}], 1)
