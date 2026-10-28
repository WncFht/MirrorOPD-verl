# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
import dataclasses

import pytest

from verl.experimental.rollout_policy.plan import CoefficientRule, RolloutPolicyConfig, build_plan


def plan(tmp_path, policy, n=1, ids=None, optimizer_step=7):
    return build_plan(
        policy,
        uids=ids or [f"uuid-{i}" for i in range(64)],
        stable_ids=list(range(64)),
        n=n,
        collection_step=3,
        optimizer_step=optimizer_step,
        seed=42,
        student_path=str(tmp_path),
        teacher_path=str(tmp_path),
    )


@pytest.mark.parametrize("student", ["current", "initial"])
@pytest.mark.parametrize("n", [1, 3])
def test_global_quota_and_uuid_independence(tmp_path, student, n):
    config = RolloutPolicyConfig.parse({"mode": "trajectory_mixture", "student": student})
    first = plan(tmp_path, config, n)
    second = plan(tmp_path, config, n, ids=[f"other-{i}" for i in range(64)])
    assert sum(r.source == "teacher" for r in first.requests) == 32 * n
    assert [(r.source, r.seed) for r in first.requests] == [(r.source, r.seed) for r in second.requests]
    assert len({r.key for phase in first.phases for r in phase.requests}) == 64 * n
    if n > 1:
        assert any(len({r.source for r in first.requests if r.row == i}) == 2 for i in range(64))
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.resolved_control = 1


@pytest.mark.parametrize("alpha,count", [(0, 0), (1, 64), (0.3, 19), (0.001, 0)])
def test_quota_rounding_and_empty_phase(tmp_path, alpha, count):
    result = plan(tmp_path, RolloutPolicyConfig.parse({"mode": "trajectory_mixture", "teacher_weight": alpha}))
    assert sum(r.source == "teacher" for r in result.requests) == count
    assert len(result.phases) == (2 if 0 < count < 64 else 1)


def test_schedule_applied_clock_and_unknown_progress(tmp_path):
    config = RolloutPolicyConfig.parse(
        {
            "mode": "token_mixture",
            "teacher_weight": {"kind": "linear", "start": 0.5, "end": 0.0, "steps": 10, "clock": "optimizer_step"},
        }
    )
    assert plan(tmp_path, config).resolved_control == pytest.approx(0.15)
    with pytest.raises(ValueError, match="unknown"):
        plan(tmp_path, config, optimizer_step=None)
    assert CoefficientRule().resolve(0, None) == 0.5


@pytest.mark.parametrize(
    "values",
    [
        {"mode": "unknown"},
        {"teacher_weight": float("nan")},
        {"teacher_weight": -1},
        {"student": "latest"},
        {"source": "checkpoint"},
        {"mode": "token_mixture", "source": "teacher"},
        {"teacher_weight": {"kind": "trb"}},
        {"deadline_seconds": float("inf")},
    ],
)
def test_invalid_configuration(values):
    with pytest.raises((ValueError, TypeError)):
        RolloutPolicyConfig.parse(values)


def test_actor_clock_counts_completed_outer_steps(tmp_path):
    config = RolloutPolicyConfig.parse(
        {
            "mode": "token_mixture",
            "teacher_weight": {"kind": "linear", "start": 0.5, "end": 0.0, "steps": 10, "clock": "outer_step"},
        }
    )
    result = plan(tmp_path, config)
    assert result.actor_version == (2, 7)
    assert result.resolved_control == pytest.approx(0.4)


def test_extra_checkpoint_is_bound_in_the_immutable_plan(tmp_path):
    extra = tmp_path / "extra"
    extra.mkdir()
    result = plan(tmp_path, RolloutPolicyConfig.parse({"source": "checkpoint", "model_path": str(extra)}))
    assert result.source_binding.path == str(extra)
    assert result.source_binding.kind == "checkpoint"
    assert result.student_binding.path == str(tmp_path)


@pytest.mark.parametrize("source,weight", [("student", 0.0), ("teacher", 1.0)])
def test_single_source_has_canonical_control(tmp_path, source, weight):
    result = plan(tmp_path, RolloutPolicyConfig.parse({"source": source}))
    assert result.resolved_control == weight
    assert result.coefficient_rule.start == weight


@pytest.mark.parametrize("alpha,source", [(0.0, "student"), (1.0, "teacher")])
def test_token_endpoints_preserve_numerical_executor_and_pure_source_tags(tmp_path, alpha, source):
    result = plan(tmp_path, RolloutPolicyConfig.parse({"mode": "token_mixture", "teacher_weight": alpha}))
    assert [phase.decoder for phase in result.phases] == ["fusion"]
    assert all(request.source == source for request in result.requests)
