# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
import dataclasses

import pytest

from verl.experimental.rollout_policy.plan import CoefficientRule, RolloutPolicyConfig, build_plan


def plan(tmp_path, policy, n=1, ids=None, optimizer_step=7):
    (tmp_path / "teacher").mkdir(exist_ok=True)
    return build_plan(
        policy,
        uids=ids or [f"uuid-{i}" for i in range(64)],
        stable_ids=list(range(64)),
        n=n,
        collection_step=3,
        optimizer_step=optimizer_step,
        seed=42,
        student_path=str(tmp_path),
        teacher_path=str(tmp_path / "teacher"),
    )


@pytest.mark.parametrize("student", ["current", "initial"])
@pytest.mark.parametrize("n", [1, 3])
def test_global_quota_and_uuid_independence(tmp_path, student, n):
    config = RolloutPolicyConfig.parse({"mode": "trajectory_mixture", "left": {"kind": student}})
    first = plan(tmp_path, config, n)
    second = plan(tmp_path, config, n, ids=[f"other-{i}" for i in range(64)])
    assert sum(r.source == "right" for r in first.requests) == 32 * n
    assert [(r.source, r.seed) for r in first.requests] == [(r.source, r.seed) for r in second.requests]
    assert len({r.key for phase in first.phases for r in phase.requests}) == 64 * n
    if n > 1:
        assert any(len({r.source for r in first.requests if r.row == i}) == 2 for i in range(64))
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.resolved_control = 1


@pytest.mark.parametrize("alpha,count", [(0, 0), (1, 64), (0.3, 19), (0.001, 0)])
def test_quota_rounding_and_empty_phase(tmp_path, alpha, count):
    result = plan(tmp_path, RolloutPolicyConfig.parse({"mode": "trajectory_mixture", "right_weight": alpha}))
    assert sum(r.source == "right" for r in result.requests) == count
    assert len(result.phases) == (2 if 0 < count < 64 else 1)


def test_schedule_applied_clock_and_unknown_progress(tmp_path):
    config = RolloutPolicyConfig.parse(
        {
            "mode": "token_mixture",
            "right_weight": {"kind": "linear", "start": 0.5, "end": 0.0, "steps": 10, "clock": "optimizer_step"},
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
        {"right_weight": float("nan")},
        {"right_weight": -1},
        {"student": "latest"},
        {"source": "checkpoint"},
        {"mode": "token_mixture", "source": "teacher"},
        {"right_weight": {"kind": "trb"}},
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
            "right_weight": {"kind": "linear", "start": 0.5, "end": 0.0, "steps": 10, "clock": "outer_step"},
        }
    )
    result = plan(tmp_path, config)
    assert result.actor_version == (2, 7)
    assert result.resolved_control == pytest.approx(0.4)


def test_extra_checkpoint_is_bound_in_the_immutable_plan(tmp_path):
    extra = tmp_path / "extra"
    extra.mkdir()
    result = plan(tmp_path, RolloutPolicyConfig.parse({"model": {"kind": "checkpoint", "ref": str(extra)}}))
    assert result.binding("model").path == str(extra)
    assert result.binding("model").kind == "checkpoint"


@pytest.mark.parametrize("kind", ["current", "initial", "teacher"])
def test_single_source_has_canonical_control(tmp_path, kind):
    result = plan(tmp_path, RolloutPolicyConfig.parse({"model": {"kind": kind}}))
    assert result.resolved_control == 0
    assert [p.decoder for p in result.phases] == ["model"]


@pytest.mark.parametrize("alpha,source", [(0.0, "left"), (1.0, "right")])
def test_token_endpoints_are_single_executors(tmp_path, alpha, source):
    result = plan(tmp_path, RolloutPolicyConfig.parse({"mode": "token_mixture", "right_weight": alpha}))
    assert [phase.decoder for phase in result.phases] == [source]
    assert result.resolved_mode == "single"
    assert result.normalization == "endpoint"
    assert all(request.source == source for request in result.requests)


@pytest.mark.parametrize("mode", ["trajectory_mixture", "token_mixture"])
def test_arbitrary_fixed_pair_and_current_right(tmp_path, mode):
    a, b = tmp_path / "A", tmp_path / "B"
    a.mkdir()
    b.mkdir()
    policy = {
        "mode": mode,
        "left": {"kind": "checkpoint", "ref": str(a)},
        "right": {"kind": "checkpoint", "ref": str(b)},
        "right_weight": 0.3,
    }
    fixed = plan(tmp_path, RolloutPolicyConfig.parse(policy))
    assert fixed.binding("left").path == str(a)
    assert fixed.binding("right").path == str(b)
    assert not any(fixed.has_current(r.source) for r in fixed.requests)
    policy["right"] = {"kind": "current"}
    dynamic = plan(tmp_path, RolloutPolicyConfig.parse(policy))
    assert dynamic.binding("right").kind == "current"
    if mode == "token_mixture":
        first, second, weight = dynamic.fusion_components
        assert first.kind == "current" and second.path == str(a)
        assert weight == pytest.approx(0.7)
    else:
        assert sum(r.source == "right" for r in dynamic.requests) == 19
        assert all(dynamic.has_current(r.source) == (r.source == "right") for r in dynamic.requests)


def test_same_fixed_binding_folds_but_current_initial_do_not(tmp_path):
    policy = {"mode": "token_mixture", "left": {"kind": "initial"}, "right": {"kind": "initial"}}
    parsed = RolloutPolicyConfig.parse(policy)
    assert not parsed.requires_fusion
    same = plan(tmp_path, parsed)
    assert same.normalization == "same_binding" and same.resolved_mode == "single"
    policy["left"] = {"kind": "current"}
    distinct = plan(tmp_path, RolloutPolicyConfig.parse(policy))
    assert distinct.resolved_mode == "token_mixture"


def test_odd_quota_never_recomputed_after_current_slot_mapping(tmp_path):
    policy = RolloutPolicyConfig.parse(
        {"mode": "trajectory_mixture", "left": {"kind": "teacher"}, "right": {"kind": "current"}}
    )
    result = build_plan(
        policy,
        uids=["u"],
        stable_ids=[0],
        n=3,
        collection_step=1,
        optimizer_step=0,
        seed=42,
        student_path=str(tmp_path),
        teacher_path=str(tmp_path),
    )
    assert [sum(r.source == side for r in result.requests) for side in ("left", "right")] == [1, 2]


@pytest.mark.parametrize("field", ["source", "student", "teacher_weight", "teacher_model_path", "model_path"])
def test_retired_fields_fail_even_when_new_fields_present(field):
    with pytest.raises(ValueError, match="Retired rollout fields"):
        RolloutPolicyConfig.parse({field: None, "model": {"kind": "current"}})
