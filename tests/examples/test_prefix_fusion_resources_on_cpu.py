# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Single-card evaluation preserves its shape when the physical allocation expands."""

import pytest

from examples.mirroropd.research.studies.prefix_fusion.scoring import require_allocated_gpu


@pytest.mark.parametrize("device", [str(i) for i in range(8)])
def test_scoring_accepts_each_allocated_single_card(monkeypatch, device):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", device)
    require_allocated_gpu()


@pytest.mark.parametrize("devices", ["", "-1", "8", "0,1", "6,7", "0,0"])
def test_scoring_rejects_wrong_shape_or_unknown_card(monkeypatch, devices):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", devices)
    with pytest.raises(ValueError):
        require_allocated_gpu()


def test_verified_checkpoint_relocation_does_not_change_plan_random_streams(tmp_path):
    from verl.experimental.rollout_policy.plan import RolloutPolicyConfig, build_plan

    old, new = tmp_path / "old", tmp_path / "new"
    for root in [old, new]:
        (root / "student").mkdir(parents=True)
        (root / "teacher").mkdir()
    kwargs = dict(uids=["a", "b"], stable_ids=["q1", "q2"], n=1, collection_step=4, optimizer_step=48, seed=42)
    policy = RolloutPolicyConfig.parse(dict(mode="token_mixture", right_weight=0.5))
    left = build_plan(policy, student_path=str(old / "student"), teacher_path=str(old / "teacher"), **kwargs)
    right = build_plan(policy, student_path=str(new / "student"), teacher_path=str(new / "teacher"), **kwargs)
    assert left.requests == right.requests
    assert left.actor_version == right.actor_version
