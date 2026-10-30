# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Model identity and resume follow loaded inputs, not incidental run settings."""

from dataclasses import replace

import pytest
from omegaconf import OmegaConf

from verl.experimental.rollout_policy.identity import (
    bind_policy,
    checkpoint_identity,
    save_bindings,
    verify_resume_bindings,
)
from verl.experimental.rollout_policy.models import requires_token_generator, resolve_bindings
from verl.experimental.rollout_policy.plan import CoefficientRule, ModelBinding, RolloutPolicyConfig, build_plan


def setup_model(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text("{}")
    (root / "model.safetensors").write_bytes(b"synthetic manifest fixture")
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {"model": {"path": str(root)}, "rollout": {"top_p": 0.9}},
            "distillation": {"teacher_models": {}},
        }
    )
    return root, config


def test_manifest_ignores_notes_but_tracks_tokenizer_inputs(tmp_path):
    root, _ = setup_model(tmp_path)
    identity = checkpoint_identity(root)
    (root / "notes.txt").write_text("unrelated documentation")
    (root / "optimizer.bin").write_bytes(b"optimizer")
    assert checkpoint_identity(root) == identity
    (root / "merges.txt").write_text("tokenizer input")
    assert checkpoint_identity(root) != identity


def test_resume_allows_execution_change_but_rejects_sampling_change(tmp_path):
    _, config = setup_model(tmp_path)
    policy = RolloutPolicyConfig.parse({"model": {"kind": "initial"}})
    _, state = bind_policy(config, policy)
    save_bindings(tmp_path, state)
    _, changed = bind_policy(config, replace(policy, deadline_seconds=17))
    verify_resume_bindings(tmp_path, changed)
    config.actor_rollout_ref.rollout.top_p = 0.8
    _, changed = bind_policy(config, policy)
    with pytest.raises(ValueError, match="identity changed"):
        verify_resume_bindings(tmp_path, changed)


def test_plan_is_independent_of_filesystem_and_keeps_global_route_ordinals():
    bindings = {
        "left": ModelBinding("current", "/unmounted/a", "a"),
        "right": ModelBinding("checkpoint", "/unmounted/b", "b"),
    }
    policy = RolloutPolicyConfig.parse({"mode": "trajectory_mixture"})
    plan = build_plan(
        policy, bindings=bindings, uids=["a", "b"], stable_ids=[0, 1], n=3, collection_step=1, optimizer_step=0, seed=42
    )
    assert [r.ordinal for r in plan.requests] == list(range(6))
    assert len({r.ordinal for phase in plan.phases for r in phase.requests}) == 6


def test_schedule_and_content_identity_determine_reachable_generator():
    policy = RolloutPolicyConfig.parse({"mode": "token_mixture"})
    same = {"left": ModelBinding("checkpoint", "/a", "same"), "right": ModelBinding("checkpoint", "/b", "same")}
    assert not requires_token_generator(policy, same)
    same["left"] = ModelBinding("current", "/a", "same")
    assert requires_token_generator(policy, same)
    assert not requires_token_generator(replace(policy, right_weight=CoefficientRule(start=0, end=0)), same)
    assert requires_token_generator(replace(policy, right_weight=CoefficientRule(kind="linear", start=0, end=1)), same)


def test_application_resolver_runs_once_per_location():
    calls = []
    policy = RolloutPolicyConfig.parse(
        {"mode": "token_mixture", "left": {"kind": "initial"}, "right": {"kind": "initial"}}
    )

    def resolve(value):
        calls.append(value)
        return "/resolved/model"

    bindings = resolve_bindings(policy, "alias", resolve_path=resolve)
    assert calls == ["alias"]
    assert bindings["left"].same_model(bindings["right"])
