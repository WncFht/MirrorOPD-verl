# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
import json

import pytest
from omegaconf import OmegaConf

from verl.experimental.rollout_policy.identity import (
    bind_policy,
    checkpoint_identity,
    save_bindings,
    verify_resume_bindings,
)
from verl.experimental.rollout_policy.plan import ModelRef, RolloutPolicyConfig


def checkpoint(path, weights):
    path.mkdir()
    (path / "config.json").write_text('{"model_type":"qwen2"}')
    (path / "model.safetensors").write_bytes(weights)
    return str(path)


def test_fixed_content_and_initial_identity_survive_resume(tmp_path):
    original = checkpoint(tmp_path / "original", b"original")
    other = checkpoint(tmp_path / "other", b"other")
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {"model": {"path": original}},
            "distillation": {"teacher_models": {"q": {"model_path": other}}},
        }
    )
    policy = RolloutPolicyConfig.parse({"mode": "token_mixture", "left": {"kind": "initial"}})
    bindings, state = bind_policy(config, policy)
    assert bindings["left"].path == original and bindings["left"].kind == "checkpoint"
    save_bindings(tmp_path, state)
    verify_resume_bindings(tmp_path, state)
    config.actor_rollout_ref.model.path = other
    _, changed = bind_policy(config, policy)
    with pytest.raises(ValueError, match="identity changed"):
        verify_resume_bindings(tmp_path, changed)
    assert json.loads((tmp_path / "rollout_state.json").read_text()) == state


def test_same_path_changed_weights_or_tokenizer_detected(tmp_path):
    path = checkpoint(tmp_path / "model", b"first")
    before = checkpoint_identity(path)
    (tmp_path / "model/model.safetensors").write_bytes(b"other")
    assert checkpoint_identity(path) != before
    changed = checkpoint_identity(path)
    (tmp_path / "model/tokenizer.json").write_text("{}")
    assert checkpoint_identity(path) != changed


def test_missing_identity_cannot_resume(tmp_path):
    with pytest.raises(ValueError, match="original frozen code"):
        verify_resume_bindings(tmp_path, {})


def test_reference_pins_symlink_and_disallows_ambiguous_teacher(tmp_path):
    target = tmp_path / "original"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target)
    bound = ModelRef("checkpoint", str(alias)).resolve(str(tmp_path))
    assert bound.path == str(target)
    with pytest.raises(ValueError, match="exactly one"):
        ModelRef("teacher").resolve(str(tmp_path))


def test_identical_fixed_content_folds_across_distinct_directories(tmp_path):
    from verl.experimental.rollout_policy.plan import build_plan

    a = checkpoint(tmp_path / "A", b"same")
    b = checkpoint(tmp_path / "B", b"same")
    config = OmegaConf.create({"actor_rollout_ref": {"model": {"path": a}}, "distillation": {"teacher_models": {}}})
    policy = RolloutPolicyConfig.parse(
        {"mode": "token_mixture", "left": {"kind": "checkpoint", "ref": a}, "right": {"kind": "checkpoint", "ref": b}}
    )
    bindings, _ = bind_policy(config, policy)
    plan = build_plan(
        policy,
        bindings=bindings,
        uids=["u"],
        stable_ids=[0],
        n=1,
        collection_step=1,
        optimizer_step=0,
        seed=0,
        student_path=a,
    )
    assert plan.normalization == "same_binding"
    assert [phase.decoder for phase in plan.phases] == ["left"]
