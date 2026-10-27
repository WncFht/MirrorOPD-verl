# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Counterfactuals must preserve saved state and execute actual zero gradients."""

import copy
import importlib

import pytest
import torch


@pytest.fixture
def probe(monkeypatch):
    return importlib.import_module("examples.mirroropd.research.precision.audit_optimizer_history")


def warm(weight_decay=0.0):
    p = torch.nn.Parameter(torch.tensor([0.5, -0.3, 1.0]))
    opt = torch.optim.AdamW([p], lr=0.02, betas=(0.9, 0.99), weight_decay=weight_decay, foreach=False)
    for _ in range(3):
        p.grad = torch.tensor([0.2, -0.4, 0.7])
        opt.step()
    theta = {"w": p.detach().clone()}
    state = {"w": {k: v.clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v) for k, v in opt.state[p].items()}}
    groups = [{**{k: copy.deepcopy(v) for k, v in opt.param_groups[0].items() if k != "params"}, "names": ["w"]}]
    return p, opt, theta, state, groups


def test_replay_matches_native_and_preserves_inputs(probe):
    p, opt, theta, state, groups = warm(0.01)
    g = {"w": torch.tensor([0.1, 0.5, -0.2])}
    original_theta, original_state = copy.deepcopy(theta), copy.deepcopy(state)
    p.grad = g["w"].clone()
    opt.step()
    actual = probe.replay_adam_update(theta, g, state, groups, "saved", device="cpu")
    torch.testing.assert_close(actual["w"], p.detach() - theta["w"], rtol=0, atol=0)
    torch.testing.assert_close(theta["w"], original_theta["w"], rtol=0, atol=0)
    for k in state["w"]:
        torch.testing.assert_close(state["w"][k], original_state["w"][k], rtol=0, atol=0)


def test_zero_gradient_keeps_history_motion_but_decay_only_does_not(probe):
    p, opt, theta, state, groups = warm()
    g = {"w": torch.tensor([0.1, 0.5, -0.2])}
    p.grad = torch.zeros_like(p)
    opt.step()
    zero = probe.replay_adam_update(theta, g, state, groups, "zero", device="cpu")["w"]
    torch.testing.assert_close(zero, p.detach() - theta["w"], rtol=0, atol=0)
    assert zero.norm() > 0
    decay = probe.replay_adam_update(theta, g, state, groups, "decay_only", device="cpu")["w"]
    torch.testing.assert_close(decay, torch.zeros_like(decay), rtol=0, atol=0)


def test_decay_control_retains_weight_decay(probe):
    _, _, theta, state, groups = warm(0.1)
    g = {"w": torch.ones_like(theta["w"])}
    actual = probe.replay_adam_update(theta, g, state, groups, "decay_only", device="cpu")["w"]
    expected = theta["w"] * (1 - groups[0]["lr"] * groups[0]["weight_decay"]) - theta["w"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_interventions_do_not_contaminate_repeated_saved_replay(probe):
    _, _, theta, state, groups = warm()
    g = {"w": torch.tensor([0.1, 0.5, -0.2])}
    before = probe.replay_adam_update(theta, g, state, groups, "saved", device="cpu")["w"]
    probe.replay_adam_update(theta, g, state, groups, "reset_m", device="cpu")
    probe.replay_adam_update(theta, g, state, groups, "fresh", device="cpu")
    after = probe.replay_adam_update(theta, g, state, groups, "saved", device="cpu")["w"]
    torch.testing.assert_close(before, after, rtol=0, atol=0)
