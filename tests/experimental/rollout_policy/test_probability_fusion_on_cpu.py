# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Arithmetic token mixtures must retain their coefficient on legal support."""

from dataclasses import asdict, replace

import pytest
import torch

from verl.experimental.rollout_policy.plan import RolloutPolicyConfig, build_plan
from verl.experimental.rollout_policy.sampling import (
    SamplingConfig,
    fusion_log_probs,
    sample_tokens,
    stop_event_probabilities,
)


@pytest.mark.parametrize("alpha", [0, 0.2, 0.5, 1])
@pytest.mark.parametrize("temperature", [0.7, 1, 2])
def test_probability_fusion_against_independent_fp64_oracle(alpha, temperature):
    student = torch.tensor([[2.0, 1000.0, -1.0, -4.0]], dtype=torch.float64)
    teacher = torch.tensor([[-1.0, -1000.0, 2.0, -3.0]], dtype=torch.float64)
    legal = [0, 2, 3]
    # Independent arithmetic oracle: normalize each model's legal entries,
    # add probabilities, then apply the common temperature and normalize.
    p = student[:, legal].softmax(-1)
    q = teacher[:, legal].softmax(-1)
    mixed = ((1 - alpha) * p + alpha * q).pow(1 / temperature)
    expected = (mixed / mixed.sum(-1, keepdim=True)).log()
    config = SamplingConfig(temperature=temperature, legal_ids=tuple(legal), fusion_space="probability")
    actual = fusion_log_probs(student, teacher, alpha, config)
    torch.testing.assert_close(actual[:, legal].double(), expected, atol=2e-6, rtol=0)
    assert torch.isneginf(actual[:, 1]).all()


def test_arithmetic_and_geometric_mixtures_are_distinct():
    student = torch.tensor([[0.9, 0.1]], dtype=torch.float64).log()
    teacher = torch.tensor([[0.2, 0.8]], dtype=torch.float64).log()
    arithmetic = fusion_log_probs(student, teacher, 0.5, SamplingConfig(fusion_space="probability"))
    geometric = fusion_log_probs(student, teacher, 0.5, SamplingConfig())
    torch.testing.assert_close(arithmetic.exp(), torch.tensor([[0.55, 0.45]]), atol=2e-6, rtol=0)
    torch.testing.assert_close(geometric.exp(), torch.tensor([[0.6, 0.4]]), atol=2e-6, rtol=0)


@pytest.mark.parametrize("alpha", [0, 0.5, 1])
def test_full_native_vocabulary_preserves_small_tail_mass(alpha):
    width = 151936
    student = torch.full((1, width), 0.44 / (width - 1)).log()
    teacher = torch.full((1, width), 0.6 / (width - 1)).log()
    student[0, 0] = torch.tensor(0.56).log()
    teacher[0, 1] = torch.tensor(0.4).log()
    expected = ((1 - alpha) * student.double().softmax(-1) + alpha * teacher.double().softmax(-1)).log()
    actual = fusion_log_probs(student, teacher, alpha, SamplingConfig(fusion_space="probability"))
    torch.testing.assert_close(actual.double().exp(), expected.exp(), atol=2e-6, rtol=0)
    torch.testing.assert_close(actual[:, :2].double(), expected[:, :2], atol=2e-6, rtol=0)


def test_disjoint_support_endpoints_and_selected_behavior_probability():
    student = torch.tensor([[0.0, -torch.inf, -torch.inf]])
    teacher = torch.tensor([[-torch.inf, -torch.inf, 0.0]])
    config = SamplingConfig(fusion_space="probability")
    actual = fusion_log_probs(student, teacher, 0.5, config)
    torch.testing.assert_close(actual.exp(), torch.tensor([[0.5, 0.0, 0.5]]), atol=2e-6, rtol=0)
    token, logprob = sample_tokens(actual, [torch.Generator().manual_seed(42)])
    assert logprob.item() == actual[0, token.item()].item()
    bad = torch.full_like(student, torch.nan)
    torch.testing.assert_close(fusion_log_probs(student, bad, 0, config), student)
    torch.testing.assert_close(fusion_log_probs(bad, teacher, 1, config), teacher)
    with pytest.raises(ValueError, match="support"):
        fusion_log_probs(student, teacher, 0.5, SamplingConfig())


def test_common_filtering_greedy_and_stop_mass():
    student = torch.tensor([[0.6, 0.3, 0.1]]).log()
    teacher = torch.tensor([[0.2, 0.3, 0.5]]).log()
    # Arithmetic mixture is (0.4, 0.3, 0.3): top-p=0.35 retains only token 0.
    config = SamplingConfig(fusion_space="probability", top_p=0.35)
    actual = fusion_log_probs(student, teacher, 0.5, config)
    assert actual[0, 0] == 0 and torch.isneginf(actual[0, 1:]).all()
    greedy = fusion_log_probs(student, teacher, 0.5, replace(config, temperature=0, top_p=1))
    torch.testing.assert_close(actual, greedy)
    diagnostic = stop_event_probabilities(
        {"student": student, "teacher": teacher}, 0.5, (0, 2), (0, 2), fusion_space="probability"
    )
    assert diagnostic["fusion/total_stop_mass"].item() == pytest.approx(0.7, abs=2e-6)


def test_fusion_space_is_frozen_in_plan_without_changing_request_streams(tmp_path):
    kwargs = dict(
        uids=["a", "b"],
        stable_ids=["row-a", "row-b"],
        n=1,
        collection_step=3,
        optimizer_step=32,
        seed=42,
        student_path=str(tmp_path),
        teacher_path=str(tmp_path),
    )
    policy = RolloutPolicyConfig.parse({"mode": "token_mixture", "fusion_space": "probability"})
    plan = build_plan(policy, **kwargs)
    geometric = build_plan(replace(policy, fusion_space="logit"), **kwargs)
    assert asdict(plan)["fusion_space"] == "probability"
    assert plan.requests == geometric.requests
    assert {row.source for row in plan.requests} == {"fusion"}
    with pytest.raises(ValueError, match="token_mixture"):
        RolloutPolicyConfig.parse({"fusion_space": "probability"})
    with pytest.raises(ValueError, match="fusion_space"):
        SamplingConfig(fusion_space="unknown")
