# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Sampler boundaries must survive migration to the native vLLM RNG."""

import pytest
import torch

from verl.experimental.rollout_policy.sampling import SamplingConfig, fusion_log_probs
from verl.workers.rollout.vllm_fusion.sampling import filter_logits


@pytest.mark.parametrize("top_k", [-1, 1, 3, 7])
@pytest.mark.parametrize("top_p", [0.1, 0.5, 0.9, 1.0])
def test_native_filter_preserves_policy(top_k, top_p):
    logits = torch.tensor([[0, 0, -1, -2, -3, -4, -5], [1, 0, 0, 0, -2, -3, -4]], dtype=torch.float32)
    k = torch.full((2,), top_k) if top_k > 0 else None
    p = torch.full((2,), top_p) if top_p < 1 else None
    actual = filter_logits(logits.clone(), k, p).log_softmax(-1)
    expected = fusion_log_probs(logits, None, 0, SamplingConfig(top_k=top_k, top_p=top_p))
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=0)


def test_exact_half_boundary_keeps_crossing_token():
    actual = filter_logits(torch.zeros(1, 2), p=torch.tensor([0.5])).softmax(-1)
    torch.testing.assert_close(actual, torch.tensor([[0.5, 0.5]]), atol=0, rtol=0)


def test_batched_filters_do_not_cross_request_boundaries():
    logits = torch.tensor([[2, 1, 0], [0, 2, 1]], dtype=torch.float32)
    k, p = torch.tensor([1, 3]), torch.tensor([1.0, 0.8])
    actual = filter_logits(logits, k, p)
    for index in range(2):
        expected = filter_logits(logits[index : index + 1], k[index : index + 1], p[index : index + 1])
        torch.testing.assert_close(actual[index : index + 1], expected)


def test_single_phase_preserves_boundary_and_restores_original_sampler():
    from types import SimpleNamespace

    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch

    from verl.workers.rollout.vllm_fusion.sampling import FusionTopKTopPSampler, configure_policy_sampler

    logits = torch.zeros(1, 2)
    top_p = torch.tensor([0.5])
    # The exact half-mass boundary is a real counterexample to substituting the
    # default sampler when a token-mixture plan normalizes to one component.
    native = apply_top_k_top_p_pytorch(logits.clone(), None, top_p)
    assert torch.isfinite(native).sum() == 1
    assert torch.isfinite(filter_logits(logits, p=top_p)).sum() == 2
    original = SimpleNamespace(use_fp64_gumbel=True)
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(sampler=SimpleNamespace(topk_topp_sampler=original, logprobs_mode="raw_logprobs"))
    )
    configure_policy_sampler(worker, True)
    configure_policy_sampler(worker, True)
    assert isinstance(worker.model_runner.sampler.topk_topp_sampler, FusionTopKTopPSampler)
    assert worker.model_runner.sampler.topk_topp_sampler.use_fp64_gumbel
    assert worker.model_runner.sampler.logprobs_mode == "processed_logprobs"
    configure_policy_sampler(worker, False)
    configure_policy_sampler(worker, False)
    assert worker.model_runner.sampler.topk_topp_sampler is original
    assert worker.model_runner.sampler.logprobs_mode == "raw_logprobs"
