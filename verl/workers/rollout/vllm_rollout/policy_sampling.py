# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Preserve MirrorOPD filter boundaries while using native per-request RNG."""

import torch
from torch import nn

from verl.experimental.rollout_policy.sampling import filter_logits


class FusionTopKTopPSampler(nn.Module):
    """Instance-local replacement of the filter stage, with processed logprobs."""

    def __init__(self, use_fp64_gumbel=False):
        super().__init__()
        self.use_fp64_gumbel = use_fp64_gumbel

    def forward(self, logits, generators, k, p):
        from vllm.v1.sample.ops.topk_topp_sampler import random_sample

        logits = filter_logits(logits, k, p)
        log_probs = logits.log_softmax(-1, dtype=torch.float32)
        return random_sample(log_probs.exp(), generators, self.use_fp64_gumbel), log_probs


def configure_policy_sampler(worker, enabled):
    """Use the policy filter for a drained single-model phase, then restore it.

    The named worker RPC changes only this engine worker. A borrowed policy
    phase restores the previous instance on release; the supported default
    single service keeps the common sampler installed for its run.
    """
    sampler = worker.model_runner.sampler
    attribute = "_rollout_policy_original_sampler"
    if enabled:
        if not hasattr(worker, attribute):
            setattr(worker, attribute, (sampler.topk_topp_sampler, sampler.logprobs_mode))
        sampler.topk_topp_sampler = FusionTopKTopPSampler(
            getattr(getattr(worker, attribute)[0], "use_fp64_gumbel", False)
        )
        sampler.logprobs_mode = "processed_logprobs"
    elif hasattr(worker, attribute):
        sampler.topk_topp_sampler, sampler.logprobs_mode = getattr(worker, attribute)
        delattr(worker, attribute)
    return {"policy_sampling": enabled}
