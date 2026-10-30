# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Preserve MirrorOPD filter boundaries while using native per-request RNG."""

import torch
from torch import nn


def filter_logits(logits, k=None, p=None):
    """Top-k threshold followed by descending, shifted top-p masking."""
    if k is not None:
        ordered = logits.sort(dim=-1, descending=True).values
        indices = (k.to(torch.long) - 1).clamp(0, logits.shape[-1] - 1)
        threshold = ordered.gather(-1, indices[:, None])
        logits = logits.masked_fill(logits < threshold, -torch.inf)
    if p is not None:
        ordered, indices = logits.sort(dim=-1, descending=True)
        remove = ordered.softmax(-1).cumsum(-1) > p[:, None]
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        logits = logits.masked_fill(torch.zeros_like(remove).scatter_(-1, indices, remove), -torch.inf)
    return logits


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

    This callable runs through vLLM collective_rpc on each engine worker. It
    changes only that worker's sampler; ordinary single and validation retain
    their original instance after the policy phase releases ownership.
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
