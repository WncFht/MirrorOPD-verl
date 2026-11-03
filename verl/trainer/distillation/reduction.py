# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Packed row reduction and detached trajectory credit, independent of kernels."""

import math
from dataclasses import dataclass

import torch

from verl.workers.config.distillation_spec import FutureCreditConfig


def nonempty_sequence_count(loss_mask):
    """Count nonempty answers, including jagged masks and empty DP partitions."""
    masks = loss_mask.unbind()
    if not masks:
        return torch.zeros((), dtype=torch.long, device=loss_mask.device)
    return torch.stack([mask.sum() > 0 for mask in masks]).sum()


def reduction_scale(mode, *, dp_size=1, num_tokens=None, num_sequences=None, fixed_scale=None):
    """Final scalar after row weighting; DP gradients are averaged by the engine."""
    if mode == "token-sum":
        return dp_size
    if mode == "token-mean":
        denominator = num_tokens
    elif mode in ("seq-mean-token-mean", "seq-mean-token-sum", "seq-mean-token-sum-norm"):
        denominator = num_sequences
    else:
        raise ValueError(f"Invalid full-vocab loss_agg_mode: {mode}")
    if denominator is None or denominator <= 0:
        raise ValueError("Full-vocab reduction requires a positive global denominator")
    scale = dp_size / denominator
    if mode == "seq-mean-token-sum-norm":
        if fixed_scale is None or not math.isfinite(fixed_scale) or fixed_scale <= 0:
            raise ValueError("Full-vocab seq-mean-token-sum-norm requires an explicit positive loss_scale_factor")
        scale /= fixed_scale
    return scale


@torch.no_grad()
def suffix_credit(cost, times, config, *, carry=None):
    """Strict discounted suffix on one contiguous response block, in O(log N) launches.

    Doubling scan avoids inverse discount powers (which overflow on long answers)
    and Python per-token loops. Gaps consume original response time. Carry holds
    the inclusive cost/mass and coordinate of the next block's first row.
    """
    if not len(cost):
        return cost.detach().clone(), carry
    gamma = config.gamma
    sums, mass = cost.detach().clone(), torch.ones_like(cost)
    stride = 1
    while stride < len(cost):
        discount = gamma ** (times[stride:] - times[:-stride]).to(cost.dtype)
        sums = torch.cat((sums[:-stride] + discount * sums[stride:], sums[-stride:]))
        mass = torch.cat((mass[:-stride] + discount * mass[stride:], mass[-stride:]))
        stride *= 2
    if carry is not None:
        next_sum, next_mass, next_time = carry
        discount = gamma ** (next_time - times).to(cost.dtype)
        sums = sums + discount * next_sum
        mass = mass + discount * next_mass
    # Shift inclusive suffixes instead of subtracting cost from a large total.
    future, weight = torch.zeros_like(cost), torch.zeros_like(cost)
    if len(cost) > 1:
        discount = gamma ** (times[1:] - times[:-1]).to(cost.dtype)
        future[:-1], weight[:-1] = discount * sums[1:], discount * mass[1:]
    if carry is not None:
        discount = gamma ** (carry[2] - times[-1]).to(cost.dtype)
        future[-1], weight[-1] = discount * carry[0], discount * carry[1]
    if config.mode in ("mean", "discounted_mean"):
        future = torch.where(weight > 0, future / weight.clamp_min(torch.finfo(cost.dtype).tiny), 0)
    if config.mode == "none":
        future.zero_()
    return future, (sums[0], mass[0], times[0])


@dataclass(frozen=True)
class PackedReduction:
    """Immutable response boundaries, actions, and raw response coordinates."""

    counts: tuple[int, ...]
    times: torch.Tensor
    actions: torch.Tensor
    mode: str = "token-mean"
    future: FutureCreditConfig | None = None

    def __post_init__(self):
        if any(type(n) is not int or n < 0 for n in self.counts):
            raise ValueError("Packed response counts must be nonnegative integers")
        if sum(self.counts) != len(self.times) or self.times.shape != self.actions.shape or self.times.ndim != 1:
            raise ValueError("Packed counts, times and actions must describe the same rows")
        if self.times.dtype != torch.long or self.actions.dtype != torch.long:
            raise ValueError("Packed times and actions must be int64")
        reduction_scale(self.mode, num_tokens=1, num_sequences=1, fixed_scale=1)
        torch._assert_async((self.times >= 0).all(), "Response times must be nonnegative")
        offset = 0
        for count in self.counts:
            end = offset + count
            if count > 1:
                torch._assert_async(
                    (self.times[offset + 1 : end] > self.times[offset : end - 1]).all(),
                    "Response times must strictly increase within each answer",
                )
            offset = end

    @property
    def future_config(self):
        return self.future if self.future is not None else FutureCreditConfig()

    def row_weights(self, dtype):
        """Only sequence token means need length-dependent local row weights."""
        if self.mode != "seq-mean-token-mean":
            return None
        return (
            torch.cat([torch.full((n,), 1 / n, device=self.times.device, dtype=dtype) for n in self.counts if n])
            if len(self.times)
            else torch.empty(0, device=self.times.device, dtype=dtype)
        )

    def blocks(self, chunk):
        """Visit each response backwards; a carry never crosses its boundary."""
        start = 0
        for count in self.counts:
            end = start + count
            stop = end
            while stop > start:
                begin = max(start, stop - chunk)
                yield begin, stop, stop == end
                stop = begin
            start = end
