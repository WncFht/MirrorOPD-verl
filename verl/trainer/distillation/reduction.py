# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Packed row reduction and detached trajectory credit, independent of kernels."""

import math
from dataclasses import dataclass, field

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
def suffix_credit(cost, times, segments, config, *, carry=None):
    """Strict segmented suffix with original time gaps and one edge-answer carry.

    Doubling never combines different answers. Nonnegative exponents avoid
    inverse discount overflow even across time resets. Shifting inclusive
    suffixes preserves small future signals next to large local costs.
    """
    if not len(cost):
        return cost.detach().clone(), carry
    gamma = config.gamma
    mean = config.mode in ("mean", "discounted_mean")
    sums = cost.detach().clone()
    mass = torch.ones_like(cost) if mean else None

    def weights(left_times, right_times, same):
        if gamma == 1:
            return same.to(cost.dtype)
        return same * gamma ** (right_times - left_times).clamp_min(0).to(cost.dtype)

    stride = 1
    while stride < len(cost):
        weight = weights(times[:-stride], times[stride:], segments[:-stride] == segments[stride:])
        sums = torch.cat((sums[:-stride] + weight * sums[stride:], sums[-stride:]))
        if mean:
            mass = torch.cat((mass[:-stride] + weight * mass[stride:], mass[-stride:]))
        stride *= 2
    if carry is not None:
        next_sum, next_mass, next_time, next_segment = carry
        weight = weights(times, next_time, segments == next_segment)
        sums = sums + weight * next_sum
        if mean:
            mass = mass + weight * next_mass
    future = torch.zeros_like(cost)
    future_mass = torch.zeros_like(cost) if mean else None
    if len(cost) > 1:
        weight = weights(times[:-1], times[1:], segments[:-1] == segments[1:])
        future[:-1] = weight * sums[1:]
        if mean:
            future_mass[:-1] = weight * mass[1:]
    if carry is not None:
        weight = weights(times[-1], carry[2], segments[-1] == carry[3])
        future[-1] = weight * carry[0]
        if mean:
            future_mass[-1] = weight * carry[1]
    if mean:
        future = torch.where(future_mass > 0, future / future_mass.clamp_min(torch.finfo(cost.dtype).tiny), 0)
    if config.mode == "none":
        future.zero_()
    return future, (sums[0], mass[0] if mean else None, times[0], segments[0])


@dataclass(frozen=True)
class PackedReduction:
    """Immutable response boundaries, actions, and raw response coordinates."""

    counts: tuple[int, ...]
    times: torch.Tensor
    actions: torch.Tensor
    mode: str = "token-mean"
    future: FutureCreditConfig | None = None
    segments: torch.Tensor = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if any(type(n) is not int or n < 0 for n in self.counts):
            raise ValueError("Packed response counts must be nonnegative integers")
        if sum(self.counts) != len(self.times) or self.times.shape != self.actions.shape or self.times.ndim != 1:
            raise ValueError("Packed counts, times and actions must describe the same rows")
        if self.times.dtype != torch.long or self.actions.dtype != torch.long:
            raise ValueError("Packed times and actions must be int64")
        object.__setattr__(
            self,
            "segments",
            torch.repeat_interleave(
                torch.arange(len(self.counts), device=self.times.device),
                torch.tensor(self.counts, device=self.times.device, dtype=torch.long),
                output_size=len(self.times),
            ),
        )
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
        """Continuous reverse chunks; answer boundaries constrain scans, not GEMMs."""
        for stop in range(len(self.times), 0, -chunk):
            yield max(0, stop - chunk), stop
