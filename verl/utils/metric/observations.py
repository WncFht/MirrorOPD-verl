# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Merge raw observations independently of gradient-loss scaling and time axes."""

import math

import torch

from .utils import Metric


class Observation(Metric):
    """A detached sufficient statistic; partitions merge before finalization.

    Unlike legacy ``Metric('sum')`` loss contributions, these states are never
    DP-averaged. The inherited marker lets engine micro-batch collectors use
    the existing append/init_list protocol without exposing a second pipeline.
    """

    def __init__(self, kind, state, samples=None):
        self.kind = kind
        self.state = state.detach().to(torch.float64)
        self.samples = [] if samples is None else samples

    @classmethod
    def ratio(cls, numerator, denominator):
        """Accumulate a numerator and its actual observation weight."""
        numerator = torch.as_tensor(numerator).detach().double()
        denominator = torch.as_tensor(denominator, device=numerator.device).detach().double()
        return cls("ratio", torch.stack((numerator, denominator)))

    @classmethod
    def mean(cls, values):
        """Observe equally weighted values, including nonfinite observations."""
        values = values.detach().double().reshape(-1)
        return cls.ratio(values.sum(), values.numel())

    @classmethod
    def maximum(cls, values):
        """Observe a maximum; an empty partition contributes negative infinity."""
        values = values.detach().double().reshape(-1)
        maximum = values.max() if values.numel() else values.new_tensor(-math.inf)
        return cls("max", maximum.reshape(1))

    @classmethod
    def ess(cls, log_weights):
        """Observe weight concentration without exponentiating extreme ratios."""
        z = log_weights.detach().double().reshape(-1)
        return cls("ess", torch.stack((z.logsumexp(0), (2 * z).logsumexp(0), z.new_tensor(z.numel()))))

    @classmethod
    def ratio_p99(cls, log_weights):
        """Retain selected update values for exact global nearest-rank p99."""
        z = log_weights.detach().reshape(-1)
        return cls("ratio_p99", z.new_empty(0), [z])

    def init_list(self):
        """Create an empty state for engine micro-batch accumulation."""
        state = torch.zeros_like(self.state)
        if self.kind == "max":
            state.fill_(-math.inf)
        elif self.kind == "ess":
            state[:2] = -math.inf
        return Observation(self.kind, state)

    def append(self, other):
        """Merge another partition without converting it into a local mean."""
        if not isinstance(other, Observation) or self.kind != other.kind:
            raise ValueError("Observation reduction types must match")
        if self.kind == "max":
            self.state = torch.maximum(self.state, other.state)
        elif self.kind == "ess":
            self.state = torch.cat((torch.logaddexp(self.state[:2], other.state[:2]), self.state[2:] + other.state[2:]))
        else:
            self.state = self.state + other.state
        self.samples.extend(other.samples)

    extend = append

    def aggregate(self):
        """Finalize once after spatial merging; undefined results are NaN."""
        state = self.state.cpu().tolist()
        if self.kind in ("ratio", "rms"):
            result = state[0] / state[1] if state[1] > 0 else math.nan
        elif self.kind == "max":
            result = state[0]
        elif self.kind == "ess":
            if state[2] <= 0 or not all(math.isfinite(x) for x in state):
                return math.nan
            result = min(1.0, math.exp(2 * state[0] - state[1] - math.log(state[2])))
        elif self.kind == "ratio_p99":
            values = torch.cat(self.samples) if self.samples else torch.empty(0)
            if values.numel() == 0 or not torch.isfinite(values).all():
                return math.nan
            rank = max(1, math.ceil(0.99 * values.numel()))
            z = values.double().kthvalue(rank).values.item()
            result = math.exp(z) if z <= math.log(torch.finfo(torch.float64).max) else math.inf
        else:
            raise ValueError(f"Unknown observation reduction: {self.kind}")
        if self.kind == "rms" and math.isfinite(result):
            result = math.sqrt(max(0.0, result))
        return result if math.isfinite(result) else math.nan

    @classmethod
    def merge_partitions(cls, partitions):
        """Finalize the union of arbitrary, possibly empty rank partitions."""
        if not partitions:
            return math.nan
        merged = partitions[0].init_list()
        for partition in partitions:
            merged.append(partition)
        return merged.aggregate()


def observations_to_cpu(metrics):
    """Move all scalar states together before the existing object collective."""
    observations = [value for value in metrics.values() if isinstance(value, Observation)]
    if not observations:
        return
    packed = torch.cat([value.state for value in observations]).cpu()
    offset = 0
    for value in observations:
        size = value.state.numel()
        value.state = packed[offset : offset + size]
        offset += size
        if value.samples:
            value.samples = [torch.cat(value.samples).cpu()]
