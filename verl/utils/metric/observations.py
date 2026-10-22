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
        return cls("max", torch.stack((maximum, values.new_tensor(values.numel()))))

    @classmethod
    def minimum(cls, values):
        values = values.detach().double().reshape(-1)
        minimum = values.min() if values.numel() else values.new_tensor(math.inf)
        return cls("min", torch.stack((minimum, values.new_tensor(values.numel()))))

    @classmethod
    def total(cls, values):
        return cls("sum", torch.as_tensor(values).detach().double().sum().reshape(1))

    @classmethod
    def std(cls, values):
        values = values.detach().double().reshape(-1)
        mean = values.mean() if values.numel() else values.new_tensor(0.0)
        return cls("std", torch.stack((values.new_tensor(values.numel()), mean, ((values - mean) ** 2).sum())))

    @classmethod
    def log_moment(cls, log_weights, kind):
        if kind not in ("log_mean", "log_std"):
            raise ValueError(kind)
        result = cls.ess(log_weights)
        result.kind = kind
        return result

    @classmethod
    def quantile(cls, values, q):
        if not 0 < q <= 1:
            raise ValueError("Quantile must be in (0, 1]")
        values = values.detach().reshape(-1)
        return cls(f"quantile:{q}", values.new_empty(0), [values])

    def is_empty(self):
        if self.kind in ("ratio", "rms", "max", "min"):
            return self.state[1].item() == 0
        if self.kind in ("ess", "ess_absolute", "log_mean", "log_std"):
            return self.state[2].item() == 0 or (self.kind.startswith("ess") and self.state[0].item() == -math.inf)
        if self.kind == "std":
            return self.state[0].item() == 0
        if self.kind == "ratio_p99" or self.kind.startswith("quantile:"):
            return not any(x.numel() for x in self.samples)
        return False

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
            state[0] = -math.inf
        elif self.kind == "min":
            state[0] = math.inf
        elif self.kind in ("ess", "ess_absolute", "log_mean", "log_std"):
            state[:2] = -math.inf
        return Observation(self.kind, state)

    def append(self, other):
        """Merge another partition without converting it into a local mean."""
        if not isinstance(other, Observation) or self.kind != other.kind:
            raise ValueError("Observation reduction types must match")
        if self.kind in ("max", "min"):
            operation = torch.maximum if self.kind == "max" else torch.minimum
            self.state = torch.stack((operation(self.state[0], other.state[0]), self.state[1] + other.state[1]))
        elif self.kind == "std":
            n, mean, m2 = self.state.unbind()
            m, other_mean, other_m2 = other.state.unbind()
            total = n + m
            delta = other_mean - mean
            self.state = torch.stack(
                (
                    total,
                    mean + delta * m / total.clamp_min(1),
                    m2 + other_m2 + delta.square() * n * m / total.clamp_min(1),
                )
            )
        elif self.kind in ("ess", "ess_absolute", "log_mean", "log_std"):
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
        elif self.kind in ("max", "min", "sum"):
            result = state[0]
        elif self.kind == "std":
            result = math.sqrt(max(0.0, state[2] / state[0])) if state[0] > 0 and math.isfinite(state[2]) else math.nan
        elif self.kind in ("log_mean", "log_std"):
            if state[2] > 0 and state[0] == -math.inf and state[1] == -math.inf:
                return 0.0
            if state[2] <= 0 or not all(math.isfinite(x) for x in state):
                return math.nan
            first, second = state[0] - math.log(state[2]), state[1] - math.log(state[2])
            if self.kind == "log_mean":
                result = math.exp(first) if first < 709.78 else math.inf
            else:
                fraction = -math.expm1(min(0.0, 2 * first - second))
                result = math.exp(second / 2) * math.sqrt(fraction) if second / 2 < 709.78 else math.inf
        elif self.kind in ("ess", "ess_absolute"):
            if state[2] <= 0 or not all(math.isfinite(x) for x in state):
                return math.nan
            result = min(1.0, math.exp(2 * state[0] - state[1] - math.log(state[2])))
            if self.kind == "ess_absolute":
                result *= state[2]
        elif self.kind == "ratio_p99" or self.kind.startswith("quantile:"):
            values = torch.cat(self.samples) if self.samples else torch.empty(0)
            if values.numel() == 0 or not torch.isfinite(values).all():
                return math.nan
            q = 0.99 if self.kind == "ratio_p99" else float(self.kind.split(":")[1])
            rank = max(1, math.ceil(q * values.numel()))
            z = values.double().kthvalue(rank).values.item()
            result = z
            if self.kind == "ratio_p99":
                result = math.exp(z) if z <= math.log(torch.finfo(torch.float64).max) else math.inf
        else:
            raise ValueError(f"Unknown observation reduction: {self.kind}")
        if self.kind == "rms" and math.isfinite(result):
            result = math.sqrt(max(0.0, result))
        return result if math.isfinite(result) else math.nan

    @classmethod
    def merged(cls, partitions):
        merged = partitions[0].init_list()
        for partition in partitions:
            merged.append(partition)
        return merged

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
    sample_cache = {}
    for value in observations:
        size = value.state.numel()
        value.state = packed[offset : offset + size]
        offset += size
        if value.samples:
            key = tuple((str(v.device), v.data_ptr(), v.numel()) for v in value.samples)
            if key not in sample_cache:
                sample_cache[key] = torch.cat(value.samples).cpu()
            value.samples = [sample_cache[key]]
