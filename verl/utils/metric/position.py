# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Relative-position statistics for any scalar token observation.

Callers select their valid tokens and provide original sequence coordinates.
This module knows neither model layouts nor the meaning of the observed values.
"""

import torch

from .observations import Observation

POSITION_BINS = tuple(f"p{start:02d}_{start + 10:02d}" for start in range(0, 100, 10))


@torch.no_grad()
def relative_position_bins(positions, lengths):
    """Map zero-based token midpoints to deciles of their original sequences.

    Both inputs are aligned 1-D integer tensors, one coordinate/length per
    selected token. Masked gaps retain their original coordinates.
    """
    if positions.ndim != 1 or positions.shape != lengths.shape:
        raise ValueError("Token positions and lengths must be aligned 1-D tensors")
    if positions.dtype not in (torch.int32, torch.int64) or lengths.dtype not in (torch.int32, torch.int64):
        raise ValueError("Token positions and lengths must be integers")
    if positions.device != lengths.device:
        raise ValueError("Token positions and lengths must use the same device")
    positions, lengths = positions.long(), lengths.long()
    if positions.numel() and torch.any((positions < 0) | (positions >= lengths)):
        raise ValueError("Token positions must lie inside their sequences")
    return torch.div((2 * positions + 1) * len(POSITION_BINS), 2 * lengths, rounding_mode="floor")


@torch.no_grad()
def position_statistics(values, bins):
    """Detached sum/count for selected tokens; preserve signed/nonfinite values.

    Bins can be reused across several metrics on the same positions. Scalar
    token indicators are also supported: their mean is a fraction.
    """
    if values.ndim != 1 or values.shape != bins.shape or values.device != bins.device:
        raise ValueError("Token values and bins must be aligned 1-D tensors on the same device")
    if bins.dtype != torch.int64:
        raise ValueError("Position bins must be int64")
    if bins.numel() and torch.any((bins < 0) | (bins >= len(POSITION_BINS))):
        raise ValueError("Position bin index is out of range")
    sums = torch.zeros(len(POSITION_BINS), dtype=torch.float64, device=values.device)
    sums.scatter_add_(0, bins, values.detach().double())
    return sums, torch.bincount(bins, minlength=len(POSITION_BINS))


def position_observations(sums, counts, *, prefix) -> dict[str, Observation]:
    """Expose mergeable statistics under a namespace chosen by the caller."""
    if sums.shape != counts.shape or sums.shape != (len(POSITION_BINS),):
        raise ValueError("Position statistics must contain one sum and count per bin")
    metrics = {}
    for label, total, count in zip(POSITION_BINS, sums, counts, strict=True):
        stem = f"{prefix.rstrip('/')}/{label}/"
        metrics[stem + "sum"] = Observation.total(total)
        metrics[stem + "token_count"] = Observation.total(count)
        metrics[stem + "mean"] = Observation.ratio(total, count)
    return metrics
