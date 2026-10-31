# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Read-only native-support RKL, sharing projection and stable arithmetic with training."""

import torch

from verl.trainer.distillation.numerics import kl_values, support_correction
from verl.trainer.distillation.profiling import full_kl_range
from verl.utils.linear import project


@torch.no_grad()
def score_rkl_rows(hs, ws, ht, wt, *, token_chunk):
    if hs.ndim != 2 or ht.ndim != 2 or ws.ndim != 2 or wt.ndim != 2:
        raise ValueError("Expected hidden/head matrices")
    if len(hs) != len(ht) or hs.shape[1] != ws.shape[1] or ht.shape[1] != wt.shape[1]:
        raise ValueError("Response-aligned hidden/head dimensions differ")
    if len(ws) > len(wt) or token_chunk < 1:
        raise ValueError("Teacher must cover student support and token_chunk must be positive")
    values = hs.new_empty(len(hs), dtype=torch.float64 if hs.dtype == torch.float64 else torch.float32)
    for start in range(0, len(hs), token_chunk):
        stop = start + token_chunk
        with full_kl_range("student_projection"):
            zp = project(hs[start:stop], ws)
        with full_kl_range("teacher_projection"):
            zt = project(ht[start:stop], wt)
        with full_kl_range("rkl_readout_reduce"):
            if zp.is_cuda and zp.dtype == zt.dtype == torch.float32:
                from verl.trainer.distillation.full_vocab_triton import difference, weighted_rows

                value, _, _ = weighted_rows(zp, difference(zp, zt), torch.float32, gradient=False)
                values[start:stop] = value + support_correction(zt, zp.shape[-1])
            else:
                values[start:stop] = kl_values(zp, zt)
    return values


def sample_sums(values, counts):
    """Keep a length-one row dimension for the engine's jagged output collector."""
    if sum(counts) != len(values):
        raise ValueError("Per-sample position counts do not cover the score rows")
    if not counts:
        return {
            "rkl_sum": values.new_empty((0, 1), dtype=torch.float64),
            "rkl_count": values.new_empty((0, 1), dtype=torch.int64),
        }
    parts = values.split(counts)
    return {
        "rkl_sum": torch.stack([p.sum(dtype=torch.float64) for p in parts])[:, None],
        "rkl_count": torch.tensor(counts, device=values.device, dtype=torch.int64)[:, None],
    }
