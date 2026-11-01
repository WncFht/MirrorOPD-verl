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


def distribution_values(zp, zt, actions=None):
    """Use the value-only fused kernel for packed FP32 logits; no dZ or VJP."""
    if zp.is_cuda and zp.dtype == zt.dtype == torch.float32:
        from verl.trainer.distillation.full_vocab_triton import difference, weighted_rows

        corrections = None
        if actions is not None:
            zp, ep = actions.coarsen(zp, compensated=True)
            zt, et = actions.coarsen(zt, compensated=True)
            corrections = (ep, et)
        zp, zt = zp.contiguous(), zt.contiguous()
        return weighted_rows(zp, difference(zp, zt, corrections=corrections), torch.float32, gradient=False)[0]
    if actions is not None:
        zp, zt = actions.coarsen(zp)[0], actions.coarsen(zt)[0]
    return kl_values(zp, zt)


@torch.no_grad()
def score_distribution_rows(hs, ws, ht, wt, *, token_chunk, distribution=None, termination_ids=()):
    """Value-only RKL and probability mass on a declared, reproducible space."""
    from verl.trainer.distillation.distribution import CompiledDistribution, DistributionSpec
    from verl.trainer.distillation.termination import TerminationActionMap

    space = (
        distribution
        if isinstance(distribution, CompiledDistribution)
        else DistributionSpec.parse(distribution).compile(len(ws), len(wt), termination_ids)
    )
    if space.spec.mode == "native" and not space.termination_ids:
        return {"rkl": score_rkl_rows(hs, ws, ht, wt, token_chunk=token_chunk)}
    if token_chunk < 1 or len(hs) != len(ht):
        raise ValueError("Invalid scoring chunk or mismatched hidden rows")
    width = len(space.student_ids) if space.student_ids else len(ws)
    actions = TerminationActionMap.create(width, space.termination_ids, hs.device) if space.termination_ids else None
    if actions is not None and space.spec.mode == "native" and len(ws) != len(wt):
        raise ValueError("Native termination readout requires equal head widths")
    names = ["rkl", "uncoarsened_rkl", "student_excluded_mass", "teacher_excluded_mass"]
    out = {k: hs.new_empty(len(hs), dtype=torch.float64 if hs.dtype == torch.float64 else torch.float32) for k in names}
    for start in range(0, len(hs), token_chunk):
        logits = [project(h[start : start + token_chunk], w) for h, w in ((hs, ws), (ht, wt))]
        selected = []
        for role, z in zip(("student", "teacher"), logits, strict=True):
            ids = space.student_ids if role == "student" else space.teacher_ids
            if not ids:
                chosen = z
            elif space.spec.common_vocab_size is not None:
                chosen = z[:, : space.spec.common_vocab_size]
            else:
                chosen = z.index_select(-1, torch.tensor(ids, device=z.device))
            selected.append(chosen)
            out[role + "_excluded_mass"][start : start + len(z)] = space.excluded_from_logits(z, role)
        raw = distribution_values(*selected)
        out["uncoarsened_rkl"][start : start + len(raw)] = raw
        out["rkl"][start : start + len(raw)] = distribution_values(*selected, actions) if actions else raw
    return out


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
