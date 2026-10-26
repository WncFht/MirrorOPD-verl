#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Fixed-checkpoint interventions for target encoding, row arithmetic and head VJP.

Uses replay_fixed_prefix's CLI and engine. Extra branch kernels: encoded64,
ideal64, ideal_head64 and centered32. Native student/teacher/anchor projections,
BF16 backbone, frozen anchor, optimizer state and final 1/eta remain unchanged.
The ideal oracle reconstructs the target in FP64 from native FP32 logits; it
cannot recover information lost in the backbone or the head forward projection.
These are diagnostic kernels, not production backends or a performance claim.
"""

import json
import os
import sys
from pathlib import Path

import replay_fixed_prefix as replay
import torch

from verl.trainer.distillation import full_vocab as kernels
from verl.trainer.distillation.full_vocab_triton import fused_log_weights, fused_mirror_rows
from verl.trainer.distillation.targets import build_log_weights

_SEEN = set()


def inputs(hs, ws, ht, wt, hq, wq, spec):
    z, u, a = kernels.project(hs, ws), kernels.project(ht, wt), kernels.project(hq, wq)
    lq, lt = a.log_softmax(-1), u.log_softmax(-1)[:, : len(ws)]
    b = fused_log_weights(lt, lq, kappa=spec.kappa, eta=spec.eta)
    return z, u, a, lq, lt, b


def rows(z, u, a, lq, lt, b, spec, mode):
    dtype = torch.float32 if mode == "centered32" else torch.float64
    lp = z.to(dtype).log_softmax(-1)
    if mode == "ideal64":
        lq = a.double().log_softmax(-1)
        lt = u.double().log_softmax(-1)[:, : z.shape[1]]
        b = build_log_weights(lt, lq, kappa=spec.kappa, eta=spec.eta)
    else:
        lq, lt, b = lq.to(dtype), lt.to(dtype), b.to(dtype)
    p = lp.exp()
    # Normalize the encoded anchor even when isolating an unchanged FP32 b.
    q = lq.softmax(-1)
    w = p if spec.objective == "mdpo" else q
    pivot = w.argmax(-1, keepdim=True)
    zz = z.to(dtype)
    residual = (zz - zz.gather(-1, pivot)) - (b - b.gather(-1, pivot))
    if mode == "ideal64" and spec.kappa == 0:
        # Constants cancel in the VJP; retain the small target displacement.
        aa, uu = a.double(), u.double()[:, : z.shape[1]]
        rz, ra, rt = [v - v.gather(-1, pivot) for v in (zz, aa, uu)]
        residual = (rz - ra) - spec.eta * (rt - ra)
    centered = residual - (w * residual).sum(-1, keepdim=True)
    g = w * centered
    nonpivot = g.scatter(-1, pivot, 0.0)
    g = nonpivot.scatter(-1, pivot, -nonpivot.sum(-1, keepdim=True))
    mdpo = (p * (lp - b)).sum(-1)
    value = mdpo if spec.objective == "mdpo" else (q * centered.square()).sum(-1) / 2
    lz = b.logsumexp(-1)
    diag = torch.stack([mdpo + lz, lz, (p * (lp - lq)).sum(-1), (p * (lp - lt)).sum(-1)], -1)
    return value, g, diag


def error(actual, reference):
    delta = actual.double() - reference
    norms = reference.norm(dim=-1)
    relative = delta.norm(dim=-1) / norms.clamp_min(1e-30)
    return {
        "relative_l2": (delta.norm() / reference.norm().clamp_min(1e-30)).item(),
        "row_relative_p90": relative.quantile(0.9).item(),
        "rows_relative_gt1": (relative > 1).sum().item(),
        "absolute_error_norm": delta.norm().item(),
        "reference_norm": reference.norm().item(),
        "zero_reference_rows": (norms == 0).sum().item(),
        "gradient_sum_abs_max": actual.double().sum(-1).abs().max().item(),
    }


def record_initial(tensors, spec):
    key = (spec.objective, spec.kappa, spec.eta)
    if key in _SEEN:
        return
    _SEEN.add(key)
    z, u, a, lq, lt, b = tensors
    _, ideal, _ = rows(*tensors, spec, "ideal64")
    _, encoded, _ = rows(*tensors, spec, "encoded64")
    _, stable, _ = rows(*tensors, spec, "centered32")
    _, current32, _ = fused_mirror_rows(z, b, lq, lt, torch.float32, objective=spec.objective, diagnostics=True)
    _, current16, _ = fused_mirror_rows(z, b, lq, lt, torch.bfloat16, objective=spec.objective, diagnostics=True)
    result = {
        "scope": "First native training micro-batch; unscaled row VJPs, all methods use the same logits",
        "objective": spec.objective,
        "kappa": spec.kappa,
        "eta": spec.eta,
        "tokens": len(z),
        "vocab": z.shape[1],
        "current_fp32_vs_encoded": error(current32, encoded),
        "current_fp32_vs_ideal": error(current32, ideal),
        "current_bf16_vs_ideal": error(current16, ideal),
        "encoded64_vs_ideal": error(encoded, ideal),
        "centered32_vs_ideal": error(stable, ideal),
        "ideal_bf16_vs_ideal": error(ideal.to(torch.bfloat16), ideal),
    }
    output = Path(sys.argv[sys.argv.index("--out") + 1])
    (output / f"initial-vjp-rank{os.environ['RANK']}.json").write_text(json.dumps(result, indent=2) + "\n")


def kernel(mode):
    def row_kernel(hs, ws, ht, wt, hq, wq, spec):
        tensors = inputs(hs, ws, ht, wt, hq, wq, spec)
        record_initial(tensors, spec)
        values, dz, diag = rows(*tensors, spec, mode)
        return values, dz.to(hs.dtype), diag

    def result(hs, ws, ht, wt, *, spec, hq, wq, token_chunk, **kwargs):
        total, values, diag = kernels.PrecomputedKL.apply(hs, ws, ht, wt, token_chunk, row_kernel, hq, wq, spec)
        return kernels.FullVocabObjectiveResult(
            total, values, dict(zip(kernels._MIRROR_DIAGNOSTICS, diag.unbind(-1), strict=True))
        )

    return result


class IdealHead(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hs, ws, ht, wt, hq, wq, spec):
        tensors = inputs(hs, ws, ht, wt, hq, wq, spec)
        record_initial(tensors, spec)
        values, dz, diag = rows(*tensors, spec, "ideal64")
        ctx.save_for_backward(dz @ ws.double(), dz.T @ hs.double())
        ctx.hdtype, ctx.wdtype = hs.dtype, ws.dtype
        ctx.mark_non_differentiable(values, diag)
        return values.sum(), values, diag

    @staticmethod
    def backward(ctx, upstream, *unused):
        dh, dw = ctx.saved_tensors
        return (
            (dh * upstream.double()).to(ctx.hdtype),
            (dw * upstream.double()).to(ctx.wdtype),
            None,
            None,
            None,
            None,
            None,
        )


def ideal_head(hs, ws, ht, wt, *, spec, hq, wq, **kwargs):
    total, values, diag = IdealHead.apply(hs, ws, ht, wt, hq, wq, spec)
    return kernels.FullVocabObjectiveResult(
        total, values, dict(zip(kernels._MIRROR_DIAGNOSTICS, diag.unbind(-1), strict=True))
    )


if __name__ == "__main__":
    replay.main(
        extra_kernels={
            "encoded64": kernel("encoded64"),
            "ideal64": kernel("ideal64"),
            "ideal_head64": ideal_head,
            "centered32": kernel("centered32"),
        }
    )
