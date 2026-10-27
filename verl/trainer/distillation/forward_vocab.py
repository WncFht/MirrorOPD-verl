# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Full forward KL with a frozen, explicitly conditioned teacher.

Only token chunks are projected. Precompute stores the student VJP, never an
autograd graph spanning response-by-vocabulary logits. Reverse KL diagnostics
retain the teacher's native normalizer for comparison with historical runs.
"""

import torch

from verl.trainer.distillation.full_vocab import FullVocabObjectiveResult, add_product, project
from verl.trainer.distillation.numerics import cumulant, log_softmax, mean, relative_difference

DIAGNOSTICS = ("full_kl", "teacher_outside_student_mass", "student_entropy", "teacher_entropy", "js", "tv")


def forward_rows(z, teacher, extra=None, *, fused=False, grad_dtype=None):
    """Return FKL, analytic student-logit VJP, and detached per-token diagnostics."""
    if z.ndim != 2 or teacher.shape != z.shape or not z.shape[-1]:
        raise ValueError("forward_rows requires equal shared-support logit shapes; pass extra teacher rows separately")
    if extra is not None and (extra.ndim != 2 or len(extra) != len(z)):
        raise ValueError("extra teacher logits must have matching token rows")
    teacher = teacher.detach()
    if fused and z.is_cuda and z.dtype == torch.float32:
        from verl.trainer.distillation.forward_vocab_triton import fused_forward_rows

        return fused_forward_rows(z.contiguous(), teacher.contiguous(), extra, grad_dtype or z.dtype)
    lp, lq = log_softmax(z), log_softmax(teacher)
    pivot = z.argmax(-1, keepdim=True)
    gap = relative_difference(z, teacher, pivot)
    fkl = cumulant(lq, gap)
    rkl = cumulant(lp, -gap)
    p, q = lp.exp(), lq.exp()
    p, q = p / p.sum(-1, keepdim=True), q / q.sum(-1, keepdim=True)
    # expm1 retains close p-q differences; direct subtraction handles large gaps
    # without overflowing exp(log p - log q) on a tiny teacher probability.
    log_ratio = gap - mean(lq, gap) - fkl[:, None]
    gradient = torch.where(log_ratio.abs() < 0.5, q * log_ratio.clamp(-0.5, 0.5).expm1(), p - q)
    other = gradient.scatter(-1, pivot, 0.0)
    gradient = other.scatter(-1, pivot, -other.sum(-1, keepdim=True))
    correction = torch.zeros_like(fkl)
    if extra is not None:
        maximum = teacher.max(-1, keepdim=True).values
        correction = torch.nn.functional.softplus(
            (extra.double() - maximum.double()).logsumexp(-1) - (teacher.double() - maximum.double()).logsumexp(-1)
        ).to(fkl.dtype)
    lm = torch.logaddexp(lp, lq) - z.new_tensor(2).log()
    info = torch.stack(
        [
            rkl + correction,
            -torch.expm1(-correction),
            -(p * lp).sum(-1),
            -(q * lq).sum(-1),
            0.5 * (p * (lp - lm) + q * (lq - lm)).sum(-1),
            0.5 * (p - q).abs().sum(-1),
        ],
        -1,
    )
    return fkl, gradient.to(grad_dtype or z.dtype), info


def block_rows(hs, ws, ht, wt, fused):
    # Project shared rows once; an extra teacher partition is usually only a
    # few padding rows, so native-normalizer diagnostics need no second full GEMM.
    z, teacher = project(hs, ws), project(ht, wt[: len(ws)])
    extra = project(ht, wt[len(ws) :]) if len(wt) > len(ws) else None
    dtype = torch.float32 if hs.dtype == torch.float16 else hs.dtype
    return forward_rows(z, teacher, extra, fused=fused, grad_dtype=dtype)


class ForwardVocab(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hs, ws, ht, wt, chunk, fused, precompute):
        dtype = torch.float64 if hs.dtype == torch.float64 else torch.float32
        values = hs.new_empty(len(hs), dtype=dtype)
        info = hs.new_empty((len(hs), len(DIAGNOSTICS)), dtype=dtype)
        dh = torch.zeros_like(hs, dtype=dtype) if precompute else None
        dw = torch.zeros_like(ws, dtype=dtype) if precompute else None
        for start in range(0, len(hs), chunk):
            stop = min(start + chunk, len(hs))
            values[start:stop], dz, info[start:stop] = block_rows(hs[start:stop], ws, ht[start:stop], wt, fused)
            if precompute:
                add_product(dh[start:stop], dz, ws)
                add_product(dw, dz.T, hs[start:stop])
            del dz
        ctx.precompute, ctx.chunk, ctx.fused = precompute, chunk, fused
        ctx.hdtype, ctx.wdtype = hs.dtype, ws.dtype
        if precompute:
            ctx.save_for_backward(
                dh if hs.dtype == torch.float16 else dh.to(hs.dtype),
                dw if ws.dtype == torch.float16 else dw.to(ws.dtype),
            )
        else:
            ctx.save_for_backward(hs, ws, ht, wt)
        ctx.mark_non_differentiable(values, info)
        return values.sum(), values, info

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, upstream, *unused):
        if ctx.precompute:
            dh, dw = ctx.saved_tensors
        else:
            hs, ws, ht, wt = ctx.saved_tensors
            dtype = torch.float64 if hs.dtype == torch.float64 else torch.float32
            dh, dw = torch.zeros_like(hs, dtype=dtype), torch.zeros_like(ws, dtype=dtype)
            for start in range(0, len(hs), ctx.chunk):
                stop = min(start + ctx.chunk, len(hs))
                _, dz, _ = block_rows(hs[start:stop], ws, ht[start:stop], wt, ctx.fused)
                add_product(dh[start:stop], dz, ws)
                add_product(dw, dz.T, hs[start:stop])
        return (dh * upstream).to(ctx.hdtype), (dw * upstream).to(ctx.wdtype), None, None, None, None, None


def full_forward_kl_result(hs, ws, ht, wt, *, backend="precompute", token_chunk=512, teacher_support="equal"):
    if backend not in ("dense", "recompute", "precompute", "precompute_fused"):
        raise ValueError("full forward KL supports dense/recompute/precompute/precompute_fused")
    if teacher_support not in ("equal", "student"):
        raise ValueError("forward_teacher_support must be equal or student")
    if any(x.ndim != 2 for x in (hs, ws, ht, wt)) or len(hs) != len(ht):
        raise ValueError("full forward KL expects aligned 2-D hidden states and heads")
    if hs.shape[1] != ws.shape[1] or ht.shape[1] != wt.shape[1] or not len(ws) or len(wt) < len(ws):
        raise ValueError("full forward KL requires compatible heads and teacher support covering student rows")
    if teacher_support == "equal" and len(ws) != len(wt):
        raise ValueError("unequal heads require explicit forward_teacher_support=student")
    if type(token_chunk) is not int or token_chunk <= 0:
        raise ValueError("forward KL token_chunk must be positive")
    ht, wt = ht.detach(), wt.detach()
    if not len(hs):
        zero = hs.sum() * 0 + ws.sum() * 0
        return FullVocabObjectiveResult(zero, hs.new_empty(0), {k: hs.new_empty(0) for k in DIAGNOSTICS})
    if backend == "dense":
        z, t = project(hs, ws), project(ht, wt[: len(ws)])
        extra = project(ht, wt[len(ws) :]) if len(wt) > len(ws) else None
        values, _, info = forward_rows(z, t, extra)
        total = values.sum()
    else:
        total, values, info = ForwardVocab.apply(
            hs, ws, ht, wt, token_chunk, backend == "precompute_fused", backend != "recompute"
        )
    return FullVocabObjectiveResult(total, values.detach(), {k: info[:, i].detach() for i, k in enumerate(DIAGNOSTICS)})
