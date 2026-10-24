# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Exact fixed-prefix reverse KL from hidden states and linear heads.

Teacher is frozen; gradients are first-order only. No distributed communication
or loss normalization lives here. Return a token sum for the engine to normalize.
"""

from typing import NamedTuple

import torch

from verl.trainer.distillation.profiling import full_kl_range


class FullKLResult(NamedTuple):
    """One differentiable objective and optional detached token diagnostics."""

    loss_sum: torch.Tensor
    token_kl: torch.Tensor | None


class ProjectionFP32(torch.autograd.Function):
    """Ordinary linear VJP for torch.mm(out_dtype=...), which lacks autograd here."""

    @staticmethod
    def forward(ctx, hidden, weight):
        ctx.save_for_backward(hidden, weight)
        return torch.mm(hidden, weight.T, out_dtype=torch.float32)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, upstream):
        hidden, weight = ctx.saved_tensors
        dz = upstream.to(hidden.dtype)
        return dz @ weight, dz.T @ hidden


def project(hidden, weight):
    if hidden.is_cuda and hidden.dtype == torch.bfloat16:
        if torch.is_grad_enabled() and (hidden.requires_grad or weight.requires_grad):
            return ProjectionFP32.apply(hidden, weight)
        return torch.mm(hidden, weight.T, out_dtype=torch.float32)
    return hidden @ weight.T


def add_product(accumulator, left, right):
    """Accumulate directly into FP32 storage instead of allocating a full head gradient per tile."""
    if left.is_cuda and left.dtype == torch.bfloat16 and accumulator.dtype == torch.float32:
        torch.addmm(accumulator, left, right, out_dtype=torch.float32, out=accumulator)
    else:
        accumulator.addmm_(left, right)


def dense_rows(hs, ws, ht, wt):
    lp = project(hs, ws).log_softmax(-1)
    lt = project(ht.detach(), wt.detach()).log_softmax(-1)
    return (lp.exp() * (lp - lt)).sum(-1)


@torch.no_grad()
def row_statistics(hs, ws, ht, wt, token_chunk, vocab_chunk):
    """Return per-row KL, student log-normalizer, and mean logit gap."""
    n, vocab = hs.shape[0], ws.shape[0]
    dtype = torch.float64 if hs.dtype == torch.float64 else torch.float32
    kl, log_zp, mean_gap = [torch.empty(n, device=hs.device, dtype=dtype) for _ in range(3)]
    for start in range(0, n, token_chunk):
        stop = min(start + token_chunk, n)
        sh, th = hs[start:stop], ht[start:stop]
        if vocab_chunk == 0:
            zp, zt = project(sh, ws), project(th, wt)
            lp, lt = zp.log_softmax(-1), zt.log_softmax(-1)
            p = lp.exp()
            kl[start:stop] = (p * (lp - lt)).sum(-1)
            log_zp[start:stop] = zp.logsumexp(-1)
            mean_gap[start:stop] = (p * (zp - zt)).sum(-1)
            del zp, zt, lp, lt, p
        else:
            mp, mt = [torch.full((stop - start,), -torch.inf, device=hs.device, dtype=dtype) for _ in range(2)]
            sp, st, gap = [torch.zeros_like(mp) for _ in range(3)]
            for v in range(0, vocab, vocab_chunk):
                zp = project(sh, ws[v : v + vocab_chunk])
                zt = project(th, wt[v : v + vocab_chunk])
                np, nt = torch.maximum(mp, zp.max(-1).values), torch.maximum(mt, zt.max(-1).values)
                ap, at = (mp - np).exp(), (mt - nt).exp()
                ep, et = (zp - np[:, None]).exp(), (zt - nt[:, None]).exp()
                sp, st = ap * sp + ep.sum(-1), at * st + et.sum(-1)
                gap = ap * gap + (ep * (zp - zt)).sum(-1)
                mp, mt = np, nt
                del zp, zt, ep, et
            log_zp[start:stop] = mp + sp.log()
            mean_gap[start:stop] = gap / sp
            kl[start:stop] = gap / sp - log_zp[start:stop] + mt + st.log()
    return kl, log_zp, mean_gap


class RecomputedKL(torch.autograd.Function):
    """First-order custom VJP; teacher frozen, head and hidden both trainable."""

    @staticmethod
    def forward(ctx, hs, ws, ht, wt, token_chunk, vocab_chunk):
        kl, log_zp, mean_gap = row_statistics(hs, ws, ht, wt, token_chunk, vocab_chunk)
        ctx.save_for_backward(hs, ws, ht, wt, log_zp, mean_gap)
        ctx.token_chunk, ctx.vocab_chunk = token_chunk, vocab_chunk or ws.shape[0]
        return kl

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, upstream):
        hs, ws, ht, wt, log_zp, mean_gap = ctx.saved_tensors
        dtype = log_zp.dtype
        dh = torch.zeros(hs.shape, device=hs.device, dtype=dtype)
        dw = torch.zeros(ws.shape, device=ws.device, dtype=dtype)
        for start in range(0, len(hs), ctx.token_chunk):
            stop = min(start + ctx.token_chunk, len(hs))
            for v in range(0, len(ws), ctx.vocab_chunk):
                end = min(v + ctx.vocab_chunk, len(ws))
                zp = project(hs[start:stop], ws[v:end])
                zt = project(ht[start:stop], wt[v:end])
                p = (zp - log_zp[start:stop, None]).exp()
                dz = p * (zp - zt - mean_gap[start:stop, None]) * upstream[start:stop, None]
                # Match low-precision GEMM inputs used for student gradient products.
                dz = dz.to(hs.dtype)
                add_product(dh[start:stop], dz, ws[v:end])
                add_product(dw[v:end], dz.T, hs[start:stop])
                del zp, zt, p, dz
        return dh.to(hs.dtype), dw.to(ws.dtype), None, None, None, None


class CachedKL(RecomputedKL):
    """Two-dimensional projection, current-token-block logits cache, backward recomputation."""

    @staticmethod
    def forward(ctx, hs, ws, ht, wt, token_chunk, vocab_chunk):
        dtype = torch.float64 if hs.dtype == torch.float64 else torch.float32
        n, vocab = len(hs), len(ws)
        kl, log_zp, mean_gap = [torch.empty(n, device=hs.device, dtype=dtype) for _ in range(3)]
        cache_p, cache_t = [torch.empty((min(token_chunk, n), vocab), device=hs.device, dtype=dtype) for _ in range(2)]
        for start in range(0, n, token_chunk):
            stop = min(start + token_chunk, n)
            zp, zt = cache_p[: stop - start], cache_t[: stop - start]
            for v in range(0, vocab, vocab_chunk):
                end = min(v + vocab_chunk, vocab)
                zp[:, v:end] = project(hs[start:stop], ws[v:end])
                zt[:, v:end] = project(ht[start:stop], wt[v:end])
            log_zp[start:stop] = zp.logsumexp(-1)
            log_zt = zt.logsumexp(-1)
            gap = torch.zeros(stop - start, device=hs.device, dtype=dtype)
            for v in range(0, vocab, vocab_chunk):
                end = min(v + vocab_chunk, vocab)
                p = (zp[:, v:end] - log_zp[start:stop, None]).exp()
                gap += (p * (zp[:, v:end] - zt[:, v:end])).sum(-1)
            mean_gap[start:stop] = gap
            kl[start:stop] = gap - log_zp[start:stop] + log_zt
        ctx.save_for_backward(hs, ws, ht, wt, log_zp, mean_gap)
        ctx.token_chunk, ctx.vocab_chunk = token_chunk, vocab_chunk
        return kl


def precompute_kl_rows(hs, ws, ht, wt):
    """One block's exact KL and logit VJP; release logits before gradient GEMMs."""
    with full_kl_range("student_projection"):
        zp = project(hs, ws)
    with full_kl_range("teacher_projection"):
        zt = project(ht, wt)
    with full_kl_range("normalization_and_kl"):
        lp, lt = zp.log_softmax(-1), zt.log_softmax(-1)
        del zp, zt
        p = lp.exp()
        lp.sub_(lt)
        del lt
        kl = (p * lp).sum(-1)
    with full_kl_range("grad_logits"):
        lp.sub_(kl[:, None]).mul_(p)
        dz = lp.to(hs.dtype)
    return kl, dz


class PrecomputedKL(torch.autograd.Function):
    """Shared accumulation/VJP for ordinary and fused block reductions."""

    @staticmethod
    def forward(ctx, hs, ws, ht, wt, token_chunk, row_kernel):
        dtype = torch.float64 if hs.dtype == torch.float64 else torch.float32
        dh = torch.zeros(hs.shape, device=hs.device, dtype=dtype)
        dw = torch.zeros(ws.shape, device=ws.device, dtype=dtype)
        total = torch.zeros((), device=hs.device, dtype=dtype)
        token_kl = torch.empty(len(hs), device=hs.device, dtype=dtype)
        for start in range(0, len(hs), token_chunk):
            stop = min(start + token_chunk, len(hs))
            kl, dz = row_kernel(hs[start:stop], ws, ht[start:stop], wt)
            total += kl.sum()
            token_kl[start:stop] = kl
            with full_kl_range("grad_hidden"):
                add_product(dh[start:stop], dz, ws)
            with full_kl_range("grad_head"):
                add_product(dw, dz.T, hs[start:stop])
            del kl, dz
        ctx.save_for_backward(dh.to(hs.dtype), dw.to(ws.dtype))
        ctx.mark_non_differentiable(token_kl)
        return total, token_kl

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, upstream, _token_kl_grad):
        dh, dw = ctx.saved_tensors
        with full_kl_range("backward_apply"):
            return dh * upstream, dw * upstream, None, None, None, None


@torch.no_grad()
def teacher_support_correction(ht, wt, student_vocab, token_chunk, vocab_chunk):
    """Sum log(Z_teacher / Z_teacher_on_student_support), with bounded memory.

    Token IDs must be aligned by the caller. The student's distribution has zero
    mass on extra teacher rows. This frozen correction preserves the exact loss
    while allowing every equal-support backend to keep its existing VJP.
    """
    total = ht.new_zeros((), dtype=torch.float64 if ht.dtype == torch.float64 else torch.float32)
    rows = torch.empty(len(ht), device=ht.device, dtype=total.dtype)
    for start in range(0, len(ht), token_chunk):
        hidden = ht[start : start + token_chunk]
        partitions = []
        for begin, end in ((0, student_vocab), (student_vocab, len(wt))):
            log_z = total.new_full((len(hidden),), -torch.inf)
            for v in range(begin, end, vocab_chunk):
                logits = project(hidden, wt[v : min(v + vocab_chunk, end)])
                log_z = torch.logaddexp(log_z, logits.logsumexp(-1))
            partitions.append(log_z)
        correction = torch.nn.functional.softplus(partitions[1] - partitions[0])
        total += correction.sum()
        rows[start : start + len(hidden)] = correction
    return FullKLResult(total, rows)


def full_reverse_kl(hs, ws, ht, wt, *, backend="recompute", token_chunk=512, vocab_chunk=8192):
    """Scalar loss interface; preserve existing callers and the original objective."""
    return full_reverse_kl_result(
        hs, ws, ht, wt, backend=backend, token_chunk=token_chunk, vocab_chunk=vocab_chunk
    ).loss_sum


def full_reverse_kl_result(hs, ws, ht, wt, *, backend="recompute", token_chunk=512, vocab_chunk=8192) -> FullKLResult:
    """Compute the loss and reuse detached token rows when the backend exposes them.

    Kernel capability is represented by token_kl=None, not by engine-side backend
    checks. Diagnostics never contribute gradients or trigger extra projections.
    """
    if backend not in ("dense", "recompute", "cached", "streaming", "precompute", "precompute_fused", "liger"):
        raise ValueError(f"Unknown full reverse-KL backend: {backend}")
    if hs.ndim != 2 or ht.ndim != 2 or ws.ndim != 2 or wt.ndim != 2:
        raise ValueError("full KL expects 2-D hidden states and heads")
    if len(hs) != len(ht) or hs.shape[1] != ws.shape[1] or ht.shape[1] != wt.shape[1]:
        raise ValueError("full KL hidden/head shapes are incompatible")
    if len(ws) > len(wt):
        raise ValueError("full reverse KL requires teacher support to cover every student output row")
    if min(token_chunk, vocab_chunk) <= 0:
        raise ValueError("full KL chunk sizes must be positive")
    if not len(ws):
        raise ValueError("full KL requires a nonempty vocabulary")
    if len(hs) == 0:
        total = hs.sum() * 0 + ws.sum() * 0
        rows = None if backend == "liger" else hs.new_empty(0).detach()
        return FullKLResult(total, rows)
    ht, wt = ht.detach(), wt.detach()
    if len(wt) > len(ws):
        correction = teacher_support_correction(ht, wt, len(ws), token_chunk, vocab_chunk)
        result = full_reverse_kl_result(
            hs, ws, ht, wt[: len(ws)], backend=backend, token_chunk=token_chunk, vocab_chunk=vocab_chunk
        )
        rows = None if result.token_kl is None else correction.token_kl + result.token_kl
        return FullKLResult(correction.loss_sum + result.loss_sum, rows)

    def finish(rows):
        return FullKLResult(rows.sum(), rows.detach())

    if backend == "dense":
        return finish(dense_rows(hs, ws, ht, wt))
    if backend in ("recompute", "streaming"):
        return finish(RecomputedKL.apply(hs, ws, ht, wt, token_chunk, vocab_chunk if backend == "streaming" else 0))
    if backend in ("precompute", "precompute_fused"):
        row_kernel = precompute_kl_rows
        if backend == "precompute_fused" and hs.is_cuda and hs.dtype == torch.bfloat16:
            from verl.trainer.distillation.full_vocab_triton import fused_kl_rows

            row_kernel = fused_kl_rows
        return FullKLResult(*PrecomputedKL.apply(hs, ws, ht, wt, token_chunk, row_kernel))
    if backend == "cached":
        return finish(CachedKL.apply(hs, ws, ht, wt, token_chunk, vocab_chunk))
    if backend == "liger":
        from liger_kernel.transformers.fused_linear_jsd import LigerFusedLinearJSD

        loss = LigerFusedLinearJSD(jsd_beta=1.0, temperature=1.0, accum_dtype=torch.float32)
        labels = torch.zeros(len(hs), dtype=torch.long, device=hs.device)
        return FullKLResult(loss(hs, ws, ht, wt, labels) * len(hs), None)
    raise ValueError(f"Unknown full reverse-KL backend: {backend}")
