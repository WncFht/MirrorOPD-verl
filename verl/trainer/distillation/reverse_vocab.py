# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Native reverse-KL execution strategies, including bounded vocabulary replay."""

from functools import partial

import torch

from verl.trainer.distillation.execution import resolve_execution, validate_fused_backend
from verl.trainer.distillation.results import FullKLResult
from verl.trainer.distillation.rows import dense_rows, precompute_kl_rows, termination_equivalent_kl_rows
from verl.trainer.distillation.termination import TerminationActionMap
from verl.trainer.distillation.vjp import add_product, precompute_rows
from verl.utils.linear import project


@torch.no_grad()
def row_statistics(hs, ws, ht, wt, token_chunk, vocab_chunk, *, cached=False):
    """Bounded-memory relative statistics; only row/tile arithmetic uses FP64.

    Streaming never allocates a token-by-full-vocabulary cache. Cached/recompute
    reuse projected logits within the current token block, not across updates.
    """
    from verl.trainer.distillation.numerics import expm1_minus_x, log1p_minus_x

    n, vocab = len(hs), len(ws)
    values = hs.new_empty(n, dtype=torch.float64 if hs.dtype == torch.float64 else torch.float32)
    statistics = hs.new_empty((n, 7), dtype=torch.float64)
    width = vocab_chunk or vocab
    for start in range(0, n, token_chunk):
        stop = min(start + token_chunk, n)
        sh, th = hs[start:stop], ht[start:stop]
        zp_cache = zt_cache = None
        if cached or not vocab_chunk:
            zp_cache = hs.new_empty((stop - start, vocab), dtype=values.dtype)
            zt_cache = torch.empty_like(zp_cache)
            for v in range(0, vocab, width):
                end = min(v + width, vocab)
                zp_cache[:, v:end] = project(sh, ws[v:end])
                zt_cache[:, v:end] = project(th, wt[v:end])

        def blocks(sh=sh, th=th, zp_cache=zp_cache, zt_cache=zt_cache):
            for v in range(0, vocab, width):
                end = min(v + width, vocab)
                zp = project(sh, ws[v:end]) if zp_cache is None else zp_cache[:, v:end]
                zt = project(th, wt[v:end]) if zt_cache is None else zt_cache[:, v:end]
                yield v, zp.double(), zt.double()

        maximum = sh.new_full((len(sh),), -torch.inf, dtype=torch.float64)
        pivot = sh.new_zeros(len(sh), dtype=torch.float64)
        teacher_pivot = torch.zeros_like(maximum)
        lzp, lzt = maximum.clone(), maximum.clone()
        for v, zp, zt in blocks():
            m, i = zp.max(-1)
            take = m > maximum
            pivot = torch.where(take, i + v, pivot)
            teacher_pivot = torch.where(take, zt.gather(-1, i[:, None]).squeeze(-1), teacher_pivot)
            maximum = torch.maximum(maximum, m)
            lzp = torch.logaddexp(lzp, zp.logsumexp(-1))
            lzt = torch.logaddexp(lzt, zt.logsumexp(-1))
        mass, moment = torch.zeros_like(maximum), torch.zeros_like(maximum)
        for _, zp, zt in blocks():
            w = (zp - lzp[:, None]).exp()
            residual = (zp - maximum[:, None]) - (zt - teacher_pivot[:, None])
            mass += w.sum(-1)
            moment += (w * residual).sum(-1)
        mu = moment / mass
        a, b, other_grad = [torch.zeros_like(maximum) for _ in range(3)]
        exp_max = maximum.new_full(maximum.shape, -torch.inf)
        for v, zp, zt in blocks():
            lp = zp - lzp[:, None]
            w = lp.exp()
            residual = (zp - maximum[:, None]) - (zt - teacher_pivot[:, None])
            x = -(residual - mu[:, None])
            remainder = torch.where(x.abs() < 0.125, w * expm1_minus_x(x), (lp + x).clamp(max=40).exp() - w * (1 + x))
            a += (w * x).sum(-1)
            b += remainder.sum(-1)
            exp_max = torch.maximum(exp_max, (lp + x).amax(-1))
            columns = torch.arange(v, v + zp.shape[1], device=hs.device)
            other_grad += torch.where(columns[None, :] == pivot[:, None], 0.0, -w * x / mass[:, None]).sum(-1)
        a, b = a / mass, b / mass
        sa, sb = a.clamp(-0.5, 0.5), b.clamp(-0.5, 0.5)
        close = torch.log1p(sb / (1 + sa)) + log1p_minus_x(sa)
        far = (lzt - teacher_pivot) - (lzp - maximum) + mu
        value = torch.where((exp_max < 40) & (a.abs() < 0.5) & (b.abs() < 0.5), close, far)
        values[start:stop] = value
        statistics[start:stop] = torch.stack([maximum, teacher_pivot, lzp, mu, mass, pivot, -other_grad], -1)
    return values, statistics


class RecomputedKL(torch.autograd.Function):
    """Recompute native projections; stable row statistics do not retain a graph."""

    @staticmethod
    def forward(ctx, hs, ws, ht, wt, token_chunk, vocab_chunk):
        kl, stats = row_statistics(hs, ws, ht, wt, token_chunk, vocab_chunk)
        ctx.save_for_backward(hs, ws, ht, wt, stats)
        ctx.token_chunk, ctx.vocab_chunk = token_chunk, vocab_chunk or ws.shape[0]
        return kl

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, upstream):
        hs, ws, ht, wt, statistics = ctx.saved_tensors
        dtype = torch.float64 if hs.dtype == torch.float64 else torch.float32
        dh = torch.zeros(hs.shape, device=hs.device, dtype=dtype)
        dw = torch.zeros(ws.shape, device=ws.device, dtype=dtype)
        for start in range(0, len(hs), ctx.token_chunk):
            stop = min(start + ctx.token_chunk, len(hs))
            maximum, reference, lzp, mu, mass, pivot, pivot_grad = statistics[start:stop].unbind(-1)
            for v in range(0, len(ws), ctx.vocab_chunk):
                end = min(v + ctx.vocab_chunk, len(ws))
                zp = project(hs[start:stop], ws[v:end]).double()
                zt = project(ht[start:stop], wt[v:end]).double()
                residual = (zp - maximum[:, None]) - (zt - reference[:, None])
                g = (zp - lzp[:, None]).exp() * (residual - mu[:, None]) / mass[:, None]
                columns = torch.arange(v, end, device=hs.device)
                g = torch.where(columns[None, :] == pivot[:, None], pivot_grad[:, None], g)
                dz = (g * upstream[start:stop, None]).to(hs.dtype)
                add_product(dh[start:stop], dz, ws[v:end])
                add_product(dw[v:end], dz.T, hs[start:stop])
        return dh.to(hs.dtype), dw.to(ws.dtype), None, None, None, None


class CachedKL(RecomputedKL):
    """Cache only the current token block, preserving vocab-chunk GEMM shapes."""

    @staticmethod
    def forward(ctx, hs, ws, ht, wt, token_chunk, vocab_chunk):
        kl, stats = row_statistics(hs, ws, ht, wt, token_chunk, vocab_chunk, cached=True)
        ctx.save_for_backward(hs, ws, ht, wt, stats)
        ctx.token_chunk, ctx.vocab_chunk = token_chunk, vocab_chunk
        return kl


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
    return FullKLResult(total, rows, rows)


def full_reverse_kl(
    hs, ws, ht, wt, *, backend="recompute", token_chunk=512, vocab_chunk=8192, termination_token_ids=()
):
    """Scalar optimization interface; empty termination IDs preserve vocabulary KL."""
    return full_reverse_kl_result(
        hs,
        ws,
        ht,
        wt,
        backend=backend,
        token_chunk=token_chunk,
        vocab_chunk=vocab_chunk,
        termination_token_ids=termination_token_ids,
    ).loss_sum


def validate_reverse_kl_inputs(hs, ws, ht, wt, *, backend, token_chunk, vocab_chunk, termination_token_ids):
    """Validate real tensors and capabilities without projecting logits or building a graph."""
    resolve_execution(backend, termination=bool(termination_token_ids))
    if hs.ndim != 2 or ht.ndim != 2 or ws.ndim != 2 or wt.ndim != 2:
        raise ValueError("full KL expects 2-D hidden states and heads")
    if len(hs) != len(ht) or hs.shape[1] != ws.shape[1] or ht.shape[1] != wt.shape[1]:
        raise ValueError("full KL hidden/head shapes are incompatible")
    if len(ws) > len(wt):
        raise ValueError("full reverse KL requires teacher support to cover every student output row")
    if any(type(c) is not int or c <= 0 for c in (token_chunk, vocab_chunk)):
        raise ValueError("full KL chunk sizes must be positive")
    if not len(ws):
        raise ValueError("full KL requires a nonempty vocabulary")
    ids = tuple(termination_token_ids)
    if ids:
        if len(ids) < 2 or len(set(ids)) != len(ids) or any(type(i) is not int or not 0 <= i < len(ws) for i in ids):
            raise ValueError("termination equivalence requires distinct supported token IDs")
        if len(ws) != len(wt):
            raise ValueError("termination equivalence requires precompute/precompute_fused and equal vocabulary widths")
    validate_fused_backend(backend, hs)
    return ids


def full_reverse_kl_result(
    hs,
    ws,
    ht,
    wt,
    *,
    backend="recompute",
    token_chunk=512,
    vocab_chunk=8192,
    termination_token_ids=(),
    row_weights=None,
) -> FullKLResult:
    """Compute the loss and reuse detached token rows when the backend exposes them.

    Kernel capability is represented by token_kl=None, not by engine-side backend
    checks. Diagnostics never contribute gradients or trigger extra projections.
    """
    ids = validate_reverse_kl_inputs(
        hs,
        ws,
        ht,
        wt,
        backend=backend,
        token_chunk=token_chunk,
        vocab_chunk=vocab_chunk,
        termination_token_ids=termination_token_ids,
    )
    if len(hs) == 0:
        total = hs.sum() * 0 + ws.sum() * 0
        rows = None if backend == "liger" else hs.new_empty(0).detach()
        return FullKLResult(total, rows, rows)
    ht, wt = ht.detach(), wt.detach()
    if len(wt) > len(ws):
        correction = teacher_support_correction(ht, wt, len(ws), token_chunk, vocab_chunk)
        result = full_reverse_kl_result(
            hs,
            ws,
            ht,
            wt[: len(ws)],
            backend=backend,
            token_chunk=token_chunk,
            vocab_chunk=vocab_chunk,
            row_weights=row_weights,
        )
        rows = None if result.token_kl is None else correction.token_kl + result.token_kl
        correction_sum = correction.loss_sum if row_weights is None else (correction.token_kl * row_weights).sum()
        return FullKLResult(correction_sum + result.loss_sum, rows, rows)

    def finish(rows):
        total = rows.sum() if row_weights is None else (rows * row_weights).sum()
        return FullKLResult(total, rows.detach(), rows.detach())

    if backend == "dense":
        return finish(dense_rows(hs, ws, ht, wt))
    if backend in ("recompute", "streaming"):
        return finish(RecomputedKL.apply(hs, ws, ht, wt, token_chunk, vocab_chunk if backend == "streaming" else 0))
    if backend in ("precompute", "precompute_fused"):
        fused = backend == "precompute_fused"
        if ids:
            actions = TerminationActionMap.create(len(ws), ids, hs.device)
            row_kernel = partial(termination_equivalent_kl_rows, actions=actions, fused=fused)
        elif fused:
            from verl.trainer.distillation.full_vocab_triton import fused_kl_rows

            row_kernel = fused_kl_rows
        else:
            row_kernel = precompute_kl_rows
        total, raw, objective_rows, _, _, _ = precompute_rows(
            hs, ws, ht, wt, token_chunk, partial(prepare_reverse, row_kernel=row_kernel), row_weights=row_weights
        )
        return FullKLResult(total, raw, objective_rows)
    if backend == "cached":
        return finish(CachedKL.apply(hs, ws, ht, wt, token_chunk, vocab_chunk))
    if backend == "liger":
        if row_weights is not None:
            raise ValueError("liger does not support per-row aggregation; use precompute")
        from liger_kernel.transformers.fused_linear_jsd import LigerFusedLinearJSD

        loss = LigerFusedLinearJSD(jsd_beta=1.0, temperature=1.0, accum_dtype=torch.float32)
        labels = torch.zeros(len(hs), dtype=torch.long, device=hs.device)
        return FullKLResult(loss(hs, ws, ht, wt, labels) * len(hs), None)
    raise ValueError(f"Unknown full reverse-KL backend: {backend}")


def prepare_reverse(hs, ws, ht, wt, hq, wq, *, row_kernel):
    output = row_kernel(hs, ws, ht, wt)
    values, dz = output[:2]
    raw = output[2] if len(output) > 2 else values
    return values, dz, raw, hs.new_empty((len(hs), 0)), None
