# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Exact fixed-prefix reverse KL from hidden states and linear heads.

Teacher is frozen; gradients are first-order only. No distributed communication
or loss normalization lives here. Return a token sum for the engine to normalize.
"""

from functools import partial
from typing import NamedTuple

import torch

from verl.trainer.distillation.profiling import full_kl_range
from verl.utils.linear import ProjectionFP32, project  # noqa: F401


class FullKLResult(NamedTuple):
    """Differentiable objective and detached raw vocabulary-KL diagnostics.

    With termination equivalence, loss_sum optimizes the coarsened action space;
    token_kl deliberately retains the original vocabulary metric.
    """

    loss_sum: torch.Tensor
    token_kl: torch.Tensor | None


def add_product(accumulator, left, right):
    """Accumulate directly into FP32 storage instead of allocating a full head gradient per tile."""
    if left.is_cuda and left.dtype in (torch.float16, torch.bfloat16) and accumulator.dtype == torch.float32:
        torch.addmm(accumulator, left, right, out_dtype=torch.float32, out=accumulator)
    else:
        accumulator.addmm_(left.to(accumulator.dtype), right.to(accumulator.dtype))


def dense_rows(hs, ws, ht, wt):
    from verl.trainer.distillation.numerics import kl_rows

    return kl_rows(project(hs, ws), project(ht.detach(), wt.detach()))[0]


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


def precompute_kl_rows(hs, ws, ht, wt):
    """One block's exact KL and logit VJP; release logits before gradient GEMMs."""
    with full_kl_range("student_projection"):
        zp = project(hs, ws)
    with full_kl_range("teacher_projection"):
        zt = project(ht, wt)
    from verl.trainer.distillation.numerics import kl_rows

    with full_kl_range("normalization_and_kl"):
        kl, dz = kl_rows(zp, zt)
    return kl, dz if hs.dtype == torch.float16 else dz.to(hs.dtype)


def termination_equivalent_kl_rows(hs, ws, ht, wt, *, termination_token_ids):
    """Coarsen a stopping-action class while preserving raw KL diagnostics.

    The class log-ratio is log(P(E)/Q(E)); its contribution to each member's
    logit VJP is p_i * (class_log_ratio - coarsened_KL). No stopping mass is
    dropped or reassigned to a preferred token ID.
    """
    from verl.trainer.distillation.numerics import kl_rows, log_softmax

    zp, zt = project(hs, ws), project(ht, wt)
    raw_kl, _ = kl_rows(zp, zt)
    ids = tuple(termination_token_ids)
    keep = torch.ones(len(ws), dtype=torch.bool, device=hs.device)
    keep[list(ids)] = False
    # Only this optional coarsened-action path promotes its row algebra. The
    # aggregated class logit must retain small within-class normalizer changes.
    student, teacher = zp.double(), zt.double()
    zs = torch.cat([student[:, keep], student[:, ids].logsumexp(-1, keepdim=True)], -1)
    ts = torch.cat([teacher[:, keep], teacher[:, ids].logsumexp(-1, keepdim=True)], -1)
    coarse_kl, gc = kl_rows(zs, ts)
    gradient = torch.empty_like(student)
    gradient[:, keep] = gc[:, :-1]
    gradient[:, ids] = gc[:, -1:] * log_softmax(student[:, ids]).exp()
    pivot = zp.argmax(-1, keepdim=True)
    nonpivot = gradient.scatter(-1, pivot, 0.0)
    gradient = nonpivot.scatter(-1, pivot, -nonpivot.sum(-1, keepdim=True))
    dtype = torch.float32 if hs.dtype == torch.float16 else hs.dtype
    return coarse_kl.to(zp.dtype), gradient.to(dtype), raw_kl


class PrecomputedKL(torch.autograd.Function):
    """Shared accumulation/VJP for ordinary and fused block reductions."""

    @staticmethod
    def forward(ctx, hs, ws, ht, wt, token_chunk, row_kernel, hq=None, wq=None, spec=None):
        ctx.input_count = len(ctx.needs_input_grad)
        dtype = torch.float64 if hs.dtype == torch.float64 else torch.float32
        dh = torch.zeros(hs.shape, device=hs.device, dtype=dtype)
        dw = torch.zeros(ws.shape, device=ws.device, dtype=dtype)
        total = torch.zeros((), device=hs.device, dtype=dtype)
        token_kl = torch.empty(len(hs), device=hs.device, dtype=dtype)
        diagnostics = torch.empty(
            (len(hs), 4 if spec is not None and spec.diagnostics else 0), device=hs.device, dtype=dtype
        )
        for start in range(0, len(hs), token_chunk):
            stop = min(start + token_chunk, len(hs))
            if spec is None:
                output = row_kernel(hs[start:stop], ws, ht[start:stop], wt)
                kl, dz = output[:2]
                diagnostic_kl = output[2] if len(output) == 3 else kl
                del output
            else:
                kl, dz, info = row_kernel(hs[start:stop], ws, ht[start:stop], wt, hq[start:stop], wq, spec)
                diagnostics[start:stop] = info
                diagnostic_kl = kl
            total += kl.sum()
            token_kl[start:stop] = diagnostic_kl
            with full_kl_range("grad_hidden"):
                add_product(dh[start:stop], dz, ws)
            with full_kl_range("grad_head"):
                add_product(dw, dz.T, hs[start:stop])
            del kl, dz
        ctx.hdtype, ctx.wdtype = hs.dtype, ws.dtype
        ctx.save_for_backward(
            dh if hs.dtype == torch.float16 else dh.to(hs.dtype),
            dw if ws.dtype == torch.float16 else dw.to(ws.dtype),
        )
        if spec is None:
            ctx.mark_non_differentiable(token_kl)
            return total, token_kl
        ctx.mark_non_differentiable(token_kl, diagnostics)
        return total, token_kl, diagnostics

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, upstream, *unused):
        dh, dw = ctx.saved_tensors
        with full_kl_range("backward_apply"):
            return ((dh * upstream).to(ctx.hdtype), (dw * upstream).to(ctx.wdtype)) + (None,) * (ctx.input_count - 2)


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


def full_reverse_kl_result(
    hs, ws, ht, wt, *, backend="recompute", token_chunk=512, vocab_chunk=8192, termination_token_ids=()
) -> FullKLResult:
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
    ids = tuple(termination_token_ids)
    if ids:
        if len(ids) < 2 or len(set(ids)) != len(ids) or any(type(i) is not int or not 0 <= i < len(ws) for i in ids):
            raise ValueError("termination equivalence requires distinct supported token IDs")
        if backend != "precompute" or len(ws) != len(wt):
            raise ValueError("termination equivalence requires precompute and equal vocabulary widths")
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
        row_kernel = partial(termination_equivalent_kl_rows, termination_token_ids=ids) if ids else precompute_kl_rows
        if backend == "precompute_fused" and hs.is_cuda and hs.dtype in (torch.float16, torch.bfloat16, torch.float32):
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


class FullVocabObjectiveResult(NamedTuple):
    """Unscaled objective sum and detached rows; final loss applies 1/eta once.

    For teacher-only KL, token_objective retains raw vocabulary KL even when
    loss_sum uses termination equivalence, so position metrics stay comparable.
    """

    loss_sum: torch.Tensor
    token_objective: torch.Tensor | None
    token_diagnostics: dict[str, torch.Tensor]


_MIRROR_DIAGNOSTICS = ("target_kl", "target_log_z", "anchor_kl", "teacher_kl")


def _coarsen_logits(logits, ids, keep):
    return torch.cat([logits[:, keep], logits[:, ids].logsumexp(-1, keepdim=True)], dim=-1)


def _mirror_rows(hs, ws, ht, wt, hq, wq, spec, *, fused=False, termination_token_ids=()):
    with full_kl_range("teacher_projection_and_normalization"):
        zt = project(ht, wt)
    with full_kl_range("anchor_projection_and_normalization"):
        zq = project(hq, wq)
    with full_kl_range("student_projection"):
        z = project(hs, ws)
    with full_kl_range("mirror_rows"):
        if fused:
            from verl.trainer.distillation.full_vocab_triton import fused_mirror_logits

            return fused_mirror_logits(z, zq, zt, torch.float32 if hs.dtype == torch.float16 else hs.dtype, spec=spec)
        from verl.trainer.distillation.numerics import mirror_logits_rows

        ids = tuple(termination_token_ids)
        if ids:
            # Construct the power-mean target in action space. Aggregation and
            # geometric mixing do not commute; coarsen q and teacher first.
            student, anchor, teacher = z.double(), zq.double(), zt.double()
            keep = torch.ones(len(ws), device=z.device, dtype=torch.bool)
            keep[list(ids)] = False
            rows, coarse_dz, info = mirror_logits_rows(
                _coarsen_logits(student, ids, keep),
                _coarsen_logits(anchor, ids, keep),
                _coarsen_logits(teacher, ids, keep),
                objective=spec.objective,
                kappa=spec.kappa,
                eta=spec.eta,
                diagnostics=spec.diagnostics,
            )
            # d logsumexp(z_E) / d z_i = p_i / P(E). The anchor and teacher
            # remain detached inside mirror_logits_rows; only learner logits chain back.
            dz = torch.empty_like(student)
            dz[:, keep] = coarse_dz[:, :-1]
            dz[:, ids] = coarse_dz[:, -1:] * student[:, ids].softmax(-1)
            pivot = z.argmax(-1, keepdim=True)
            others = dz.scatter(-1, pivot, 0.0)
            dz = others.scatter(-1, pivot, -others.sum(-1, keepdim=True))
        else:
            rows, dz, info = mirror_logits_rows(
                z, zq, zt, objective=spec.objective, kappa=spec.kappa, eta=spec.eta, diagnostics=spec.diagnostics
            )
        dz = dz.float() if hs.dtype == torch.float16 else dz.to(hs.dtype)
    diag = (
        torch.stack([info[k] for k in _MIRROR_DIAGNOSTICS], dim=-1) if spec.diagnostics else z.new_empty((len(hs), 0))
    )
    return rows, dz, diag


def full_vocab_objective_result(
    hs,
    ws,
    ht,
    wt,
    *,
    spec,
    hq=None,
    wq=None,
    backend="precompute",
    token_chunk=512,
    vocab_chunk=8192,
    termination_token_ids=(),
    forward_teacher_support="equal",
):
    """Dispatch teacher KL, MDPO and BPO; MDPO EOS diagnostics use coarsened actions."""
    if spec.teacher_only_forward_kl:
        if termination_token_ids:
            raise ValueError("forward KL does not support termination equivalence")
        from verl.trainer.distillation.forward_vocab import full_forward_kl_result

        return full_forward_kl_result(
            hs, ws, ht, wt, backend=backend, token_chunk=token_chunk, teacher_support=forward_teacher_support
        )
    if spec.teacher_only_kl:
        result = full_reverse_kl_result(
            hs,
            ws,
            ht,
            wt,
            backend=backend,
            token_chunk=token_chunk,
            vocab_chunk=vocab_chunk,
            termination_token_ids=termination_token_ids,
        )
        return FullVocabObjectiveResult(result.loss_sum, result.token_kl, {})
    if termination_token_ids:
        ids = tuple(termination_token_ids)
        if spec.objective != "mdpo" or backend not in ("dense", "precompute"):
            raise ValueError("Mirror termination equivalence requires MDPO with dense/precompute backend")
        if len(wt) != len(ws) or len(ids) < 2 or len(set(ids)) != len(ids):
            raise ValueError("Mirror termination equivalence requires equal vocabularies and distinct stop IDs")
        if any(type(i) is not int or i < 0 or i >= len(ws) for i in ids):
            raise ValueError("Mirror termination token IDs must be within the shared vocabulary")
    if spec.payload_kind != "hidden" or spec.objective not in ("mdpo", "bpo"):
        raise ValueError("Full-vocab mirror objective requires MDPO or BPO with hidden payload")
    if backend not in ("dense", "precompute", "precompute_fused"):
        raise ValueError(f"Unsupported mirror full-vocab backend: {backend}")
    if hq is None or wq is None:
        raise ValueError("Mirror full-vocab objective requires frozen anchor hidden/head")
    if any(x.ndim != 2 for x in (hs, ws, ht, wt, hq, wq)):
        raise ValueError("full-vocab expects 2-D hidden states and heads")
    if (
        len(hs) != len(ht)
        or len(hs) != len(hq)
        or len(wq) != len(ws)
        or len(wt) < len(ws)
        or hs.shape[1] != ws.shape[1]
        or ht.shape[1] != wt.shape[1]
        or hq.shape[1] != wq.shape[1]
    ):
        raise ValueError("Mirror hidden/head shapes or teacher/anchor support are incompatible")
    if min(token_chunk, vocab_chunk, len(ws)) <= 0:
        raise ValueError("Positive chunk sizes and nonempty vocabulary required")
    ht, wt, hq, wq = (x.detach() for x in (ht, wt, hq, wq))
    if len(hs) == 0:
        return FullVocabObjectiveResult(hs.sum() * 0 + ws.sum() * 0, hs.new_empty(0), {})
    if backend == "dense":
        rows, _, diag = _mirror_rows(hs, ws, ht, wt, hq, wq, spec, termination_token_ids=termination_token_ids)
        total = rows.sum()
    else:
        from functools import partial

        row_kernel = partial(
            _mirror_rows,
            fused=backend == "precompute_fused"
            and hs.is_cuda
            and hs.dtype in (torch.float16, torch.bfloat16, torch.float32),
            termination_token_ids=termination_token_ids,
        )
        total, rows, diag = PrecomputedKL.apply(hs, ws, ht, wt, token_chunk, row_kernel, hq, wq, spec)
    info = {k: diag[:, i].detach() for i, k in enumerate(_MIRROR_DIAGNOSTICS)} if spec.diagnostics else {}
    return FullVocabObjectiveResult(total, rows.detach(), info)
