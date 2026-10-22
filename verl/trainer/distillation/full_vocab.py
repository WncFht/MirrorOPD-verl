# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Exact fixed-prefix reverse KL from hidden states and linear heads.

Teacher is frozen; gradients are first-order only. No distributed communication
or loss normalization lives here. Return a token sum for the engine to normalize.
"""

import torch


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
                dh[start:stop] += project(dz, ws[v:end].T)
                dw[v:end] += project(dz.T.contiguous(), hs[start:stop].T)
        return dh.to(hs.dtype), dw.to(ws.dtype), None, None, None, None


class PrecomputedKL(torch.autograd.Function):
    """Keep only one token block's logits; accumulate gradients during forward."""

    @staticmethod
    def forward(ctx, hs, ws, ht, wt, token_chunk):
        dtype = torch.float64 if hs.dtype == torch.float64 else torch.float32
        dh = torch.zeros(hs.shape, device=hs.device, dtype=dtype)
        dw = torch.zeros(ws.shape, device=ws.device, dtype=dtype)
        total = torch.zeros((), device=hs.device, dtype=dtype)
        for start in range(0, len(hs), token_chunk):
            stop = min(start + token_chunk, len(hs))
            zp, zt = project(hs[start:stop], ws), project(ht[start:stop], wt)
            lp, lt = zp.log_softmax(-1), zt.log_softmax(-1)
            p = lp.exp()
            kl = (p * (lp - lt)).sum(-1)
            total += kl.sum()
            dz = (p * (lp - lt - kl[:, None])).to(hs.dtype)
            dh[start:stop] = project(dz, ws.T)
            dw += project(dz.T.contiguous(), hs[start:stop].T)
        ctx.save_for_backward(dh.to(hs.dtype), dw.to(ws.dtype))
        return total

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, upstream):
        dh, dw = ctx.saved_tensors
        return dh * upstream, dw * upstream, None, None, None


def full_reverse_kl(hs, ws, ht, wt, *, backend="recompute", token_chunk=512, vocab_chunk=8192):
    """Return unnormalized sum; empty local ranks remain attached to both parameters."""
    if backend not in ("dense", "recompute", "streaming", "precompute", "liger"):
        raise ValueError(f"Unknown full reverse-KL backend: {backend}")
    if hs.ndim != 2 or ht.ndim != 2 or ws.ndim != 2 or wt.ndim != 2:
        raise ValueError("full KL expects 2-D hidden states and heads")
    if len(hs) != len(ht) or len(ws) != len(wt) or hs.shape[1] != ws.shape[1] or ht.shape[1] != wt.shape[1]:
        raise ValueError("full KL hidden/head shapes are incompatible")
    if min(token_chunk, vocab_chunk) <= 0:
        raise ValueError("full KL chunk sizes must be positive")
    if not len(ws):
        raise ValueError("full KL requires a nonempty vocabulary")
    if len(hs) == 0:
        return hs.sum() * 0 + ws.sum() * 0
    ht, wt = ht.detach(), wt.detach()
    if backend == "dense":
        return dense_rows(hs, ws, ht, wt).sum()
    if backend in ("recompute", "streaming"):
        return RecomputedKL.apply(hs, ws, ht, wt, token_chunk, vocab_chunk if backend == "streaming" else 0).sum()
    if backend == "precompute":
        return PrecomputedKL.apply(hs, ws, ht, wt, token_chunk)
    if backend == "liger":
        from liger_kernel.transformers.fused_linear_jsd import LigerFusedLinearJSD

        loss = LigerFusedLinearJSD(jsd_beta=1.0, temperature=1.0, accum_dtype=torch.float32)
        labels = torch.zeros(len(hs), dtype=torch.long, device=hs.device)
        return loss(hs, ws, ht, wt, labels) * len(hs)
    raise ValueError(f"Unknown full reverse-KL backend: {backend}")
