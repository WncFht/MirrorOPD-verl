# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""BF16 CUDA full-vocab kernels: teacher KL, weighted targets and MDPO/BPO.

Tiled reductions share diagnostics without dense softmax intermediates. cuBLAS
still does projections and gradient products. The public dispatcher falls back
to ordinary precompute for other devices/dtypes.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from verl.trainer.distillation.full_vocab import project
from verl.trainer.distillation.profiling import full_kl_range


@triton.jit
def _partials(ZP, ZT, MP, MT, SP, ST, GAP, V: tl.constexpr, P: tl.constexpr, B: tl.constexpr):
    row, part = tl.program_id(0), tl.program_id(1)
    col = part * B + tl.arange(0, B)
    valid = col < V
    a = tl.load(ZP + row * V + col, valid, 0).to(tl.float32)
    b = tl.load(ZT + row * V + col, valid, 0).to(tl.float32)
    ma = tl.max(tl.where(valid, a, -float("inf")), 0)
    mb = tl.max(tl.where(valid, b, -float("inf")), 0)
    ea = tl.where(valid, tl.exp(a - ma), 0.0)
    eb = tl.where(valid, tl.exp(b - mb), 0.0)
    ix = row * P + part
    tl.store(MP + ix, ma)
    tl.store(MT + ix, mb)
    tl.store(SP + ix, tl.sum(ea, 0))
    tl.store(ST + ix, tl.sum(eb, 0))
    tl.store(GAP + ix, tl.sum(ea * (a - b), 0))


@triton.jit
def _rows(MP, MT, SP, ST, GAP, LZ, MU, KL, P: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    j = tl.arange(0, B)
    ix = row * P + j
    mp = tl.load(MP + ix, j < P, -float("inf"))
    mt = tl.load(MT + ix, j < P, -float("inf"))
    ap = tl.exp(mp - tl.max(mp, 0))
    at = tl.exp(mt - tl.max(mt, 0))
    sp = tl.sum(ap * tl.load(SP + ix, j < P, 0), 0)
    st = tl.sum(at * tl.load(ST + ix, j < P, 0), 0)
    mu = tl.sum(ap * tl.load(GAP + ix, j < P, 0), 0) / sp
    lzp = tl.max(mp, 0) + tl.log(sp)
    lzt = tl.max(mt, 0) + tl.log(st)
    tl.store(LZ + row, lzp)
    tl.store(MU + row, mu)
    tl.store(KL + row, mu - lzp + lzt)


@triton.jit
def _gradient(ZP, ZT, LZ, MU, DZ, V: tl.constexpr, B: tl.constexpr):
    row, part = tl.program_id(0), tl.program_id(1)
    col = part * B + tl.arange(0, B)
    ix = row * V + col
    a = tl.load(ZP + ix, col < V, 0)
    b = tl.load(ZT + ix, col < V, 0)
    p = tl.exp(a - tl.load(LZ + row))
    dz = p * (a - b - tl.load(MU + row))
    tl.store(DZ + ix, dz, col < V)


def fused_rows(zp, zt, dtype):
    assert zp.is_cuda and zp.is_contiguous() and zt.is_contiguous()
    n, v = zp.shape
    parts = triton.cdiv(v, 4096)
    buffers = [torch.empty((n, parts), device=zp.device, dtype=torch.float32) for _ in range(5)]
    lz, mu, kl = [torch.empty(n, device=zp.device, dtype=torch.float32) for _ in range(3)]
    dz = torch.empty_like(zp, dtype=dtype)
    _partials[(n, parts)](zp, zt, *buffers, v, parts, 4096)
    _rows[(n,)](*buffers, lz, mu, kl, parts, triton.next_power_of_2(parts))
    _gradient[(n, parts)](zp, zt, lz, mu, dz, v, 4096)
    return kl, dz


def fused_kl_rows(hs, ws, ht, wt):
    """Fused block reduction; allocation and gradient accumulation live in the caller."""
    if not hs.is_cuda or hs.dtype != torch.bfloat16:
        raise ValueError("fused precompute requires BF16 CUDA")
    with full_kl_range("student_projection"):
        zp = project(hs, ws)
    with full_kl_range("teacher_projection"):
        zt = project(ht, wt)
    with full_kl_range("fused_kl_and_grad_logits"):
        return fused_rows(zp, zt, hs.dtype)


@triton.jit
def _bpo_partial_mean(Z, BIAS, LQ, PART, V: tl.constexpr, P: tl.constexpr, B: tl.constexpr):
    row, part = tl.program_id(0), tl.program_id(1)
    col = part * B + tl.arange(0, B)
    valid = col < V
    ix = row * V + col
    q = tl.where(valid, tl.exp(tl.load(LQ + ix, valid, -float("inf"))), 0.0)
    d = tl.load(Z + ix, valid, 0).to(tl.float32) - tl.load(BIAS + ix, valid, 0).to(tl.float32)
    tl.store(PART + row * P + part, tl.sum(q * d, 0))


@triton.jit
def _bpo_mean(PART, MU, P: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    j = tl.arange(0, B)
    tl.store(MU + row, tl.sum(tl.load(PART + row * P + j, j < P, 0), 0))


@triton.jit
def _bpo_gradient(Z, BIAS, LQ, MU, DZ, LOSS, V: tl.constexpr, P: tl.constexpr, B: tl.constexpr):
    row, part = tl.program_id(0), tl.program_id(1)
    col = part * B + tl.arange(0, B)
    valid = col < V
    ix = row * V + col
    q = tl.where(valid, tl.exp(tl.load(LQ + ix, valid, -float("inf"))), 0.0)
    d = tl.load(Z + ix, valid, 0).to(tl.float32) - tl.load(BIAS + ix, valid, 0).to(tl.float32)
    centered = d - tl.load(MU + row)
    tl.store(DZ + ix, q * centered, valid)
    tl.store(LOSS + row * P + part, 0.5 * tl.sum(q * centered * centered, 0))


def fused_bpo_rows(z, b, log_q, dtype):
    """Stable centered variance under frozen q; no student softmax or second-moment subtraction."""
    assert z.is_cuda and z.is_contiguous() and b.is_contiguous() and log_q.is_contiguous()
    n, v = z.shape
    parts = triton.cdiv(v, 4096)
    partial = torch.empty((n, parts), device=z.device, dtype=torch.float32)
    mu = torch.empty(n, device=z.device, dtype=torch.float32)
    dz = torch.empty_like(z, dtype=dtype)
    _bpo_partial_mean[(n, parts)](z, b, log_q, partial, v, parts, 4096)
    _bpo_mean[(n,)](partial, mu, parts, triton.next_power_of_2(parts))
    _bpo_gradient[(n, parts)](z, b, log_q, mu, dz, partial, v, parts, 4096)
    return partial.sum(-1), dz


@triton.jit
def _target_weights(
    LT,
    LQ,
    OUT,
    V: tl.constexpr,
    TS: tl.constexpr,
    K: tl.constexpr,
    ETA: tl.constexpr,
    LOG_ETA: tl.constexpr,
    LOG_ONE_MINUS_ETA: tl.constexpr,
    B: tl.constexpr,
):
    row, part = tl.program_id(0), tl.program_id(1)
    col = part * B + tl.arange(0, B)
    t = tl.load(LT + row * TS + col, col < V, 0)
    q = tl.load(LQ + row * V + col, col < V, 0)
    if K == 0:
        b = q + ETA * (t - q)
    else:
        x = K * (t - q)
        # Clamp the unselected expm1 branch: both branches execute on GPU.
        near = libdevice.log1p(ETA * libdevice.expm1(tl.minimum(tl.maximum(x, -0.5), 0.5)))
        a, c = x + LOG_ETA, LOG_ONE_MINUS_ETA
        far = tl.maximum(a, c) + libdevice.log1p(tl.exp(-tl.abs(a - c)))
        b = q + tl.where(tl.abs(x) < 0.5, near, far) / K
    tl.store(OUT + row * V + col, b, col < V)


def fused_log_weights(log_teacher, log_anchor, *, kappa, eta):
    """Frozen FP32 target; teacher normalization must already include extra rows."""
    import math

    if eta == 1:
        return log_teacher.detach().contiguous()
    n, v = log_anchor.shape
    out = torch.empty_like(log_anchor)
    _target_weights[(n, triton.cdiv(v, 1024))](
        log_teacher,
        log_anchor,
        out,
        v,
        log_teacher.stride(0),
        kappa,
        eta,
        math.log(eta),
        math.log1p(-eta),
        1024,
    )
    return out


@triton.jit
def _mirror_partials(
    Z,
    BIAS,
    LQ,
    LT,
    MP,
    SP,
    GAP,
    MB,
    SB,
    GQ,
    GT,
    QG,
    BREF,
    V: tl.constexpr,
    TS: tl.constexpr,
    P: tl.constexpr,
    BPO: tl.constexpr,
    DIAG: tl.constexpr,
    B: tl.constexpr,
):
    row, part = tl.program_id(0), tl.program_id(1)
    col = part * B + tl.arange(0, B)
    valid = col < V
    ix = row * V + col
    z = tl.load(Z + ix, valid, 0).to(tl.float32)
    b = tl.load(BIAS + ix, valid, 0).to(tl.float32)
    j = row * P + part
    masked_z = tl.where(valid, z, -float("inf"))
    mp = tl.max(masked_z, 0)
    ep = tl.where(valid, tl.exp(z - mp), 0.0)
    tl.store(MP + j, mp)
    tl.store(SP + j, tl.sum(ep, 0))
    if BPO:
        tl.store(GAP + j, tl.sum(ep * (z - b), 0))
    else:
        # Form differences before subtraction: never materialize z-b, whose
        # large common log-normalizer can dominate near-target gradients.
        pivot = tl.argmax(masked_z, 0)
        bref = tl.load(BIAS + row * V + part * B + pivot)
        centered = (z - mp) - (b - bref)
        tl.store(BREF + j, bref)
        tl.store(GAP + j, tl.sum(ep * centered, 0))
    if DIAG:
        q = tl.load(LQ + ix, valid, 0)
        t = tl.load(LT + row * TS + col, valid, 0)
        mb = tl.max(tl.where(valid, b, -float("inf")), 0)
        tl.store(MB + j, mb)
        tl.store(SB + j, tl.sum(tl.where(valid, tl.exp(b - mb), 0.0), 0))
        if BPO:
            tl.store(GQ + j, tl.sum(ep * (z - q), 0))
            tl.store(GT + j, tl.sum(ep * (z - t), 0))
            eq = tl.where(valid, tl.exp(q), 0.0)
            tl.store(QG + j, tl.sum(eq * (z - b), 0))
        else:
            tl.store(GQ + j, tl.sum(ep * ((z - mp) - q), 0))
            tl.store(GT + j, tl.sum(ep * ((z - mp) - t), 0))


@triton.jit
def _mirror_reduce(
    MP,
    SP,
    GAP,
    MB,
    SB,
    GQ,
    GT,
    QG,
    BREF,
    LZ,
    MU,
    LOSS,
    DIAGNOSTICS,
    MAXZ,
    BZERO,
    P: tl.constexpr,
    BPO: tl.constexpr,
    DIAG: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0)
    j = tl.arange(0, B)
    ix = row * P + j
    mp = tl.load(MP + ix, j < P, -float("inf"))
    maximum = tl.max(mp, 0)
    ap = tl.exp(mp - maximum)
    partial_sp = tl.load(SP + ix, j < P, 0)
    sp = tl.sum(ap * partial_sp, 0)
    log_sp = tl.log(sp)
    gap = tl.load(GAP + ix, j < P, 0)
    if BPO:
        mu = tl.sum(ap * gap, 0) / sp
        lzp = maximum + log_sp
        raw = mu - lzp
        tl.store(LZ + row, lzp)
        tl.store(MU + row, tl.sum(tl.load(QG + ix, j < P, 0), 0))
    else:
        pivot_part = tl.argmax(mp, 0)
        bref = tl.load(BREF + ix, j < P, 0)
        bzero = tl.sum(tl.where(j == pivot_part, bref, 0), 0)
        # Translate tile moments into the global-max frame. Mask padded
        # offsets before multiplication to avoid 0 * -inf.
        offset = tl.where(j < P, (mp - maximum) - (bref - bzero), 0.0)
        mu = tl.sum(ap * (gap + partial_sp * offset), 0) / sp
        raw = (mu - log_sp) - bzero
        tl.store(LZ + row, log_sp)
        tl.store(MU + row, mu)
        tl.store(MAXZ + row, maximum)
        tl.store(BZERO + row, bzero)
        tl.store(LOSS + row, raw)
    if DIAG:
        mb = tl.load(MB + ix, j < P, -float("inf"))
        maximum_b = tl.max(mb, 0)
        sb = tl.sum(tl.exp(mb - maximum_b) * tl.load(SB + ix, j < P, 0), 0)
        lzb = maximum_b + tl.log(sb)
        gq = tl.load(GQ + ix, j < P, 0)
        gt = tl.load(GT + ix, j < P, 0)
        if BPO:
            qkl = tl.sum(ap * gq, 0) / sp - lzp
            tkl = tl.sum(ap * gt, 0) / sp - lzp
        else:
            offset = tl.where(j < P, mp - maximum, 0.0)
            qkl = tl.sum(ap * (gq + partial_sp * offset), 0) / sp - log_sp
            tkl = tl.sum(ap * (gt + partial_sp * offset), 0) / sp - log_sp
        tl.store(DIAGNOSTICS + row * 4, raw + lzb)
        tl.store(DIAGNOSTICS + row * 4 + 1, lzb)
        tl.store(DIAGNOSTICS + row * 4 + 2, qkl)
        tl.store(DIAGNOSTICS + row * 4 + 3, tkl)


@triton.jit
def _mirror_gradient(Z, BIAS, LZ, MU, MAXZ, BZERO, DZ, V: tl.constexpr, B: tl.constexpr):
    row, part = tl.program_id(0), tl.program_id(1)
    col = part * B + tl.arange(0, B)
    ix = row * V + col
    z = tl.load(Z + ix, col < V, 0).to(tl.float32)
    b = tl.load(BIAS + ix, col < V, 0).to(tl.float32)
    # LZ is relative to the row maximum, not the absolute log-normalizer.
    relative_z = z - tl.load(MAXZ + row)
    centered = relative_z - (b - tl.load(BZERO + row))
    p = tl.exp(relative_z - tl.load(LZ + row))
    tl.store(DZ + ix, p * (centered - tl.load(MU + row)), col < V)


def fused_mirror_rows(z, b, log_q, log_teacher, dtype, *, objective, diagnostics):
    """Raw MDPO/BPO and optional KL diagnostics, sharing row statistics.

    No dense p/dZ diagnostic temporaries. Order: target KL, log Z_m, anchor KL,
    teacher KL. q/T/b are frozen and all returned values are unscaled by eta.
    """
    n, v = z.shape
    diag = torch.empty((n, 4 if diagnostics else 0), device=z.device, dtype=torch.float32)
    if objective == "bpo" and not diagnostics:
        rows, dz = fused_bpo_rows(z, b, log_q, dtype)
        return rows, dz, diag
    parts = triton.cdiv(v, 4096)
    # Unused optional statistics share a dummy pointer; constexpr removes access.
    count = 8 if objective == "bpo" and diagnostics else 7 if diagnostics else 3
    buffers = [torch.empty((n, parts), device=z.device, dtype=torch.float32) for _ in range(count)]
    buffers += [buffers[0]] * (8 - count)
    lz, mu, rows = [torch.empty(n, device=z.device, dtype=torch.float32) for _ in range(3)]
    dz = torch.empty_like(z, dtype=dtype)
    bpo = objective == "bpo"
    bref = buffers[0] if bpo else torch.empty_like(buffers[0])
    maxz, bzero = (mu, mu) if bpo else (torch.empty_like(mu), torch.empty_like(mu))
    _mirror_partials[(n, parts)](
        z,
        b,
        log_q,
        log_teacher,
        *buffers,
        bref,
        v,
        log_teacher.stride(0) if diagnostics else v,
        parts,
        bpo,
        diagnostics,
        4096,
    )
    _mirror_reduce[(n,)](
        *buffers,
        bref,
        lz,
        mu,
        rows,
        diag,
        maxz,
        bzero,
        parts,
        bpo,
        diagnostics,
        triton.next_power_of_2(parts),
    )
    if bpo:
        partial = buffers[0]
        _bpo_gradient[(n, parts)](z, b, log_q, mu, dz, partial, v, parts, 4096)
        rows = partial.sum(-1)
    else:
        _mirror_gradient[(n, parts)](z, b, lz, mu, maxz, bzero, dz, v, 4096)
    return rows, dz, diag
