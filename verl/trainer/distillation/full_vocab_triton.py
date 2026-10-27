# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Tiled stable row reductions shared by full KL, MDPO and BPO.

Logit differences are formed before centering. Small KL values use exponential
remainders; the pivot VJP is reconstructed before its final low-precision cast.
Projection and dH/dW accumulation remain in the common precompute engine.
"""

import math

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from verl.trainer.distillation.full_vocab import project
from verl.trainer.distillation.profiling import full_kl_range


@triton.jit
def _expm1mx(x):
    s = tl.minimum(tl.maximum(x, -0.125), 0.125)
    # Through x^12; both value and first derivative retain the near-zero term.
    p = 1.0 / 479001600
    p = 1.0 / 39916800 + s * p
    p = 1.0 / 3628800 + s * p
    p = 1.0 / 362880 + s * p
    p = 1.0 / 40320 + s * p
    p = 1.0 / 5040 + s * p
    p = 1.0 / 720 + s * p
    p = 1.0 / 120 + s * p
    p = 1.0 / 24 + s * p
    p = 1.0 / 6 + s * p
    p = 0.5 + s * p
    safe = tl.minimum(tl.maximum(x, -80.0), 80.0)
    return tl.where(tl.abs(x) < 0.125, s * s * p, libdevice.expm1(safe) - safe)


@triton.jit
def _log1pmx(x):
    s = tl.minimum(tl.maximum(x, -0.0625), 0.0625)
    p = -1.0 / 14
    for n in tl.static_range(13, 1, -1):
        sign = 1.0 if n % 2 == 1 else -1.0
        p = sign / n + s * p
    safe = tl.maximum(x, -0.5)
    return tl.where(tl.abs(x) < 0.0625, s * s * p, libdevice.log1p(safe) - safe)


@triton.jit
def _weighted_partials(Z, R, PART, V: tl.constexpr, P: tl.constexpr, B: tl.constexpr):
    row, part = tl.program_id(0), tl.program_id(1)
    c = part * B + tl.arange(0, B)
    z = tl.load(Z + row * V + c, c < V, -float("inf")).to(tl.float32)
    r = tl.load(R + row * V + c, c < V, 0).to(tl.float32)
    pivot = tl.argmax(z, 0)
    maximum = tl.max(z, 0)
    ref = tl.sum(tl.where(tl.arange(0, B) == pivot, r, 0), 0)
    w = tl.exp(z - maximum)
    tail = tl.sum(tl.where(tl.arange(0, B) == pivot, 0.0, w), 0)
    moment = tl.sum(w * (r - ref), 0)
    ix = (row * P + part) * 4
    tl.store(PART + ix, maximum)
    tl.store(PART + ix + 1, tail)
    tl.store(PART + ix + 2, moment)
    tl.store(PART + ix + 3, ref)


@triton.jit
def _weighted_reduce(Z, PART, STATS, PIVOT, V: tl.constexpr, P: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    j = tl.arange(0, B)
    ix = (row * P + j) * 4
    mx = tl.load(PART + ix, j < P, -float("inf"))
    tail = tl.load(PART + ix + 1, j < P, 0)
    moment = tl.load(PART + ix + 2, j < P, 0)
    ref = tl.load(PART + ix + 3, j < P, 0)
    maximum = tl.max(mx, 0)
    first = tl.argmax(mx, 0)
    r0 = tl.sum(tl.where(j == first, ref, 0), 0)
    weight = tl.exp(mx - maximum)
    others = tl.sum(weight * (tail + tl.where(j == first, 0.0, 1.0)), 0)
    mu = tl.sum(weight * (moment + (1 + tail) * (ref - r0)), 0) / (1 + others)
    tl.store(STATS + row * 4, maximum)
    tl.store(STATS + row * 4 + 1, libdevice.log1p(others))
    tl.store(STATS + row * 4 + 2, mu)
    tl.store(STATS + row * 4 + 3, r0)
    # Select the first maximum within its tile without storing dense indices.
    c = first * 4096 + tl.arange(0, 4096)
    z = tl.load(Z + row * V + c, c < V, -float("inf"))
    tl.store(PIVOT + row, first * 4096 + tl.argmax(z, 0))


@triton.jit
def _weighted_apply(
    Z,
    R,
    STATS,
    PIVOT,
    DZ,
    PART,
    V: tl.constexpr,
    P: tl.constexpr,
    VARIANCE: tl.constexpr,
    GRAD: tl.constexpr,
    B: tl.constexpr,
):
    row, part = tl.program_id(0), tl.program_id(1)
    c = part * B + tl.arange(0, B)
    valid = c < V
    z = tl.load(Z + row * V + c, valid, 0).to(tl.float32)
    r = tl.load(R + row * V + c, valid, 0).to(tl.float32)
    maximum = tl.load(STATS + row * 4)
    ln = tl.load(STATS + row * 4 + 1)
    mu = tl.load(STATS + row * 4 + 2)
    ref = tl.load(STATS + row * 4 + 3)
    lp = tl.where(valid, (z - maximum) - ln, -float("inf"))
    w = tl.exp(lp)
    centered = tl.where(valid, (r - ref) - mu, 0.0)
    pivot = tl.load(PIVOT + row)
    g = tl.where(c == pivot, 0.0, w * centered)
    if GRAD:
        tl.store(DZ + row * V + c, g, valid)
    ix = (row * P + part) * 7
    tl.store(PART + ix, tl.sum(g, 0))
    tl.store(PART + ix + 1, tl.sum(w, 0))
    if VARIANCE:
        tl.store(PART + ix + 2, 0.5 * tl.sum(w * centered * centered, 0))
    else:
        x = -centered
        exponent = lp + x
        rem = tl.where(tl.abs(x) < 0.125, w * _expm1mx(x), tl.exp(tl.minimum(exponent, 40.0)) - w * (1 + x))
        mx = tl.max(exponent, 0)
        tl.store(PART + ix + 2, tl.sum(w * x, 0))
        tl.store(PART + ix + 3, tl.sum(rem, 0))
        tl.store(PART + ix + 4, mx)
        tl.store(PART + ix + 5, tl.sum(tl.exp(exponent - mx), 0))


@triton.jit
def _weighted_finish(
    PART,
    PIVOT,
    DZ,
    VALUE,
    V: tl.constexpr,
    P: tl.constexpr,
    VARIANCE: tl.constexpr,
    GRAD: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0)
    j = tl.arange(0, B)
    ix = (row * P + j) * 7
    mass = tl.sum(tl.load(PART + ix + 1, j < P, 0), 0)
    if GRAD:
        pivot = tl.load(PIVOT + row)
        tl.store(DZ + row * V + pivot, -tl.sum(tl.load(PART + ix, j < P, 0), 0))
    if VARIANCE:
        value = tl.sum(tl.load(PART + ix + 2, j < P, 0), 0) / mass
    else:
        a = tl.sum(tl.load(PART + ix + 2, j < P, 0), 0) / mass
        b = tl.sum(tl.load(PART + ix + 3, j < P, 0), 0) / mass
        mx = tl.load(PART + ix + 4, j < P, -float("inf"))
        maximum = tl.max(mx, 0)
        total = tl.sum(tl.exp(mx - maximum) * tl.load(PART + ix + 5, j < P, 0), 0)
        safe_a = tl.minimum(tl.maximum(a, -0.5), 0.5)
        safe_b = tl.minimum(tl.maximum(b, -0.5), 0.5)
        close = libdevice.log1p(safe_b / (1 + safe_a)) + _log1pmx(safe_a)
        far = maximum + tl.log(total) - tl.log(mass) - a
        value = tl.where((maximum < 40) & (tl.abs(a) < 0.5) & (tl.abs(b) < 0.5), close, far)
    tl.store(VALUE + row, value)


def weighted_rows(logits, residual, dtype, *, variance=False, gradient=True):
    n, v = logits.shape
    parts = triton.cdiv(v, 4096)
    partial = torch.empty((n, parts, 4), device=logits.device, dtype=torch.float32)
    stats = torch.empty((n, 4), device=logits.device, dtype=torch.float32)
    pivot = torch.empty(n, dtype=torch.int32, device=logits.device)
    values = torch.empty(n, device=logits.device, dtype=torch.float32)
    dz = torch.empty_like(logits, dtype=dtype) if gradient else values
    tail = torch.empty((n, parts, 7), device=logits.device, dtype=torch.float32)
    _weighted_partials[(n, parts)](logits, residual, partial, v, parts, 4096, enable_fp_fusion=False)
    _weighted_reduce[(n,)](
        logits, partial, stats, pivot, v, parts, triton.next_power_of_2(parts), enable_fp_fusion=False
    )
    _weighted_apply[(n, parts)](
        logits, residual, stats, pivot, dz, tail, v, parts, variance, gradient, 4096, enable_fp_fusion=False
    )
    _weighted_finish[(n,)](
        tail, pivot, dz, values, v, parts, variance, gradient, triton.next_power_of_2(parts), enable_fp_fusion=False
    )
    return values, dz if gradient else None, stats


@triton.jit
def _difference(
    Z, T, A, PIVOT, OUT, V: tl.constexpr, TS: tl.constexpr, ETA: tl.constexpr, ANCHOR: tl.constexpr, B: tl.constexpr
):
    row, part = tl.program_id(0), tl.program_id(1)
    c = part * B + tl.arange(0, B)
    p = tl.load(PIVOT + row)
    z0 = tl.load(Z + row * V + p).to(tl.float64)
    t0 = tl.load(T + row * TS + p).to(tl.float64)
    z = tl.load(Z + row * V + c, c < V, 0).to(tl.float64)
    t = tl.load(T + row * TS + c, c < V, 0).to(tl.float64)
    if ANCHOR:
        a0 = tl.load(A + row * V + p).to(tl.float64)
        a = tl.load(A + row * V + c, c < V, 0).to(tl.float64)
        result = ((z - z0) - (a - a0)) - ETA * ((t - t0) - (a - a0))
    else:
        result = (z - z0) - (t - t0)
    tl.store(OUT + row * V + c, result, c < V)


def difference(z, teacher, pivot=None, *, anchor=None, eta=1.0):
    pivot = z.argmax(-1) if pivot is None else pivot.reshape(-1)
    r = torch.empty_like(z)
    _difference[(len(z), triton.cdiv(z.shape[1], 1024))](
        z,
        teacher,
        z if anchor is None else anchor,
        pivot,
        r,
        z.shape[1],
        teacher.stride(0),
        eta,
        anchor is not None,
        1024,
        enable_fp_fusion=False,
    )
    return r


def fused_rows(zp, zt, dtype):
    value, gradient, _ = weighted_rows(zp, difference(zp, zt), dtype)
    return value, gradient


def fused_kl_rows(hs, ws, ht, wt):
    if not hs.is_cuda or hs.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("fused precompute requires FP16/BF16/FP32 CUDA")
    with full_kl_range("student_projection"):
        zp = project(hs, ws)
    with full_kl_range("teacher_projection"):
        zt = project(ht, wt)
    with full_kl_range("fused_kl_and_grad_logits"):
        return fused_rows(zp, zt, torch.float32 if hs.dtype == torch.float16 else hs.dtype)


def fused_bpo_rows(z, b, log_q, dtype):
    r = difference(z, b, log_q.argmax(-1))
    value, gradient, _ = weighted_rows(log_q, r, dtype, variance=True)
    return value, gradient


def fused_mirror_rows(z, b, log_q, log_teacher, dtype, *, objective, diagnostics):
    """Compatibility for callers that already encoded b; cannot recover lost increments."""
    if objective == "bpo":
        value, gradient = fused_bpo_rows(z, b, log_q, dtype)
    else:
        value, gradient = fused_rows(z, b, dtype)
        value = value - b.logsumexp(-1)
    info = z.new_empty((len(z), 0))
    if diagnostics:
        target, _, _ = weighted_rows(z, difference(z, b), dtype, gradient=False)
        anchor, _, _ = weighted_rows(z, difference(z, log_q), dtype, gradient=False)
        teacher, _, _ = weighted_rows(z, difference(z, log_teacher), dtype, gradient=False)
        info = torch.stack([target, b.logsumexp(-1), anchor, teacher - log_teacher.logsumexp(-1)], -1)
    return value, gradient, info


@triton.jit
def _power(x, eta, K: tl.constexpr):
    if K == 0:
        return eta * x
    y = K * x
    near = libdevice.log1p(eta * libdevice.expm1(tl.minimum(tl.maximum(y, -0.5), 0.5)))
    a = tl.log(eta) + y
    b = libdevice.log1p(-eta)
    far = tl.maximum(a, b) + libdevice.log1p(tl.exp(-tl.abs(a - b)))
    return tl.where(tl.abs(y) < 0.5, near, far) / K


@triton.jit
def _exp_wide(x):
    if x.dtype == tl.float64:
        return libdevice.exp(x)
    return tl.exp(x)


@triton.jit
def _logadd(a, b):
    return tl.maximum(a, b) + libdevice.log1p(_exp_wide(-tl.abs(a - b)))


@triton.jit
def _softplus_wide(x):
    return tl.maximum(x, 0.0) + libdevice.log1p(_exp_wide(-tl.abs(x)))


@triton.jit
def _power_static(x, K: tl.constexpr, ETA: tl.constexpr, LE: tl.constexpr, LC: tl.constexpr):
    if K == 0 or ETA == 1:
        return ETA * x
    y = K * x
    near = libdevice.log1p(ETA * libdevice.expm1(tl.minimum(tl.maximum(y, -0.5), 0.5)))
    return tl.where(tl.abs(y) < 0.5, near, _logadd(y + LE, tl.full(y.shape, LC, y.dtype))) / K


@triton.jit
def _power_remainder(x, K: tl.constexpr, ETA: tl.constexpr, LE: tl.constexpr, LC: tl.constexpr):
    if K == 0 or ETA == 1:
        return tl.full(x.shape, 0.0, x.dtype)
    if ETA <= 0.5:
        e, y, le, lc = ETA, K * x, LE, LC
    else:
        e, y, le, lc = 1 - ETA, -K * x, LC, LE
    safe = tl.minimum(tl.maximum(y, -0.5), 0.5)
    a = e * libdevice.expm1(safe)
    near = e * _expm1mx(safe) + _log1pmx(a)
    far = _logadd(y + le, tl.full(y.shape, lc, y.dtype)) - e * y
    return tl.where(tl.abs(y) < 0.5, near, far) / K


@triton.jit
def _factored_h(y, a, l0, l1):
    near = libdevice.log1p(a * libdevice.expm1(-tl.minimum(y, 0.5)))
    return tl.where(y < 0.5, near, _logadd(l0, l1 - y))


@triton.jit
def _power_gap(x, K: tl.constexpr, ETA: tl.constexpr, LE: tl.constexpr, LC: tl.constexpr, COEFF: tl.constexpr):
    if K == 1 or ETA == 1:
        return tl.full(x.shape, 0.0, tl.float32)
    small = tl.minimum(tl.maximum(x if ETA <= 0.5 else -x, -0.5), 0.5)
    polynomial = tl.full(x.shape, COEFF[14], tl.float32)
    for j in tl.static_range(13, -1, -1):
        polynomial = COEFF[j] + small * polynomial
    polynomial = small * small * polynomial
    wide = x.to(tl.float64) if K > 0.99 else x
    far = _power_remainder(wide, 1.0, ETA, LE, LC) - _power_remainder(wide, K, ETA, LE, LC)
    if K > 0:
        absx = tl.abs(wide)
        a = tl.where(wide >= 0, tl.full(wide.shape, 1 - ETA, wide.dtype), tl.full(wide.shape, ETA, wide.dtype))
        l0 = tl.where(wide >= 0, tl.full(wide.shape, LE, wide.dtype), tl.full(wide.shape, LC, wide.dtype))
        l1 = tl.where(wide >= 0, tl.full(wide.shape, LC, wide.dtype), tl.full(wide.shape, LE, wide.dtype))
        factored = _factored_h(absx, a, l0, l1) - _factored_h(K * absx, a, l0, l1) / K
        far = tl.where(absx > 40, factored, far)
        if K > 0.99:
            mid = (1 + K) / 2
            odds = LE - LC + mid * wide
            lp, lcp = -_softplus_wide(-odds), -_softplus_wide(odds)
            divergence = _exp_wide(lp) * (lp - LE) + _exp_wide(lcp) * (lcp - LC)
            divided = (1 - K) / (mid * mid) * divergence
            far = tl.where((1 - K) * absx < 1e-4, divided, far)
    return tl.where(tl.abs(x) < 0.5, polynomial, far.to(tl.float32))


@triton.jit
def _split_residual(
    Z,
    Q,
    T,
    PIVOT,
    QSTATS,
    QT,
    R,
    NORM,
    V: tl.constexpr,
    TS: tl.constexpr,
    P: tl.constexpr,
    K: tl.constexpr,
    ETA: tl.constexpr,
    LE: tl.constexpr,
    LC: tl.constexpr,
    COEFF: tl.constexpr,
    NORMALIZER: tl.constexpr,
    B: tl.constexpr,
):
    row, part = tl.program_id(0), tl.program_id(1)
    c = part * B + tl.arange(0, B)
    a = tl.load(PIVOT + row)
    q0, t0, z0 = tl.load(Q + row * V + a), tl.load(T + row * TS + a), tl.load(Z + row * V + a)
    q = tl.load(Q + row * V + c, c < V, 0)
    t = tl.load(T + row * TS + c, c < V, 0)
    z = tl.load(Z + row * V + c, c < V, 0)
    dq = q.to(tl.float64) - q0.to(tl.float64)
    dt = t.to(tl.float64) - t0.to(tl.float64)
    dz = z.to(tl.float64) - z0.to(tl.float64)
    delta = dt - dq
    if ETA == 1:
        residual = dz - dt
    elif K == 0:
        residual = (dz - dq) - ETA * delta
    else:
        ra = tl.load(QSTATS + row * 4 + 2) - tl.load(QT + row)
        odds = LE - LC + K * ra
        choose_teacher = odds > 0
        small_weight = tl.sigmoid(-tl.abs(odds))
        increment = _power(tl.where(choose_teacher, -delta, delta).to(tl.float32), small_weight, K)
        residual = (dz - tl.where(choose_teacher, dt, dq)) - increment
    tl.store(R + row * V + c, residual, c < V)
    if NORMALIZER:
        ra = tl.load(QSTATS + row * 4 + 2) - tl.load(QT + row)
        log_ratio = delta.to(tl.float32) + ra
        lp = (q - tl.load(QSTATS + row * 4)) - tl.load(QSTATS + row * 4 + 1)
        gap = _power_gap(log_ratio, K, ETA, LE, LC, COEFF)
        log_mix = lp + _power_static(log_ratio, 1.0, ETA, LE, LC)
        deficit = tl.sum(tl.where(c < V, tl.exp(log_mix) * (-libdevice.expm1(-gap)), 0.0), 0)
        log_target = tl.where(c < V, lp + _power_static(log_ratio, K, ETA, LE, LC), -float("inf"))
        maximum = tl.max(log_target, 0)
        log_sum = maximum + tl.log(tl.sum(tl.exp(log_target - maximum), 0))
        tl.store(NORM + (row * P + part) * 2, deficit)
        tl.store(NORM + (row * P + part) * 2 + 1, log_sum)


def fused_mirror_logits(z, anchor, teacher, dtype, *, spec):
    """Production split-target path: raw logits in, stable value/VJP out."""
    from verl.trainer.distillation.numerics import power_gap_coefficients, support_correction

    n, v = z.shape
    pivot = anchor.argmax(-1)
    normalizer = spec.objective == "mdpo" or spec.diagnostics
    need_qt = spec.eta != 1 and (normalizer or spec.kappa != 0)
    correction = support_correction(teacher, v) if (need_qt or normalizer) else z.new_zeros(n)
    if need_qt:
        qt, _, stats = weighted_rows(anchor, difference(anchor, teacher, pivot), dtype, gradient=False)
        qt = qt + correction
    else:
        qt, stats = z.new_empty(n), z.new_empty((n, 4))
    residual = torch.empty_like(z)
    parts = triton.cdiv(v, 1024)
    partial = z.new_empty((n, parts, 2))
    need_deficit = normalizer and spec.kappa != 1 and spec.eta != 1
    _split_residual[(n, parts)](
        z,
        anchor,
        teacher,
        pivot,
        stats,
        qt,
        residual,
        partial,
        v,
        teacher.stride(0),
        parts,
        spec.kappa,
        spec.eta,
        math.log(spec.eta),
        math.log1p(-spec.eta) if spec.eta < 1 else 0.0,
        power_gap_coefficients(spec.kappa, spec.eta),
        need_deficit,
        1024,
        enable_fp_fusion=False,
    )
    if normalizer:
        if spec.eta == 1:
            log_z = -correction
        else:
            near_mass = torch.log1p(spec.eta * torch.expm1(-correction.clamp(max=0.5)))
            far_mass = torch.logaddexp(correction.new_tensor(math.log1p(-spec.eta)), math.log(spec.eta) - correction)
            log_mass = torch.where(correction < 0.5, near_mass, far_mass)
            if spec.kappa == 1:
                log_z = log_mass
            else:
                ratio = partial[:, :, 0].sum(-1) / log_mass.exp()
                close = log_mass + torch.log1p(-ratio.clamp(max=0.5))
                log_z = torch.where(ratio < 0.5, close, partial[:, :, 1].logsumexp(-1))
    del partial
    value, gradient, _ = weighted_rows(
        z if spec.objective == "mdpo" else anchor, residual, dtype, variance=spec.objective == "bpo"
    )
    if spec.objective == "mdpo":
        target = value
        value = value - log_z
    info = z.new_empty((n, 0))
    if spec.diagnostics:
        if spec.objective == "bpo":
            target, _, _ = weighted_rows(z, residual, dtype, gradient=False)
        del residual
        aq, _, _ = weighted_rows(z, difference(z, anchor), dtype, gradient=False)
        at, _, _ = weighted_rows(z, difference(z, teacher), dtype, gradient=False)
        info = torch.stack([target, log_z, aq, at + correction], -1)
    return value, gradient, info


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
