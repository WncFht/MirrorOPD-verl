# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Tiled FKL VJP and alignment diagnostics sharing the existing stable reductions."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from verl.trainer.distillation.full_vocab_triton import difference, weighted_rows


@triton.jit
def _apply(Z, T, GAP, SP, SQ, FKL, PIVOT, DZ, PART, V: tl.constexpr, P: tl.constexpr, B: tl.constexpr):
    row, part = tl.program_id(0), tl.program_id(1)
    c = part * B + tl.arange(0, B)
    valid = c < V
    zp = tl.load(Z + row * V + c, valid, 0).to(tl.float32)
    zq = tl.load(T + row * V + c, valid, 0).to(tl.float32)
    mp, np = tl.load(SP + row * 4), tl.load(SP + row * 4 + 1)
    mq, nq = tl.load(SQ + row * 4), tl.load(SQ + row * 4 + 1)
    lp, lq = (zp - mp) - np, (zq - mq) - nq
    p, q = tl.where(valid, tl.exp(lp), 0), tl.where(valid, tl.exp(lq), 0)
    gap = tl.load(GAP + row * V + c, valid, 0).to(tl.float64)
    center = tl.load(SQ + row * 4 + 2).to(tl.float64) + tl.load(SQ + row * 4 + 3).to(tl.float64)
    d = (gap + center - tl.load(FKL + row).to(tl.float64)).to(tl.float32)
    g = tl.where(tl.abs(d) < 0.5, q * libdevice.expm1(tl.minimum(tl.maximum(d, -0.5), 0.5)), p - q)
    pivot = tl.load(PIVOT + row)
    g = tl.where(valid & (c != pivot), g, 0.0)
    tl.store(DZ + row * V + c, g, valid)
    maximum = tl.maximum(lp, lq)
    lm = maximum + tl.log(tl.exp(lp - maximum) + tl.exp(lq - maximum)) - 0.6931471805599453
    ix = (row * P + part) * 5
    tl.store(PART + ix, tl.sum(g, 0))
    tl.store(PART + ix + 1, tl.sum(tl.where(valid, -p * lp, 0.0), 0))
    tl.store(PART + ix + 2, tl.sum(tl.where(valid, -q * lq, 0.0), 0))
    tl.store(PART + ix + 3, tl.sum(tl.where(valid, 0.5 * (p * (lp - lm) + q * (lq - lm)), 0.0), 0))
    tl.store(PART + ix + 4, tl.sum(tl.abs(p - q), 0) * 0.5)


@triton.jit
def _finish(PART, PIVOT, DZ, INFO, V: tl.constexpr, P: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, B)
    ix = (row * P + i) * 5
    pivot = tl.load(PIVOT + row)
    tl.store(DZ + row * V + pivot, -tl.sum(tl.load(PART + ix, i < P, 0.0), 0))
    for k in tl.static_range(1, 5):
        tl.store(INFO + row * 4 + k - 1, tl.sum(tl.load(PART + ix + k, i < P, 0.0), 0))


def fused_forward_rows(z, teacher, extra, dtype):
    n, v = z.shape
    gap = difference(z, teacher)
    rkl, _, sp = weighted_rows(z, gap, torch.float32, gradient=False)
    fkl, _, sq = weighted_rows(teacher, -gap, torch.float32, gradient=False)
    parts = triton.cdiv(v, 4096)
    partial = z.new_empty((n, parts, 5))
    info = z.new_empty((n, 4))
    gradient = torch.empty_like(z, dtype=dtype)
    pivot = z.argmax(-1)
    _apply[(n, parts)](z, teacher, gap, sp, sq, fkl, pivot, gradient, partial, v, parts, 4096, enable_fp_fusion=False)
    _finish[(n,)](partial, pivot, gradient, info, v, parts, triton.next_power_of_2(parts), enable_fp_fusion=False)
    correction = torch.zeros_like(fkl)
    if extra is not None:
        log_tail = (extra.double() - sq[:, :1].double()).logsumexp(-1)
        correction = torch.nn.functional.softplus(log_tail - sq[:, 1].double()).float()
    diagnostics = torch.cat([(rkl + correction)[:, None], (-torch.expm1(-correction))[:, None], info], -1)
    return fkl, gradient, diagnostics
