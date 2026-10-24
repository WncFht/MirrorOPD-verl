# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Optional BF16 CUDA reverse-KL reduction with bounded logits storage.

Three Triton passes replace full-vocabulary softmax intermediates. cuBLAS still
does projections and gradient products. The public dispatcher falls back to
ordinary precompute for other devices/dtypes.
"""

import torch
import triton
import triton.language as tl

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
