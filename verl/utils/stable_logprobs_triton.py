# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Tiled selected log-probability forward/VJP with a reconstructed target entry."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _softplus(x):
    return tl.maximum(x, 0.0) + libdevice.log1p(tl.exp(-tl.abs(x)))


@triton.jit
def _partial(X, Y, PART, V: tl.constexpr, P: tl.constexpr, B: tl.constexpr):
    row, part = tl.program_id(0), tl.program_id(1)
    c = part * B + tl.arange(0, B)
    label = tl.load(Y + row)
    z = tl.load(X + row * V + c, c < V, -float("inf")).to(tl.float32)
    z = tl.where(c == label, -float("inf"), z)
    maximum = tl.max(z, 0)
    safe = tl.where(maximum == -float("inf"), 0.0, maximum)
    total = tl.sum(tl.exp(z - safe), 0)
    ix = (row * P + part) * 2
    tl.store(PART + ix, maximum)
    tl.store(PART + ix + 1, total)


@triton.jit
def _reduce(X, Y, PART, STATS, OUT, V: tl.constexpr, P: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    j = tl.arange(0, B)
    ix = (row * P + j) * 2
    m = tl.load(PART + ix, j < P, -float("inf"))
    maximum = tl.max(m, 0)
    maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
    total = tl.sum(tl.exp(m - maximum) * tl.load(PART + ix + 1, j < P, 0), 0)
    log_total = tl.log(tl.maximum(total, 1.1754943508222875e-38))
    label = tl.load(Y + row)
    target = tl.load(X + row * V + tl.maximum(label, 0))
    margin = (maximum - target) + log_total
    bad_target = (target == float("inf")) | ((target == -float("inf")) & (total == 0))
    margin = tl.where(total == 0, -float("inf"), margin)
    margin = tl.where(bad_target, float("nan"), margin)
    loss = _softplus(margin)
    log_mass = -_softplus(-margin)
    tl.store(STATS + row * 3, maximum)
    tl.store(STATS + row * 3 + 1, log_total)
    tl.store(STATS + row * 3 + 2, log_mass)
    tl.store(OUT + row, tl.where(label == -100, 0.0, -loss))


@triton.jit
def _backward(X, Y, STATS, UP, DX, SUM, V: tl.constexpr, P: tl.constexpr, B: tl.constexpr):
    row, part = tl.program_id(0), tl.program_id(1)
    c = part * B + tl.arange(0, B)
    z = tl.load(X + row * V + c, c < V, -float("inf")).to(tl.float32)
    maximum = tl.load(STATS + row * 3)
    log_total = tl.load(STATS + row * 3 + 1)
    log_mass = tl.load(STATS + row * 3 + 2)
    label = tl.load(Y + row)
    upstream = tl.load(UP + row).to(tl.float32)
    exponent = ((z - maximum) - log_total) + log_mass
    # Mask the target BEFORE exp: a very confident target may overflow here,
    # while all non-target probabilities are bounded by one.
    exponent = tl.where((c != label) & (label != -100), exponent, -float("inf"))
    gradient = -tl.exp(exponent) * upstream
    tl.store(DX + row * V + c, gradient, c < V)
    tl.store(SUM + row * P + part, tl.sum(gradient, 0))


@triton.jit
def _target(Y, SUM, DX, V: tl.constexpr, P: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    j = tl.arange(0, B)
    label = tl.load(Y + row)
    gradient = -tl.sum(tl.load(SUM + row * P + j, j < P, 0), 0)
    tl.store(DX + row * V + tl.maximum(label, 0), gradient, label >= 0)


class _FusedLogProb(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, inplace):
        n, v = logits.shape
        parts = triton.cdiv(v, 4096)
        partial = torch.empty((n, parts, 2), device=logits.device, dtype=torch.float32)
        stats = torch.empty((n, 3), device=logits.device, dtype=torch.float32)
        value = torch.empty(n, device=logits.device, dtype=torch.float32)
        _partial[(n, parts)](logits, labels, partial, v, parts, 4096, enable_fp_fusion=False)
        _reduce[(n,)](
            logits, labels, partial, stats, value, v, parts, triton.next_power_of_2(parts), enable_fp_fusion=False
        )
        ctx.save_for_backward(logits, labels, stats)
        ctx.inplace = inplace and not logits.is_leaf
        return value

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, upstream):
        logits, labels, stats = ctx.saved_tensors
        n, v = logits.shape
        parts = triton.cdiv(v, 4096)
        gradient = logits if ctx.inplace else torch.empty_like(logits)
        sums = torch.empty((n, parts), device=logits.device, dtype=torch.float32)
        _backward[(n, parts)](
            logits, labels, stats, upstream.contiguous(), gradient, sums, v, parts, 4096, enable_fp_fusion=False
        )
        _target[(n,)](labels, sums, gradient, v, parts, triton.next_power_of_2(parts))
        return gradient, None, None


def selected_logprobs_cuda(logits, labels, *, inplace_backward):
    return _FusedLogProb.apply(logits, labels, inplace_backward)
