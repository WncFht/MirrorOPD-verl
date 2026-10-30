# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Inference-only dense/grouped GEMM with BF16 weights and FP32 output."""

import torch
import triton
import triton.language as tl


@triton.jit
def _grouped_fp32(
    X, W, Offsets, Y, N: tl.constexpr, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr
):
    expert, n_tile = tl.program_id(0), tl.program_id(1)
    start = tl.load(Offsets + expert - 1, mask=expert > 0, other=0)
    end = tl.load(Offsets + expert)
    ns = n_tile * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    for tile in range(tl.cdiv(end - start, BM)):
        rows = start + tile * BM + tl.arange(0, BM)
        accum = tl.full((BM, BN), 0, tl.float32)
        for k in range(tl.cdiv(K, BK)):
            cols = k * BK + ks
            x = tl.load(X + rows[:, None] * K + cols[None, :], (rows[:, None] < end) & (cols[None, :] < K), 0)
            w = tl.load(
                W + expert * N * K + ns[None, :] * K + cols[:, None],
                (ns[None, :] < N) & (cols[:, None] < K),
                0,
            )
            accum = tl.dot(x, w.to(x.dtype), accum, input_precision="tf32x3")
        tl.store(Y + rows[:, None] * N + ns[None, :], accum, (rows[:, None] < end) & (ns[None, :] < N))


def grouped_fp32(inputs, weights, offsets):
    if inputs.dtype not in (torch.bfloat16, torch.float32) or weights.dtype != torch.bfloat16:
        raise ValueError("Grouped inference GEMM requires BF16 weights and BF16/FP32 inputs")
    if not inputs.is_contiguous() or not weights.is_contiguous():
        raise ValueError("Grouped inference GEMM requires contiguous operands")
    experts, output_width, input_width = weights.shape
    output = torch.empty((inputs.shape[0], output_width), device=inputs.device, dtype=torch.float32)
    _grouped_fp32[(experts, triton.cdiv(output_width, 64))](
        inputs, weights, offsets, output, output_width, input_width, 16, 64, 64, num_warps=4
    )
    return output


@triton.jit(do_not_specialize=["M"])
def _linear_fp32(X, W, Y, M, N: tl.constexpr, K: tl.constexpr):
    rows = tl.program_id(0) * 16 + tl.arange(0, 16)
    ns = tl.program_id(1) * 64 + tl.arange(0, 64)
    ks = tl.arange(0, 32)
    accum = tl.full((16, 64), 0, tl.float32)
    for k in range(tl.cdiv(K, 32)):
        cols = k * 32 + ks
        x = tl.load(X + rows[:, None] * K + cols[None, :], (rows[:, None] < M) & (cols[None, :] < K), 0)
        w = tl.load(W + ns[None, :] * K + cols[:, None], (ns[None, :] < N) & (cols[:, None] < K), 0)
        accum = tl.dot(x.to(tl.float32), w.to(tl.float32), accum, input_precision="tf32x3")
    tl.store(Y + rows[:, None] * N + ns[None, :], accum, (rows[:, None] < M) & (ns[None, :] < N))


def linear_fp32(inputs, weights):
    inputs = inputs.reshape(-1, inputs.shape[-1]).contiguous()
    if weights.dtype not in (torch.bfloat16, torch.float32) or not weights.is_contiguous():
        raise ValueError("Inference linear requires contiguous BF16/FP32 weights")
    rows, inner = inputs.shape
    output_width = weights.shape[0]
    output = torch.empty((rows, output_width), device=inputs.device, dtype=torch.float32)
    _linear_fp32[(triton.cdiv(rows, 16), triton.cdiv(output_width, 64))](
        inputs, weights, output, rows, output_width, inner, num_warps=4
    )
    return output


@triton.jit
def _rms_fp32(X, W, Y, N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    values = tl.load(X + row * N + offsets, offsets < N, 0).to(tl.float32)
    scale = tl.rsqrt(tl.sum(values * values, axis=0) / N + EPS)
    weights = tl.load(W + offsets, offsets < N, 0).to(tl.float32)
    tl.store(Y + row * N + offsets, values * scale * weights, offsets < N)


def rms_fp32(inputs, weights, epsilon):
    shape = inputs.shape
    inputs = inputs.reshape(-1, shape[-1]).contiguous()
    output = torch.empty_like(inputs, dtype=torch.float32)
    _rms_fp32[(inputs.shape[0],)](inputs, weights, output, shape[-1], epsilon, triton.next_power_of_2(shape[-1]))
    return output.reshape(shape)
