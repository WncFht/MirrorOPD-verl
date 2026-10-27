# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Linear projection with FP32 outputs from low-precision CUDA operands."""

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
    if hidden.is_cuda and hidden.dtype in (torch.float16, torch.bfloat16):
        if torch.is_grad_enabled() and (hidden.requires_grad or weight.requires_grad):
            return ProjectionFP32.apply(hidden, weight)
        return torch.mm(hidden, weight.T, out_dtype=torch.float32)
    return hidden @ weight.T
