# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Scalar precomputed VJP storage, independent of objective and traversal."""

import torch

from verl.trainer.distillation.profiling import full_kl_range


class PrecomputedVJP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hs, ws, ht, wt, hq, wq, program):
        outputs, dh, dw = program(hs, ws, ht, wt, hq, wq)
        ctx.hdtype, ctx.wdtype = hs.dtype, ws.dtype
        ctx.save_for_backward(
            dh if hs.dtype == torch.float16 else dh.to(hs.dtype),
            dw if ws.dtype == torch.float16 else dw.to(ws.dtype),
        )
        ctx.mark_non_differentiable(*outputs[1:])
        return outputs

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, upstream, *unused):
        dh, dw = ctx.saved_tensors
        with full_kl_range("backward_apply"):
            return (dh * upstream).to(ctx.hdtype), (dw * upstream).to(ctx.wdtype), None, None, None, None, None


def precompute(hs, ws, ht, wt, hq=None, wq=None, *, program):
    """Pass trainable H/W explicitly so wrapped-head autograd hooks see them."""
    return PrecomputedVJP.apply(hs, ws, ht, wt, hq, wq, program)
