# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Scalar precomputed VJP storage, independent of objective and traversal."""

import torch

from verl.trainer.distillation.profiling import full_kl_range


def add_product(accumulator, left, right):
    """Accumulate directly into FP32 storage instead of allocating a full head gradient per tile."""
    if left.is_cuda and left.dtype in (torch.float16, torch.bfloat16) and accumulator.dtype == torch.float32:
        torch.addmm(accumulator, left, right, out_dtype=torch.float32, out=accumulator)
    else:
        accumulator.addmm_(left.to(accumulator.dtype), right.to(accumulator.dtype))


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


def precompute_rows(
    hs,
    ws,
    ht,
    wt,
    token_chunk,
    prepare,
    *,
    hq=None,
    wq=None,
    row_weights=None,
    diagnostic_count=0,
    reduction=None,
    fused=False,
    actions=None,
):
    """Execute a typed block preparer through the common precomputed VJP."""
    from functools import partial

    program = partial(
        run_blocks,
        chunk=token_chunk,
        prepare=prepare,
        row_weights=row_weights,
        diagnostic_count=diagnostic_count,
        reduction=reduction,
        fused=fused,
        actions=actions,
        accumulate=True,
    )
    return precompute(hs, ws, ht, wt, hq, wq, program=program)


def run_blocks(
    hs,
    ws,
    ht,
    wt,
    hq,
    wq,
    *,
    chunk,
    prepare,
    diagnostic_count=0,
    row_weights=None,
    reduction=None,
    fused=False,
    actions=None,
    accumulate=False,
):
    """One bounded-workspace loop; future adds credit before gradient accumulation."""
    from verl.trainer.distillation.reduction import suffix_credit
    from verl.trainer.distillation.trajectory import action_score

    dtype = torch.float64 if hs.dtype == torch.float64 else torch.float32
    raw = hs.new_empty(len(hs), dtype=dtype)
    costs = torch.empty_like(raw)
    info = hs.new_empty((len(hs), diagnostic_count), dtype=dtype)
    future_enabled = reduction is not None and reduction.future_config.mode != "none"
    coefficients = torch.empty_like(costs) if future_enabled else costs.new_empty(0)
    score_sum = costs.new_zeros(())
    total = hs.sum() * 0 + ws.sum() * 0 if not accumulate else costs.new_zeros(())
    dh, dw = (torch.zeros_like(hs, dtype=dtype), torch.zeros_like(ws, dtype=dtype)) if accumulate else (None, None)
    weights = reduction.row_weights(dtype) if reduction is not None else row_weights
    blocks = (
        reduction.blocks(chunk)
        if future_enabled
        else ((start, min(start + chunk, len(hs))) for start in range(0, len(hs), chunk))
    )
    carry = None
    for start, stop in blocks:
        block = prepare(hs[start:stop], ws, ht[start:stop], wt, None if hq is None else hq[start:stop], wq)
        values, dz, metric, diag, logits = block
        local_weight = 1 if weights is None else weights[start:stop]
        if future_enabled:
            config = reduction.future_config
            future, carry = suffix_credit(
                values, reduction.times[start:stop], reduction.segments[start:stop], config, carry=carry
            )
            logits, statistics = logits
            if config.mode == "discounted_sequence":
                local_weight = local_weight * config.gamma ** reduction.times[start:stop].to(dtype)
            if accumulate and fused:
                from verl.trainer.distillation.full_vocab_triton import finish_future

                sampled_lp, dz = finish_future(
                    logits, dz, reduction.actions[start:stop], future, local_weight, actions, statistics
                )
                dz = dz.float() if hs.dtype == torch.float16 else dz.to(hs.dtype)
            else:
                sampled_lp, score = action_score(logits, reduction.actions[start:stop], actions)
            score_term = future * sampled_lp
            total = total + ((values + score_term) * local_weight).sum()
            coefficients[start:stop] = future
            score_sum = score_sum + (score_term.detach() * local_weight).sum()
            if accumulate and not fused:
                dz = (dz + future[:, None] * score) * (
                    local_weight if isinstance(local_weight, int) else local_weight[:, None]
                )
                dz = dz.float() if hs.dtype == torch.float16 else dz.to(hs.dtype)
        else:
            total = (
                total + (values * local_weight).sum() if not accumulate else total.add_((values * local_weight).sum())
            )
            if accumulate and weights is not None:
                dz = (dz * local_weight[:, None]).to(dz.dtype)
        raw[start:stop], costs[start:stop], info[start:stop] = metric.detach(), values.detach(), diag.detach()
        if accumulate:
            with full_kl_range("grad_hidden"):
                add_product(dh[start:stop], dz, ws)
            with full_kl_range("grad_head"):
                add_product(dw, dz.T, hs[start:stop])
        del block, dz, logits
    return (total, raw, costs, info, coefficients, score_sum), dh, dw
