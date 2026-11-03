# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Single-projection, reverse-block trajectory surrogate for full-vocab MDPO.

Future coefficients and cost diagnostics are detached. Only the scalar training
numerator is differentiable, so FSDP's wrapped-head pytree hooks see its VJP.
"""

import torch

from verl.trainer.distillation.full_vocab import (
    _MIRROR_DIAGNOSTICS,
    FullVocabObjectiveResult,
    _mirror_rows,
    add_product,
    project,
    termination_equivalent_kl_rows,
    validate_mirror_inputs,
    validate_reverse_kl_inputs,
)
from verl.trainer.distillation.numerics import kl_rows, log_softmax, support_correction
from verl.trainer.distillation.reduction import suffix_credit
from verl.trainer.distillation.termination import TerminationActionMap


def action_score(logits, sampled, actions):
    """Sampled action log-probability and score VJP, without a dense one-hot."""
    lp = log_softmax(logits)
    value = lp.gather(1, sampled[:, None]).squeeze(1)
    gradient = -lp.exp()
    if actions is None:
        gradient.scatter_add_(1, sampled[:, None], torch.ones_like(value[:, None]))
    else:
        # Differentiate log P(stop class) directly in native coordinates. Only
        # the tiny stop group uses FP64, not a second full-vocabulary workspace.
        is_stop = torch.isin(sampled, actions.stop)
        stop_lp = lp.index_select(1, actions.stop).double()
        value = torch.where(is_stop, stop_lp.logsumexp(-1).to(lp.dtype), value)
        gradient.scatter_add_(1, sampled[:, None], (~is_stop[:, None]).to(gradient.dtype))
        conditional = stop_lp.softmax(-1).to(gradient.dtype) * is_stop[:, None]
        gradient.index_add_(1, actions.stop, conditional)
    # Preserve a tiny 1-p at a near-certain sampled action, as the local VJP does.
    pivot = logits.argmax(-1, keepdim=True)
    other = gradient.scatter(-1, pivot, 0.0)
    gradient = other.scatter(-1, pivot, -other.sum(-1, keepdim=True))
    return value, gradient


def trajectory_rows(hs, ws, ht, wt, hq, wq, spec, fused, actions):
    """Retain the current student projection for the score term; reuse stable VJPs."""
    z = project(hs, ws)
    if not spec.teacher_only_kl:
        values, dz, info = _mirror_rows(hs, ws, ht, wt, hq, wq, spec, fused=fused, actions=actions, student_logits=z)
        return values, dz, values, info, z
    if actions is not None:
        values, dz, raw = termination_equivalent_kl_rows(hs, ws, ht, wt, actions=actions, fused=fused, student_logits=z)
    else:
        teacher = project(ht, wt)
        if fused:
            from verl.trainer.distillation.full_vocab_triton import fused_rows

            values, dz = fused_rows(z, teacher[:, : len(ws)].contiguous(), torch.float32)
            values = values + support_correction(teacher, len(ws))
        else:
            # Includes the frozen full teacher normalizer in future costs too.
            values, dz = kl_rows(z, teacher)
        raw = values
    return values, dz, raw, z.new_empty((len(hs), 0)), z


def trajectory_blocks(hs, ws, ht, wt, hq, wq, spec, reduction, chunk, fused, actions, *, precompute):
    dtype = torch.float64 if hs.dtype == torch.float64 else torch.float32
    costs, raw = hs.new_empty(len(hs), dtype=dtype), hs.new_empty(len(hs), dtype=dtype)
    coefficients = torch.empty_like(costs)
    score_sum = costs.new_zeros(())
    info = hs.new_empty((len(hs), 4 if spec.diagnostics and not spec.teacher_only_kl else 0), dtype=dtype)
    weights = reduction.row_weights(dtype)
    config = reduction.future_config
    total = hs.sum() * 0 + ws.sum() * 0
    dh, dw = (torch.zeros_like(hs, dtype=dtype), torch.zeros_like(ws, dtype=dtype)) if precompute else (None, None)
    carry = None
    for start, stop, reset in reduction.blocks(chunk):
        values, dz, metric, diag, z = trajectory_rows(
            hs[start:stop], ws, ht[start:stop], wt, None if hq is None else hq[start:stop], wq, spec, fused, actions
        )
        future, carry = suffix_credit(values, reduction.times[start:stop], config, carry=None if reset else carry)
        sampled_lp, score = action_score(z, reduction.actions[start:stop], actions)
        local_weight = 1 if weights is None else weights[start:stop]
        if config.mode == "discounted_sequence":
            local_weight = local_weight * config.gamma ** reduction.times[start:stop].to(dtype)
        total = total + ((values + future * sampled_lp) * local_weight).sum()
        coefficients[start:stop] = future
        score_sum = score_sum + (future * sampled_lp.detach() * local_weight).sum()
        costs[start:stop], raw[start:stop], info[start:stop] = values.detach(), metric.detach(), diag.detach()
        if precompute:
            dz = (dz + future[:, None] * score) * (
                local_weight if isinstance(local_weight, int) else local_weight[:, None]
            )
            dz = dz.float() if hs.dtype == torch.float16 else dz.to(hs.dtype)
            add_product(dh[start:stop], dz, ws)
            add_product(dw, dz.T, hs[start:stop])
    return total, costs, raw, info, coefficients, score_sum, dh, dw


class TrajectoryVJP(torch.autograd.Function):
    """Cache only dH/dW and O(tokens) observations; release each vocabulary block."""

    @staticmethod
    def forward(ctx, hs, ws, ht, wt, hq, wq, spec, reduction, chunk, fused, actions):
        total, costs, raw, info, coefficients, score_sum, dh, dw = trajectory_blocks(
            hs, ws, ht, wt, hq, wq, spec, reduction, chunk, fused, actions, precompute=True
        )
        ctx.hdtype, ctx.wdtype = hs.dtype, ws.dtype
        ctx.save_for_backward(
            dh if hs.dtype == torch.float16 else dh.to(hs.dtype), dw if ws.dtype == torch.float16 else dw.to(ws.dtype)
        )
        ctx.mark_non_differentiable(costs, raw, info, coefficients, score_sum)
        return total, costs, raw, info, coefficients, score_sum

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, upstream, *unused):
        dh, dw = ctx.saved_tensors
        return ((dh * upstream).to(ctx.hdtype), (dw * upstream).to(ctx.wdtype)) + (None,) * 9


def trajectory_objective_result(
    hs, ws, ht, wt, *, spec, hq, wq, backend, token_chunk, vocab_chunk, termination_token_ids, distribution, reduction
):
    """Validate opt-in capabilities, preserving the requested backend without fallback."""
    from verl.trainer.distillation.distribution import CompiledDistribution, DistributionSpec

    if spec.payload_kind != "hidden" or spec.objective != "mdpo":
        raise ValueError("Future credit supports full-vocabulary reverse KL and MDPO only")
    if backend not in ("dense", "precompute", "precompute_fused"):
        raise ValueError("Future credit requires dense/precompute/precompute_fused")
    space = (
        distribution.spec
        if isinstance(distribution, CompiledDistribution)
        else DistributionSpec.parse(distribution or {})
    )
    if space.mode != "native":
        raise ValueError("Future credit does not support common-conditioned distributions")
    if sum(reduction.counts) != len(hs):
        raise ValueError("Trajectory layout must match hidden rows")
    validation = dict(
        backend=backend,
        token_chunk=token_chunk,
        vocab_chunk=vocab_chunk,
        termination_token_ids=termination_token_ids,
    )
    if spec.teacher_only_kl:
        validate_reverse_kl_inputs(hs, ws, ht, wt, **validation)
    else:
        validate_mirror_inputs(hs, ws, ht, wt, spec=spec, hq=hq, wq=wq, **validation)
    ht, wt = ht.detach(), wt.detach()
    hq, wq = (None if x is None else x.detach() for x in (hq, wq))
    actions = TerminationActionMap.create(len(ws), termination_token_ids, hs.device) if termination_token_ids else None
    arguments = (hs, ws, ht, wt, hq, wq, spec, reduction, token_chunk, backend == "precompute_fused", actions)
    if backend == "dense":
        total, costs, raw, info, coefficients, score_sum, _, _ = trajectory_blocks(*arguments, precompute=False)
    else:
        total, costs, raw, info, coefficients, score_sum = TrajectoryVJP.apply(*arguments)
    diagnostics = {k: info[:, i] for i, k in enumerate(_MIRROR_DIAGNOSTICS)} if info.shape[1] else {}
    return FullVocabObjectiveResult(total, raw, diagnostics, costs, costs.sum(), coefficients, score_sum)
