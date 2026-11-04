# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Authoritative full-vocabulary objective dispatch; normalization belongs to the loss layer."""

from functools import partial

import torch

from verl.trainer.distillation.distribution import CompiledDistribution, DistributionSpec
from verl.trainer.distillation.execution import validate_fused_backend
from verl.trainer.distillation.results import FullVocabObjectiveResult
from verl.trainer.distillation.reverse_vocab import full_reverse_kl_result, validate_reverse_kl_inputs
from verl.trainer.distillation.rows import _MIRROR_DIAGNOSTICS, prepare_conditional, prepare_objective
from verl.trainer.distillation.termination import TerminationActionMap
from verl.trainer.distillation.validation import validate_mirror_inputs
from verl.trainer.distillation.vjp import precompute_rows, run_blocks


def full_vocab_objective_result(
    hs,
    ws,
    ht,
    wt,
    *,
    spec,
    hq=None,
    wq=None,
    backend="precompute",
    token_chunk=512,
    vocab_chunk=8192,
    termination_token_ids=(),
    forward_teacher_support="equal",
    distribution=None,
    reduction=None,
):
    """Dispatch teacher KL, MDPO and BPO; MDPO EOS diagnostics use coarsened actions."""
    from verl.trainer.distillation.execution import resolve_execution

    future = "none" if reduction is None else reduction.future_config.mode
    objective = "reverse_kl" if spec.teacher_only_kl else spec.objective
    declared_space = (
        distribution.spec
        if isinstance(distribution, CompiledDistribution)
        else DistributionSpec.parse(distribution or {})
    )
    plan = resolve_execution(
        backend,
        objective=objective,
        future=future,
        distribution=declared_space.mode,
        termination=bool(termination_token_ids),
        aggregate="token-mean" if reduction is None else reduction.mode,
        reference=True,
    )
    validate_fused_backend(backend, hs)
    if reduction is not None and sum(reduction.counts) != len(hs):
        raise ValueError("Packed reduction must match hidden rows")
    if plan.future:
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
        actions = (
            TerminationActionMap.create(len(ws), termination_token_ids, hs.device) if termination_token_ids else None
        )
        return execute_objective(
            hs, ws, ht, wt, hq, wq, spec=spec, plan=plan, reduction=reduction, token_chunk=token_chunk, actions=actions
        )
    row_weights = (
        None
        if reduction is None
        else reduction.row_weights(torch.float64 if hs.dtype == torch.float64 else torch.float32)
    )
    if distribution is not None:
        space = (
            distribution
            if isinstance(distribution, CompiledDistribution)
            else DistributionSpec.parse(distribution).compile(len(ws), len(wt), termination_token_ids)
        )
        if space.spec.mode != "native":
            if not spec.teacher_only_kl:
                raise ValueError("Conditional distributions require teacher-only reverse KL")
            if backend in ("precompute", "precompute_fused"):
                names = ("student_excluded_mass", "teacher_excluded_mass")
                actions = (
                    TerminationActionMap.create(len(space.student_ids), space.termination_ids, hs.device)
                    if space.termination_ids
                    else None
                )
                fused = backend == "precompute_fused"
                prepare = partial(prepare_conditional, space=space, actions=actions, fused=fused)
                total, raw, objective_rows, info, _, _ = precompute_rows(
                    hs,
                    ws,
                    ht.detach(),
                    wt.detach(),
                    token_chunk,
                    prepare,
                    diagnostic_count=len(names),
                    row_weights=row_weights,
                )
                return FullVocabObjectiveResult(
                    total, raw, {k: info[:, i] for i, k in enumerate(names)}, objective_rows, objective_rows.sum()
                )
            result = full_vocab_objective_result(
                hs,
                space.select(ws, "student"),
                ht,
                space.select(wt, "teacher"),
                spec=spec,
                backend=backend,
                token_chunk=token_chunk,
                vocab_chunk=vocab_chunk,
                termination_token_ids=space.termination_ids,
                reduction=reduction,
            )
            diagnostics = dict(result.token_diagnostics)
            for role, h, w in (("student", hs, ws), ("teacher", ht, wt)):
                diagnostics[role + "_excluded_mass"] = space.excluded_mass(
                    h, w, role, token_chunk=token_chunk, vocab_chunk=vocab_chunk
                )
            return result._replace(token_diagnostics=diagnostics)
    if spec.teacher_only_forward_kl:
        if termination_token_ids:
            raise ValueError("forward KL does not support termination equivalence")
        from verl.trainer.distillation.forward_vocab import full_forward_kl_result

        return full_forward_kl_result(
            hs,
            ws,
            ht,
            wt,
            backend=backend,
            token_chunk=token_chunk,
            teacher_support=forward_teacher_support,
            row_weights=row_weights,
        )
    if spec.teacher_only_kl:
        result = full_reverse_kl_result(
            hs,
            ws,
            ht,
            wt,
            backend=backend,
            token_chunk=token_chunk,
            vocab_chunk=vocab_chunk,
            termination_token_ids=termination_token_ids,
            row_weights=row_weights,
        )
        return FullVocabObjectiveResult(
            result.loss_sum,
            result.token_kl,
            {},
            result.objective_rows if result.objective_rows is not None else result.token_kl,
            result.loss_sum.detach() if row_weights is None else result.objective_rows.sum(),
        )
    validate_mirror_inputs(
        hs,
        ws,
        ht,
        wt,
        spec=spec,
        hq=hq,
        wq=wq,
        backend=backend,
        token_chunk=token_chunk,
        vocab_chunk=vocab_chunk,
        termination_token_ids=termination_token_ids,
    )
    ht, wt, hq, wq = (x.detach() for x in (ht, wt, hq, wq))
    if len(hs) == 0:
        zero = hs.sum() * 0 + ws.sum() * 0
        rows = hs.new_empty(0)
        return FullVocabObjectiveResult(zero, rows, {}, rows, zero.detach())
    actions = TerminationActionMap.create(len(ws), termination_token_ids, hs.device) if termination_token_ids else None
    return execute_objective(
        hs,
        ws,
        ht,
        wt,
        hq,
        wq,
        spec=spec,
        plan=plan,
        reduction=reduction,
        token_chunk=token_chunk,
        actions=actions,
        row_weights=row_weights,
    )


def execute_objective(hs, ws, ht, wt, hq, wq, *, spec, plan, reduction, token_chunk, actions, row_weights=None):
    """Choose autograd storage once; all local/future blocks use the same loop."""
    prepare = partial(prepare_objective, spec=spec, fused=plan.fused, actions=actions, future=plan.future)
    diagnostic_count = 4 if spec.diagnostics and not spec.teacher_only_kl else 0
    ht, wt = ht.detach(), wt.detach()
    hq, wq = (None if x is None else x.detach() for x in (hq, wq))
    if plan.strategy == "dense":
        outputs, _, _ = run_blocks(
            hs,
            ws,
            ht,
            wt,
            hq,
            wq,
            chunk=token_chunk,
            prepare=prepare,
            diagnostic_count=diagnostic_count,
            reduction=reduction,
            row_weights=row_weights,
            actions=actions,
        )
    else:
        outputs = precompute_rows(
            hs,
            ws,
            ht,
            wt,
            token_chunk,
            prepare,
            hq=hq,
            wq=wq,
            diagnostic_count=diagnostic_count,
            reduction=reduction,
            row_weights=row_weights,
            fused=plan.fused,
            actions=actions,
        )
    total, raw, costs, info, coefficients, score_sum = outputs
    diagnostics = {k: info[:, i] for i, k in enumerate(_MIRROR_DIAGNOSTICS)} if diagnostic_count else {}
    return FullVocabObjectiveResult(
        total,
        raw,
        diagnostics,
        costs,
        costs.sum(),
        coefficients if plan.future else None,
        score_sum if plan.future else None,
    )
