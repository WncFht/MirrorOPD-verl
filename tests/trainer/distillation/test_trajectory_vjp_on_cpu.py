# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent dense references for weighted local and strict-future VJPs."""

import itertools

import pytest
import torch

from verl.trainer.distillation.full_vocab import full_vocab_objective_result
from verl.trainer.distillation.output import FullVocabOutput
from verl.trainer.distillation.reduction import PackedReduction
from verl.workers.config.distillation_spec import DistillationLossSettings, FutureCreditConfig

MODES = ["none", "sum", "mean", "discounted_sum", "discounted_mean", "discounted_sequence"]


def test_near_certain_sampled_action_retains_tiny_score_gradient():
    from verl.trainer.distillation.trajectory import action_score

    logits = torch.tensor([[0.0, -30.0, -35.0]], dtype=torch.float64)
    value, gradient = action_score(logits, torch.tensor([0]), None)
    tail = logits[:, 1:].exp()
    expected = torch.cat((tail.sum(-1, keepdim=True), -tail), -1) / (1 + tail.sum(-1, keepdim=True))
    torch.testing.assert_close(gradient, expected, atol=1e-27, rtol=1e-13)
    torch.testing.assert_close(value, -torch.log1p(tail.sum(-1)), atol=1e-27, rtol=1e-13)


def spec(eta=1.0, kappa=0.5, objective="mdpo"):
    return DistillationLossSettings(
        names=("mirror",), payload_kind="hidden", objective=objective, eta=eta, kappa=kappa, diagnostics=True
    )


def independent_reference(hs, ws, ht, wt, hq, wq, settings, layout, stop_ids):
    def lp(h, w):
        z = h @ w.T
        if stop_ids:
            keep = [i for i in range(len(w)) if i not in stop_ids]
            z = torch.cat((z[:, keep], z[:, stop_ids].logsumexp(-1, keepdim=True)), -1)
        return z.log_softmax(-1)

    student, teacher = lp(hs, ws), lp(ht.detach(), wt.detach())
    teacher = teacher[:, : student.shape[-1]]
    if settings.teacher_only_kl:
        target = teacher
    else:
        anchor = lp(hq.detach(), wq.detach())
        if settings.kappa == 0:
            target = (1 - settings.eta) * anchor + settings.eta * teacher
        else:
            target = (
                torch.logaddexp(
                    anchor * settings.kappa + student.new_tensor(1 - settings.eta).log(),
                    teacher * settings.kappa + student.new_tensor(settings.eta).log(),
                )
                / settings.kappa
            )
    costs = (student.exp() * (student - target)).sum(-1)
    sampled = layout.actions
    if stop_ids:
        keep = [i for i in range(len(ws)) if i not in stop_ids]
        sampled = torch.tensor(
            [len(keep) if int(a) in stop_ids else keep.index(int(a)) for a in sampled], device=sampled.device
        )
    sampled_lp = student.gather(1, sampled[:, None]).squeeze(1)
    result, offset = costs.sum() * 0, 0
    for n in layout.counts:
        for t in range(offset, offset + n):
            future = costs.new_zeros(())
            weights = layout.future_config.gamma ** (layout.times[t + 1 : offset + n] - layout.times[t]).double()
            if layout.future_config.mode != "none":
                future = (costs[t + 1 : offset + n].detach() * weights).sum()
                if layout.future_config.mode in ("mean", "discounted_mean") and len(weights):
                    future /= weights.sum()
            local_weight = 1 / n if layout.mode == "seq-mean-token-mean" else 1
            if layout.future_config.mode == "discounted_sequence":
                local_weight *= layout.future_config.gamma ** layout.times[t].double()
            result = result + local_weight * (costs[t] + future * sampled_lp[t])
        offset += n
    return result, costs.detach()


@pytest.mark.parametrize("backend,chunk", [("dense", 3), ("precompute", 1), ("precompute", 3), ("precompute", 20)])
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("eta,kappa,stop_ids", [(1.0, 0.5, []), (0.2, 0.0, []), (0.2, 0.5, []), (0.2, 1.0, [1, 4])])
def test_surrogate_value_hidden_head_vjp_and_metrics(backend, chunk, mode, eta, kappa, stop_ids):
    torch.manual_seed(91)
    hs, ws, ht, wt, hq, wq = [
        torch.randn(*shape, dtype=torch.float64, requires_grad=True)
        for shape in [(7, 3), (9, 3), (7, 4), (9, 4), (7, 3), (9, 3)]
    ]
    settings = spec(eta, kappa)
    future = FutureCreditConfig(mode=mode, gamma=0.87 if mode.startswith("discounted") else 1.0)
    layout = PackedReduction(
        (4, 0, 1, 2),
        torch.tensor([0, 1, 3, 6, 0, 2, 5]),
        torch.tensor([0, 2, 1, 4, 3, 1, 4]),
        "seq-mean-token-mean",
        future,
    )
    expected, costs = independent_reference(hs, ws, ht, wt, hq, wq, settings, layout, stop_ids)
    result = full_vocab_objective_result(
        hs,
        ws,
        ht,
        wt,
        hq=hq,
        wq=wq,
        spec=settings,
        reduction=layout,
        backend=backend,
        token_chunk=chunk,
        termination_token_ids=stop_ids,
    )
    torch.testing.assert_close(result.loss_sum, expected, atol=2e-11, rtol=2e-11)
    actual_grad = torch.autograd.grad(result.loss_sum * 0.37, (hs, ws, ht, wt, hq, wq), allow_unused=True)
    expected_grad = torch.autograd.grad(expected * 0.37, (hs, ws))
    for a, e in zip(actual_grad[:2], expected_grad, strict=True):
        torch.testing.assert_close(a, e, atol=2e-11, rtol=2e-11)
    assert actual_grad[2:] == (None,) * 4
    assert not result.token_objective.requires_grad
    torch.testing.assert_close(result.local_objective_sum, costs.sum(), atol=2e-11, rtol=2e-11)
    output = FullVocabOutput.from_result(result, settings, 7, termination=bool(stop_ids))
    key = "distillation/full_kl" if settings.teacher_only_kl else "distillation/mdpo/objective"
    # The observation's numerator remains an unweighted local sum, never the surrogate.
    observation = output.observations[key]
    torch.testing.assert_close(observation.state[0], costs.sum() / output.divisor)
    if mode != "none":
        from torch.utils._pytree import tree_flatten

        assert [x for x in tree_flatten(result)[0] if isinstance(x, torch.Tensor) and x.requires_grad] == [
            result.loss_sum
        ]
        assert result.future_coefficients[[3, 4, 6]].eq(0).all()
        local = output.observations["distillation/local_surrogate"].state[0]
        score = output.observations["distillation/future/score_surrogate"].state[0]
        total = output.observations["distillation/training_surrogate"].state[0]
        torch.testing.assert_close(local + score, total)


@pytest.mark.parametrize("eta,kappa", [(1.0, 0.5), (0.2, 0.0), (0.2, 1.0)])
@pytest.mark.parametrize("mode,gamma", [("sum", 1.0), ("discounted_sequence", 0.6)])
def test_exact_enumeration_matches_expected_trajectory_cost_gradient(eta, kappa, mode, gamma):
    """Sum all binary paths; includes state-dependent MDPO normalization costs."""
    torch.manual_seed(13)
    theta = torch.randn(7, 2, dtype=torch.float64, requires_grad=True)
    teacher = torch.randn(7, 2, dtype=torch.float64)
    anchor = torch.randn(7, 2, dtype=torch.float64)
    eye = torch.eye(2, dtype=torch.float64)
    settings = spec(eta, kappa)
    exact, estimate = theta.sum() * 0, theta.sum() * 0
    for path in itertools.product(range(2), repeat=3):
        states = [0, 1 + path[0], 3 + 2 * path[0] + path[1]]
        student = theta[states].log_softmax(-1)
        probability = student[range(3), path].sum().exp()
        layout = PackedReduction(
            (3,), torch.arange(3), torch.tensor(path), future=FutureCreditConfig(mode=mode, gamma=gamma)
        )
        result = full_vocab_objective_result(
            theta[states],
            eye,
            teacher[states],
            eye,
            hq=anchor[states],
            wq=eye,
            spec=settings,
            reduction=layout,
            backend="precompute",
            token_chunk=2,
        )
        # Rebuild live costs for the exact expectation, including path-probability derivatives.
        local_layout = PackedReduction((3,), torch.arange(3), torch.tensor(path))
        live_cost, _ = independent_reference(
            theta[states], eye, teacher[states], eye, anchor[states], eye, settings, local_layout, []
        )
        if gamma == 1:
            objective = live_cost
        else:
            target = teacher[states].log_softmax(-1)
            q = anchor[states].log_softmax(-1)
            target = (
                target
                if eta == 1
                else (
                    (1 - eta) * q + eta * target
                    if kappa == 0
                    else torch.logaddexp(
                        q * kappa + theta.new_tensor(1 - eta).log(),
                        target * kappa + theta.new_tensor(eta).log(),
                    )
                    / kappa
                )
            )
            live_rows = (student.exp() * (student - target)).sum(-1)
            objective = (live_rows * gamma ** torch.arange(3).double()).sum()
        exact = exact + probability * objective
        estimate = estimate + probability.detach() * result.loss_sum
    torch.testing.assert_close(
        torch.autograd.grad(estimate, theta)[0], torch.autograd.grad(exact, theta)[0], atol=2e-10, rtol=2e-10
    )


@pytest.mark.parametrize("backend", ["dense", "precompute"])
@pytest.mark.parametrize("objective", ["bpo", "forward_kl"])
def test_future_rejects_unsupported_objectives_even_empty(backend, objective):
    settings = spec(eta=0.2 if objective == "bpo" else 1.0, objective=objective)
    layout = PackedReduction(
        (0,), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long), future=FutureCreditConfig(mode="sum")
    )
    with pytest.raises(ValueError, match="Future credit"):
        full_vocab_objective_result(
            torch.empty(0, 2),
            torch.eye(2),
            torch.empty(0, 2),
            torch.eye(2),
            spec=settings,
            hq=torch.empty(0, 2),
            wq=torch.eye(2),
            reduction=layout,
            backend=backend,
        )


@pytest.mark.parametrize("backend", ["dense", "recompute", "cached", "streaming", "precompute"])
@pytest.mark.parametrize("extra", [0, 3])
def test_immediate_reverse_kl_weighting_preserves_full_teacher_normalizer(backend, extra):
    torch.manual_seed(31)
    hs = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    ws = torch.randn(7, 3, dtype=torch.float64, requires_grad=True)
    ht, wt = torch.randn_like(hs), torch.randn(7 + extra, 3, dtype=torch.float64)
    layout = PackedReduction(
        (3, 0, 2), torch.tensor([0, 1, 4, 0, 2]), torch.zeros(5, dtype=torch.long), "seq-mean-token-mean"
    )
    result = full_vocab_objective_result(
        hs, ws, ht, wt, spec=spec(), reduction=layout, backend=backend, token_chunk=2, vocab_chunk=3
    )
    lp, lt = (hs @ ws.T).log_softmax(-1), (ht @ wt.T).log_softmax(-1)[:, :7]
    values = (lp.exp() * (lp - lt)).sum(-1)
    expected = values[:3].mean() + values[3:].mean()
    torch.testing.assert_close(result.loss_sum, expected)
    for a, e in zip(
        torch.autograd.grad(result.loss_sum, (hs, ws)), torch.autograd.grad(expected, (hs, ws)), strict=True
    ):
        torch.testing.assert_close(a, e)
    torch.testing.assert_close(result.local_objective_sum, values.sum())


@pytest.mark.parametrize("backend", ["dense", "recompute", "precompute"])
def test_immediate_forward_kl_weighting(backend):
    torch.manual_seed(17)
    hs = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    ws = torch.randn(7, 3, dtype=torch.float64, requires_grad=True)
    ht, wt = torch.randn_like(hs), torch.randn_like(ws)
    layout = PackedReduction(
        (3, 2), torch.tensor([0, 1, 4, 0, 2]), torch.zeros(5, dtype=torch.long), "seq-mean-token-mean"
    )
    result = full_vocab_objective_result(
        hs, ws, ht, wt, spec=spec(objective="forward_kl"), reduction=layout, backend=backend, token_chunk=2
    )
    lp, lt = (hs @ ws.T).log_softmax(-1), (ht @ wt.T).log_softmax(-1)
    values = (lt.exp() * (lt - lp)).sum(-1)
    expected = values[:3].mean() + values[3:].mean()
    for a, e in zip(
        torch.autograd.grad(result.loss_sum, (hs, ws)), torch.autograd.grad(expected, (hs, ws)), strict=True
    ):
        torch.testing.assert_close(a, e)
    torch.testing.assert_close(result.loss_sum, expected)


@pytest.mark.parametrize("backend", ["dense", "precompute"])
def test_immediate_bpo_weighting(backend):
    torch.manual_seed(17)
    hs = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    ws = torch.randn(7, 3, dtype=torch.float64, requires_grad=True)
    ht, wt, hq, wq = torch.randn_like(hs), torch.randn_like(ws), torch.randn_like(hs), torch.randn_like(ws)
    layout = PackedReduction(
        (3, 2), torch.tensor([0, 1, 4, 0, 2]), torch.zeros(5, dtype=torch.long), "seq-mean-token-mean"
    )
    result = full_vocab_objective_result(
        hs, ws, ht, wt, hq=hq, wq=wq, spec=spec(0.2, 0.5, "bpo"), reduction=layout, backend=backend, token_chunk=2
    )
    lp, lt, lq = [(h @ w.T).log_softmax(-1) for h, w in [(hs, ws), (ht, wt), (hq, wq)]]
    log_target = torch.logaddexp(0.5 * lq + lq.new_tensor(0.8).log(), 0.5 * lt + lt.new_tensor(0.2).log()) / 0.5
    residual = lp - log_target
    centered = residual - (lq.exp() * residual).sum(-1, keepdim=True)
    values = 0.5 * (lq.exp() * centered.square()).sum(-1)
    expected = values[:3].mean() + values[3:].mean()
    torch.testing.assert_close(result.loss_sum, expected)
    for a, e in zip(
        torch.autograd.grad(result.loss_sum, (hs, ws)), torch.autograd.grad(expected, (hs, ws)), strict=True
    ):
        torch.testing.assert_close(a, e)


@pytest.mark.parametrize("extra,stop_ids", [(3, []), (0, [1, 4])])
def test_teacher_future_support_and_raw_eos_diagnostics(extra, stop_ids):
    torch.manual_seed(17)
    hs = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    ws = torch.randn(7, 3, dtype=torch.float64, requires_grad=True)
    ht, wt = torch.randn_like(hs), torch.randn(7 + extra, 3, dtype=torch.float64)
    layout = PackedReduction(
        (3, 2), torch.tensor([0, 1, 4, 0, 2]), torch.tensor([0, 2, 1, 3, 4]), future=FutureCreditConfig(mode="sum")
    )
    result = full_vocab_objective_result(
        hs,
        ws,
        ht,
        wt,
        spec=spec(),
        reduction=layout,
        backend="precompute",
        token_chunk=2,
        termination_token_ids=stop_ids,
    )
    expected, _ = independent_reference(hs, ws, ht, wt, None, None, spec(), layout, stop_ids)
    torch.testing.assert_close(result.loss_sum, expected)
    for a, e in zip(
        torch.autograd.grad(result.loss_sum, (hs, ws)), torch.autograd.grad(expected, (hs, ws)), strict=True
    ):
        torch.testing.assert_close(a, e)
    raw_lp, raw_lt = (hs @ ws.T).log_softmax(-1), (ht @ wt.T).log_softmax(-1)[:, :7]
    raw_kl = (raw_lp.exp() * (raw_lp - raw_lt)).sum(-1)
    torch.testing.assert_close(result.token_objective, raw_kl)


@pytest.mark.parametrize("backend", ["dense", "precompute"])
def test_empty_future_rank_has_finite_zero_value_and_gradients(backend):
    h = torch.empty(0, 2, requires_grad=True)
    w = torch.eye(2, requires_grad=True)
    layout = PackedReduction(
        (0,), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long), future=FutureCreditConfig(mode="mean")
    )
    result = full_vocab_objective_result(h, w, h.detach(), w.detach(), spec=spec(), reduction=layout, backend=backend)
    assert result.loss_sum == 0
    dh, dw = torch.autograd.grad(result.loss_sum, (h, w))
    assert dh.shape == h.shape
    assert dw.eq(0).all()


@pytest.mark.parametrize("counts,times", [((3, 2), [0, 1, 4, 0, 2]), ((1, 0, 1, 1, 1, 1), [0, 0, 0, 0, 0])])
def test_precompute_reuses_student_projection_and_backward_does_not_reproject(monkeypatch, counts, times):
    from verl.trainer.distillation import reverse_vocab, rows

    torch.manual_seed(17)
    hs = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    ws = torch.randn(7, 3, dtype=torch.float64, requires_grad=True)
    ht, wt, hq, wq = torch.randn_like(hs), torch.randn_like(ws), torch.randn_like(hs), torch.randn_like(ws)
    roles = {ws.data_ptr(): "student", wt.data_ptr(): "teacher", wq.data_ptr(): "anchor"}
    calls = {name: 0 for name in roles.values()}
    original = rows.project

    def record(hidden, weight):
        calls[roles[weight.data_ptr()]] += 1
        return original(hidden, weight)

    monkeypatch.setattr(reverse_vocab, "project", record)
    monkeypatch.setattr(rows, "project", record)
    layout = PackedReduction(
        counts, torch.tensor(times), torch.zeros(5, dtype=torch.long), future=FutureCreditConfig(mode="sum")
    )
    result = full_vocab_objective_result(
        hs, ws, ht, wt, hq=hq, wq=wq, spec=spec(0.2), reduction=layout, backend="precompute", token_chunk=2
    )
    assert calls == {"student": 3, "teacher": 3, "anchor": 3}
    result.loss_sum.backward()
    assert calls == {"student": 3, "teacher": 3, "anchor": 3}


@pytest.mark.parametrize("mode", ["none", "sum"])
@pytest.mark.parametrize("eta", [1.0, 0.3])
@pytest.mark.parametrize("backend", ["dense", "precompute"])
@pytest.mark.parametrize(
    "invalid",
    [
        "teacher_short",
        "teacher_long",
        "teacher_hidden_width",
        "student_head_width",
        "teacher_support",
        "token_chunk",
        "vocab_chunk",
    ],
)
def test_actual_inputs_are_validated_before_local_or_future_computation(mode, eta, backend, invalid, monkeypatch):
    from unittest.mock import Mock

    from verl.trainer.distillation import reverse_vocab, rows

    reject_projection = Mock(side_effect=AssertionError("Invalid inputs must fail before projection"))
    monkeypatch.setattr(reverse_vocab, "project", reject_projection)
    monkeypatch.setattr(rows, "project", reject_projection)
    args = dict(
        hs=torch.zeros(2, 3),
        ws=torch.zeros(5, 3),
        ht=torch.zeros(2, 4),
        wt=torch.zeros(5, 4),
        hq=torch.zeros(2, 3),
        wq=torch.zeros(5, 3),
        token_chunk=2,
        vocab_chunk=3,
    )
    if invalid == "teacher_short":
        args["ht"] = torch.zeros(1, 4)
    elif invalid == "teacher_long":
        args["ht"] = torch.zeros(3, 4)
    elif invalid == "teacher_hidden_width":
        args["ht"] = torch.zeros(2, 6)
    elif invalid == "student_head_width":
        args["ws"] = torch.zeros(5, 6)
    elif invalid == "teacher_support":
        args["wt"] = torch.zeros(4, 4)
    else:
        args[invalid] = 0
    layout = PackedReduction(
        (2,), torch.arange(2), torch.zeros(2, dtype=torch.long), future=FutureCreditConfig(mode=mode)
    )
    with pytest.raises(ValueError):
        full_vocab_objective_result(**args, spec=spec(eta), reduction=layout, backend=backend)
    reject_projection.assert_not_called()


@pytest.mark.parametrize("mode", ["none", "sum"])
@pytest.mark.parametrize("rows", [0, 2])
@pytest.mark.parametrize("invalid", ["short", "long", "head_width", "support", "missing"])
def test_real_anchor_validation_also_applies_to_empty_ranks(mode, rows, invalid):
    hs, ws = torch.zeros(rows, 3, requires_grad=True), torch.zeros(5, 3, requires_grad=True)
    ht, wt = torch.zeros(rows, 4), torch.zeros(5, 4)
    hq, wq = torch.zeros(rows, 3), torch.zeros(5, 3)
    if invalid == "short":
        if rows == 0:
            hq = torch.zeros(1, 3)
        else:
            hq = torch.zeros(rows - 1, 3)
    elif invalid == "long":
        hq = torch.zeros(rows + 1, 3)
    elif invalid == "head_width":
        wq = torch.zeros(5, 6)
    elif invalid == "support":
        wq = torch.zeros(4, 3)
    else:
        hq = None
    layout = PackedReduction(
        (rows,), torch.arange(rows), torch.zeros(rows, dtype=torch.long), future=FutureCreditConfig(mode=mode)
    )
    with pytest.raises(ValueError, match="anchor|shapes"):
        full_vocab_objective_result(hs, ws, ht, wt, hq=hq, wq=wq, spec=spec(0.3), reduction=layout)
