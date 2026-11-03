# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""One engine/loss contract must preserve endpoint, mirror and EOS semantics."""

import math
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from verl.trainer.distillation.full_vocab import full_vocab_objective_result
from verl.trainer.distillation.losses import distillation_loss
from verl.trainer.distillation.output import FullVocabOutput
from verl.utils import tensordict_utils as tu
from verl.workers.config.distillation import DistillationConfig, DistillationLossConfig, FullVocabConfig


@pytest.mark.parametrize(
    "objective,eta,stop",
    [("mdpo", 1, False), ("mdpo", 1, True), ("mdpo", 0.1, False), ("mdpo", 0.1, True), ("bpo", 0.3, False)],
)
def test_production_result_keeps_objective_gradients_and_diagnostic_scaling(objective, eta, stop):
    torch.manual_seed(91)
    hs = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    ws = torch.randn(7, 4, dtype=torch.float64, requires_grad=True)
    ht, wt = torch.randn_like(hs), torch.randn_like(ws)
    hq, wq = torch.randn_like(hs), torch.randn_like(ws)
    loss_config = DistillationLossConfig(
        loss_mode="mirror",
        mirror=dict(objective=objective, kappa=0.5, eta=eta, diagnostics=True),
        full_vocab=FullVocabConfig(backend="precompute", termination_token_ids=[1, 4] if stop else []),
        use_policy_gradient=False,
        use_task_rewards=False,
        loss_max_clamp=None,
        log_prob_min_clamp=None,
    )
    spec = loss_config.loss_settings
    result = full_vocab_objective_result(
        hs, ws, ht, wt, spec=spec, hq=hq, wq=wq, termination_token_ids=[1, 4] if stop else []
    )
    output = FullVocabOutput.from_result(result, spec, 3, termination=stop, bins=torch.tensor([0, 3, 9]))
    data = TensorDict({}, batch_size=[])
    tu.assign_non_tensor(data, dp_size=2, batch_num_tokens=11)
    loss, metrics = distillation_loss(
        SimpleNamespace(loss_agg_mode="token-mean"),
        DistillationConfig(distillation_loss=loss_config),
        {"full_vocab": output},
        data,
    )
    expected = result.loss_sum * 2 / 11 / eta
    torch.testing.assert_close(loss, expected)
    actual = torch.autograd.grad(loss, (hs, ws), retain_graph=True)
    reference = torch.autograd.grad(expected, (hs, ws))
    for a, b in zip(actual, reference, strict=True):
        torch.testing.assert_close(a, b)
    for name, values in result.token_diagnostics.items():
        torch.testing.assert_close(metrics[f"distillation/{name}"].state[0], values.detach().sum().double())
    primary = "full_kl" if eta == 1 else f"{objective}/objective"
    torch.testing.assert_close(
        metrics[f"distillation/{primary}"].state[0],
        (result.token_objective.sum() if stop and eta == 1 else result.loss_sum / eta).detach().double(),
    )


def test_fsdp_pytree_hooks_can_see_the_differentiable_numerator():
    from torch.utils._pytree import tree_flatten

    total = torch.tensor(2.0, requires_grad=True)
    output = FullVocabOutput(total, 0.1, {})
    leaves, _ = tree_flatten({"full_vocab": output})
    assert any(leaf is total for leaf in leaves)
    assert [leaf for leaf in leaves if isinstance(leaf, torch.Tensor) and leaf.requires_grad] == [total]


def test_conditional_coarse_metrics_keep_their_space_and_fsdp_visible_objective():
    from torch.utils._pytree import tree_flatten

    from verl.workers.config.distillation_spec import resolve_distillation_spec

    spec = resolve_distillation_spec({"loss_mode": "mirror", "mirror": {"objective": "mdpo", "eta": 1}})
    h = torch.ones(1, 1, dtype=torch.float64)
    p = torch.tensor([0.05, 0.45, 0.5, 0.2], dtype=torch.float64).log().reshape(-1, 1).requires_grad_()
    q = torch.tensor([0.45, 0.05, 0.5], dtype=torch.float64).log().reshape(-1, 1)
    result = full_vocab_objective_result(
        h,
        p,
        h,
        q,
        spec=spec,
        backend="precompute",
        distribution={"mode": "common_conditioned", "common_vocab_size": 3},
        termination_token_ids=(0, 1),
    )
    assert any(t is result.loss_sum for t in tree_flatten(result)[0])
    output = FullVocabOutput.from_result(result, spec, 1, termination=True, measure_name="conditional_coarsened_rkl")
    assert "distillation/full_kl" not in output.observations
    assert "distillation/conditional_rkl" in output.observations
    torch.testing.assert_close(
        output.observations["distillation/conditional_rkl"].state[0],
        torch.tensor(0.4 * math.log(9), dtype=torch.float64),
    )
    torch.testing.assert_close(
        output.observations["distillation/conditional_coarsened_rkl/objective"].state[0],
        torch.tensor(0.0, dtype=torch.float64),
        atol=1e-14,
        rtol=0,
    )
    assert any(t is output.objective_sum for t in tree_flatten(output)[0])
    output.objective_sum.backward()
    torch.testing.assert_close(p.grad, torch.zeros_like(p), atol=1e-14, rtol=0)


@pytest.mark.parametrize(
    "mode", ["token-mean", "token-sum", "seq-mean-token-mean", "seq-mean-token-sum", "seq-mean-token-sum-norm"]
)
@pytest.mark.parametrize("future_mode", ["none", "sum", "mean", "discounted_sequence"])
@pytest.mark.parametrize("eta,stop_ids", [(1.0, []), (1.0, [1, 4]), (0.3, []), (0.3, [1, 4])])
def test_aggregation_eta_and_future_keep_training_gradient_and_local_metrics_separate(mode, future_mode, eta, stop_ids):
    from test_trajectory_vjp_on_cpu import independent_reference

    from verl.trainer.distillation.reduction import PackedReduction

    torch.manual_seed(46)
    hs, ws, ht, wt, hq, wq = [
        torch.randn(*shape, dtype=torch.float64, requires_grad=True)
        for shape in [(5, 3), (7, 3), (5, 4), (7, 4), (5, 3), (7, 3)]
    ]
    loss_cfg = DistillationLossConfig(
        loss_mode="mirror",
        mirror={"eta": eta, "kappa": 0.5},
        full_vocab={
            "backend": "precompute",
            "termination_token_ids": stop_ids,
            "future": {"mode": future_mode, "gamma": 0.8 if future_mode.startswith("discounted") else 1.0},
        },
        use_policy_gradient=False,
        use_task_rewards=False,
        loss_max_clamp=None,
        log_prob_min_clamp=None,
    )
    spec = loss_cfg.loss_settings
    layout = PackedReduction(
        (3, 0, 2), torch.tensor([0, 1, 4, 0, 2]), torch.tensor([1, 2, 4, 0, 3]), mode, loss_cfg.full_vocab.future
    )
    expected_total, local_costs = independent_reference(hs, ws, ht, wt, hq, wq, spec, layout, stop_ids)
    result = full_vocab_objective_result(
        hs,
        ws,
        ht,
        wt,
        hq=hq,
        wq=wq,
        spec=spec,
        reduction=layout,
        token_chunk=2,
        termination_token_ids=stop_ids,
    )
    data = TensorDict({}, batch_size=[])
    tu.assign_non_tensor(data, dp_size=3, batch_num_tokens=5, full_vocab_num_sequences=2, global_batch_size=99)
    loss, observations = distillation_loss(
        SimpleNamespace(loss_agg_mode=mode, loss_scale_factor=7),
        SimpleNamespace(distillation_loss=loss_cfg),
        {"full_vocab": FullVocabOutput.from_result(result, spec, 5, termination=bool(stop_ids), num_sequences=2)},
        data,
    )
    scale = {
        "token-mean": 3 / 5,
        "token-sum": 3,
        "seq-mean-token-mean": 3 / 2,
        "seq-mean-token-sum": 3 / 2,
        "seq-mean-token-sum-norm": 3 / 14,
    }[mode]
    expected_loss = expected_total * scale / eta
    torch.testing.assert_close(loss, expected_loss)
    got = torch.autograd.grad(loss, (hs, ws, ht, wt, hq, wq), allow_unused=True)
    wanted = torch.autograd.grad(expected_loss, (hs, ws))
    assert got[2:] == (None,) * 4
    for actual, expected in zip(got[:2], wanted, strict=True):
        torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(result.local_objective_sum, local_costs.sum())
    primary = "full_kl" if eta == 1 else "mdpo/objective"
    if eta == 1 and stop_ids:
        lp, lt = (hs @ ws.T).log_softmax(-1), (ht @ wt.T).log_softmax(-1)
        expected_local_metric = (lp.exp() * (lp - lt)).sum()
        torch.testing.assert_close(observations["distillation/termination_equiv_kl"].state[0], local_costs.sum())
    else:
        expected_local_metric = local_costs.sum() / eta
    torch.testing.assert_close(
        observations["distillation/" + primary].state,
        torch.stack([expected_local_metric.detach(), local_costs.new_tensor(5)]),
    )
    assert observations["distillation/valid_sequences"].state.item() == 2
    if future_mode != "none":
        score = observations["distillation/future/score_surrogate"].state
        local = observations["distillation/local_surrogate"].state
        total = observations["distillation/training_surrogate"].state
        torch.testing.assert_close(total[0], expected_total.detach() / eta)
        torch.testing.assert_close(local[0] + score[0], total[0])
        assert total[1].item() == 5
        # Logged surrogate is a token ratio; final loss uses the selected global scale.
        torch.testing.assert_close(loss.detach(), total[0] * scale)
    assert all(not observation.state.requires_grad for observation in observations.values())


def test_output_rejects_legacy_results_without_an_explicit_local_numerator():
    from verl.workers.config.distillation_spec import resolve_distillation_spec

    settings = resolve_distillation_spec({"loss_mode": "mirror", "mirror": {"eta": 1}})
    with pytest.raises(TypeError, match="FullVocabObjectiveResult"):
        FullVocabOutput.from_result(SimpleNamespace(loss_sum=torch.tensor(-4.0)), settings, 2)
