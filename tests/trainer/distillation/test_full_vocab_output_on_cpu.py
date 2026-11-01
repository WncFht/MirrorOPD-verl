# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""One engine/loss contract must preserve endpoint, mirror and EOS semantics."""

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
