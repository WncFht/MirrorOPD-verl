# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from test_sampled_token_loss_on_cpu import inputs, loss_config

from verl.trainer.distillation.losses import distillation_loss
from verl.utils import tensordict_utils as tu
from verl.workers.utils.loss_context import LossContext
from verl.workers.utils.losses import ppo_loss


@pytest.mark.parametrize("kind", ["namespace", "dataclass", "omega"])
def test_legacy_config_binding_is_per_call_and_preserves_shared_state(kind):
    if kind == "namespace":
        config = SimpleNamespace(global_batch_info={"sentinel": 7})
    elif kind == "dataclass":
        config = loss_config()
    else:
        config = OmegaConf.create({"global_batch_info": {"sentinel": 7}})
        OmegaConf.set_readonly(config, True)
    before = dict(config.global_batch_info)
    first = LossContext(2, 6, 2).bind_legacy_config(config)
    second = LossContext(4, 12, 4).bind_legacy_config(config)
    assert dict(config.global_batch_info) == before
    assert first.global_batch_info["batch_num_tokens"] == 6
    assert second.global_batch_info["batch_num_tokens"] == 12
    assert first.global_batch_info is not second.global_batch_info


@pytest.mark.parametrize("mode", ["vanilla", "token_ppo", "token_is", "reinforce"])
def test_distillation_calls_do_not_mutate_actor_or_objective_config(mode):
    x = torch.tensor([[-0.8, -1.2]], requires_grad=True)
    teacher = torch.tensor([[-0.5, -1.3]])
    output, data = inputs(x, teacher, x.detach())
    data["old_log_probs"] = x.detach().clone()
    actor = SimpleNamespace(loss_agg_mode="token-mean", loss_scale_factor=None, global_batch_info={"sentinel": 9})
    objective = loss_config(mode)
    original = dict(objective.global_batch_info)
    a, _ = distillation_loss(actor, SimpleNamespace(distillation_loss=objective), output, data)
    tu.assign_non_tensor(data, batch_num_tokens=4)
    b, _ = distillation_loss(actor, SimpleNamespace(distillation_loss=objective), output, data)
    torch.testing.assert_close(a, 2 * b)
    assert actor.global_batch_info == {"sentinel": 9}
    assert objective.global_batch_info == original


def test_task_ppo_with_diagnostics_uses_private_normalization_config():
    x = torch.tensor([[-0.8, -1.2]], requires_grad=True)
    output, data = inputs(x, x.detach(), x.detach())
    data["old_log_probs"] = x.detach().clone()
    data["advantages"] = torch.ones_like(x)
    tu.assign_non_tensor(data, collect_metrics=True, metrics_has_anchor=True)
    config = OmegaConf.create(
        {
            "global_batch_info": {"sentinel": 3},
            "loss_agg_mode": "token-mean",
            "loss_scale_factor": None,
            "policy_loss": {"loss_mode": "vanilla"},
            "clip_ratio": 0.2,
            "clip_ratio_low": None,
            "clip_ratio_high": None,
            "use_kl_loss": False,
            "entropy_coeff": 0.0,
        }
    )
    loss, metrics = ppo_loss(config, output, data)
    loss.backward()
    assert config.global_batch_info == {"sentinel": 3}
    assert "advantage/task/mean" in metrics
    torch.testing.assert_close(x.grad, torch.full_like(x, -0.5))
