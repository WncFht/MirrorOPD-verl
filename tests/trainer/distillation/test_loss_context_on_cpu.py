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


@pytest.mark.parametrize("mode", ["vanilla", "token_ppo", "token_is", "reinforce"])
def test_distillation_calls_do_not_mutate_actor_or_objective_config(mode):
    x = torch.tensor([[-0.8, -1.2]], requires_grad=True)
    teacher = torch.tensor([[-0.5, -1.3]])
    output, data = inputs(x, teacher, x.detach())
    data["old_log_probs"] = x.detach().clone()
    actor = SimpleNamespace(loss_agg_mode="token-mean", loss_scale_factor=None, global_batch_info={"sentinel": 9})
    objective = loss_config(mode)
    original = dict(objective)
    a, _ = distillation_loss(actor, SimpleNamespace(distillation_loss=objective), output, data)
    tu.assign_non_tensor(data, batch_num_tokens=4)
    b, _ = distillation_loss(actor, SimpleNamespace(distillation_loss=objective), output, data)
    torch.testing.assert_close(a, 2 * b)
    assert actor.global_batch_info == {"sentinel": 9}
    assert dict(objective) == original


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


@pytest.mark.parametrize("name,kind", [("k1", "sampled"), ("forward_kl_topk", "topk"), ("full_forward_kl", "hidden")])
def test_objective_metadata_has_one_immutable_payload_source(name, kind):
    from dataclasses import FrozenInstanceError

    from verl.workers.config.distillation_spec import get_distillation_loss_settings

    spec = get_distillation_loss_settings(name)
    assert spec.payload_kind == kind
    assert spec.use_topk == (kind == "topk")
    assert spec.use_estimator == (kind == "sampled")
    assert isinstance(spec.names, tuple)
    with pytest.raises(FrozenInstanceError):
        spec.names = ("changed",)
    with pytest.raises(FrozenInstanceError):
        spec.payload_kind = "changed"
    assert get_distillation_loss_settings(name).payload_kind == kind


@pytest.mark.parametrize("mode", ["vanilla", "dppo_tv", "dppo_kl", "gspo", "sapo", "gpg", "cispo", "geo_mean"])
def test_registered_policy_losses_use_explicit_global_context(mode):
    from verl.trainer.ppo.core_algos import get_policy_loss_fn
    from verl.workers.config import ActorConfig

    config = ActorConfig(
        strategy="fsdp2",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        clip_ratio=0.2,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
    )
    # Same tensor graph and static config, changing only the global population.
    logp = torch.tensor([[-0.8, -1.2]], requires_grad=True)
    function = get_policy_loss_fn(mode)
    mode_name = "seq-mean-token-mean" if mode == "sapo" else "token-mean"
    kwargs = dict(
        old_log_prob=logp.detach(),
        log_prob=logp,
        advantages=torch.tensor([[1.0, -0.5]]),
        response_mask=torch.ones_like(logp),
        loss_agg_mode=mode_name,
        config=config,
    )
    a, _ = function(**kwargs, context=LossContext(dp_size=2, batch_num_tokens=6, global_batch_size=2))
    b, _ = function(**kwargs, context=LossContext(dp_size=2, batch_num_tokens=12, global_batch_size=4))
    torch.testing.assert_close(a, 2 * b)
    ga = torch.autograd.grad(a, logp, retain_graph=True)[0]
    gb = torch.autograd.grad(b, logp)[0]
    torch.testing.assert_close(ga, 2 * gb)
    assert not hasattr(config, "global_batch_info")


def test_geometric_policy_preserves_sequence_population_across_unequal_microbatches():
    from verl.trainer.ppo.core_algos import compute_policy_loss_geo_mean
    from verl.workers.config import ActorConfig

    cfg = ActorConfig(strategy="fsdp2", rollout_n=1, ppo_micro_batch_size_per_gpu=1)
    logp = torch.tensor([[-1.0, -0.7], [-0.8, -0.6], [-1.2, -0.9]], requires_grad=True)
    old = logp.detach() - 0.01
    advantages = torch.tensor([[1.0, 1.0], [2.0, 2.0], [-0.5, -0.5]])
    mask = torch.tensor([[1.0, 0.0], [1.0, 1.0], [1.0, 1.0]])
    context = LossContext(dp_size=2, global_batch_size=3, batch_num_tokens=5)
    full, _ = compute_policy_loss_geo_mean(old, logp, advantages, mask, config=cfg, context=context)
    split = sum(
        compute_policy_loss_geo_mean(old[s], logp[s], advantages[s], mask[s], config=cfg, context=context)[0]
        for s in [slice(0, 1), slice(1, 3)]
    )
    torch.testing.assert_close(full, split)
    full_grad = torch.autograd.grad(full, logp, retain_graph=True)[0]
    split_grad = torch.autograd.grad(split, logp)[0]
    torch.testing.assert_close(full_grad, split_grad)
