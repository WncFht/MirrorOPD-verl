# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from verl.trainer.distillation.full_vocab import full_reverse_kl
from verl.trainer.distillation.hidden import prediction_positions
from verl.trainer.distillation.losses import distillation_loss
from verl.utils import tensordict_utils as tu
from verl.workers.config.distillation import DistillationConfig, DistillationLossConfig


@pytest.mark.parametrize("backend", ["dense", "recompute", "cached", "streaming", "precompute", "precompute_fused"])
@pytest.mark.parametrize("tokens", [0, 1, 7])
def test_value_and_vjp(backend, tokens):
    torch.manual_seed(42)
    hs = torch.randn(tokens, 5, dtype=torch.float64, requires_grad=True)
    ws = torch.randn(19, 5, dtype=torch.float64, requires_grad=True)
    ht = torch.randn(tokens, 8, dtype=torch.float64, requires_grad=True)
    wt = torch.randn(19, 8, dtype=torch.float64, requires_grad=True)
    lp, lt = (hs @ ws.T).log_softmax(-1), (ht.detach() @ wt.detach().T).log_softmax(-1)
    expected = (lp.exp() * (lp - lt)).sum()
    eg = torch.autograd.grad(expected * 0.37, (hs, ws))
    actual = full_reverse_kl(hs, ws, ht, wt, backend=backend, token_chunk=3, vocab_chunk=4)
    ag = torch.autograd.grad(actual * 0.37, (hs, ws, ht, wt), allow_unused=True)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    for a, e in zip(ag[:2], eg, strict=True):
        torch.testing.assert_close(a, e, atol=1e-10, rtol=1e-10)
    assert ag[2:] == (None, None)


def test_shift_mask_and_sample_boundaries():
    ids = torch.nested.nested_tensor([torch.arange(6), torch.arange(4)], layout=torch.jagged)
    masks = torch.nested.nested_tensor([torch.tensor([1, 0, 1]), torch.tensor([0, 1])], layout=torch.jagged)
    positions, packed = prediction_positions(ids, masks)
    assert [p.tolist() for p in positions] == [[2, 4], [2]]
    assert packed.tolist() == [2, 4, 8]
    with pytest.raises(ValueError, match="preceding"):
        prediction_positions(ids, torch.nested.nested_tensor([torch.ones(6), torch.ones(4)], layout=torch.jagged))


def test_global_token_normalization():
    config = SimpleNamespace(loss_agg_mode="token-mean")
    loss_config = DistillationLossConfig(
        loss_mode="full_reverse_kl",
        use_policy_gradient=False,
        use_task_rewards=False,
        loss_max_clamp=None,
        log_prob_min_clamp=None,
    )
    distillation = DistillationConfig(distillation_loss=loss_config)
    data = TensorDict({}, batch_size=[])
    tu.assign_non_tensor(data, dp_size=2, batch_num_tokens=11)
    total = torch.tensor(7.0, requires_grad=True)
    loss, _ = distillation_loss(config, distillation, {"full_kl_sum": total, "full_kl_count": 3}, data)
    loss.backward()
    assert total.grad.item() == pytest.approx(2 / 11)
    with pytest.raises(ValueError, match="direct loss"):
        DistillationLossConfig(loss_mode="full_reverse_kl")
