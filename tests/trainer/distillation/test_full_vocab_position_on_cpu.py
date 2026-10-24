# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Position diagnostics must preserve loss/VJP and merge by token counts."""

import sys
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from verl.trainer.distillation.full_vocab import FullKLResult, full_reverse_kl, full_reverse_kl_result
from verl.trainer.distillation.losses import distillation_loss
from verl.utils import tensordict_utils as tu
from verl.utils.metric.observations import Observation
from verl.workers.config.distillation import DistillationConfig, DistillationLossConfig


@pytest.mark.parametrize("backend", ["dense", "recompute", "streaming", "cached", "precompute", "precompute_fused"])
@pytest.mark.parametrize("tokens,extra", [(0, 0), (7, 0), (7, 4)])
def test_diagnostic_rows_are_exact_and_do_not_change_gradients(backend, tokens, extra):
    torch.manual_seed(82)
    hs = torch.randn(tokens, 5, dtype=torch.float64, requires_grad=True)
    ws = torch.randn(17, 5, dtype=torch.float64, requires_grad=True)
    ht = torch.randn(tokens, 8, dtype=torch.float64, requires_grad=True)
    wt = torch.randn(17 + extra, 8, dtype=torch.float64, requires_grad=True)
    lp = (hs @ ws.T).log_softmax(-1)
    lt = (ht.detach() @ wt.detach().T).log_softmax(-1)[:, :17]
    expected_rows = (lp.exp() * (lp - lt)).sum(-1)
    expected_grad = torch.autograd.grad(expected_rows.sum() * 0.37, (hs, ws))
    total, rows = full_reverse_kl_result(hs, ws, ht, wt, backend=backend, token_chunk=3, vocab_chunk=4)
    assert not rows.requires_grad
    torch.testing.assert_close(rows, expected_rows.detach(), atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(total, rows.sum(), atol=1e-10, rtol=1e-10)
    grads = torch.autograd.grad(total * 0.37, (hs, ws, ht, wt), allow_unused=True)
    for actual, expected in zip(grads[:2], expected_grad, strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    assert grads[2:] == (None, None)


def test_scalar_only_backend_expresses_capability_in_result(monkeypatch):
    class ScalarKernel:
        def __init__(self, **kwargs):
            pass

        def __call__(self, hs, ws, ht, wt, labels):
            return (hs.square().sum() + ws.square().sum()) / len(hs)

    monkeypatch.setitem(
        sys.modules, "liger_kernel.transformers.fused_linear_jsd", SimpleNamespace(LigerFusedLinearJSD=ScalarKernel)
    )
    hs, ws, ht, wt = [torch.ones(shape) for shape in [(2, 3), (7, 3), (2, 3), (7, 3)]]
    result = full_reverse_kl_result(hs, ws, ht, wt, backend="liger")
    assert isinstance(result, FullKLResult) and result.token_kl is None
    torch.testing.assert_close(full_reverse_kl(hs, ws, ht, wt, backend="liger"), result.loss_sum)


def test_loss_statistics_merge_unequal_microbatches_and_reconstruct_prefix():
    cfg = DistillationConfig(
        distillation_loss=DistillationLossConfig(
            loss_mode="full_reverse_kl",
            use_policy_gradient=False,
            use_task_rewards=False,
            loss_max_clamp=None,
            log_prob_min_clamp=None,
        )
    )
    data = TensorDict({}, batch_size=[])
    tu.assign_non_tensor(data, dp_size=2, batch_num_tokens=40)
    parts = []
    for ns, vs in [
        (torch.arange(10), torch.arange(10) * 0.2),
        (torch.ones(10), torch.arange(10) * 0.03),
        (torch.zeros(10), torch.zeros(10)),
    ]:
        _, obs = distillation_loss(
            SimpleNamespace(loss_agg_mode="token-mean"),
            cfg,
            {
                "full_kl_sum": vs.sum(),
                "full_kl_count": ns.sum(),
                "full_kl_position_sums": vs,
                "full_kl_position_counts": ns,
            },
            data,
        )
        parts.append(obs)
    merged = {k: Observation.merge_partitions([part[k] for part in parts]) for k in parts[0]}
    sums = [merged[f"distillation/full_kl_position/p{i * 10:02d}_{(i + 1) * 10:02d}/sum"] for i in range(10)]
    counts = [merged[f"distillation/full_kl_position/p{i * 10:02d}_{(i + 1) * 10:02d}/token_count"] for i in range(10)]
    assert sum(sums) / sum(counts) == pytest.approx(merged["distillation/full_kl"])
    assert sum(sums[:3]) / sum(counts[:3]) == pytest.approx(0.69 / 6)
    assert merged["distillation/full_kl_position/p20_30/mean"] == pytest.approx(0.46 / 3)
