# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Trajectory credit and aggregate factorization against literal definitions."""

import pytest
import torch

from verl.trainer.distillation.reduction import PackedReduction, reduction_scale, suffix_credit
from verl.workers.config.distillation_spec import FutureCreditConfig


@pytest.mark.parametrize("mode", ["none", "sum", "mean", "discounted_sum", "discounted_mean", "discounted_sequence"])
@pytest.mark.parametrize("gamma", [1.0, 0.93, 0.001])
@pytest.mark.parametrize("chunk", [1, 2, 7])
def test_suffix_definition_and_chunk_carry(mode, gamma, chunk):
    if not mode.startswith("discounted") and gamma != 1:
        pytest.skip("discount only applies to discounted modes")
    config = FutureCreditConfig(mode=mode, gamma=gamma)
    cost = torch.tensor([0.5, -0.2, 1.1, 0.8, -0.3], dtype=torch.float64, requires_grad=True)
    times = torch.tensor([0, 2, 5, 6, 10000])
    layout = PackedReduction((3, 0, 1, 1), times, torch.zeros_like(times), future=config)
    actual, expected = torch.zeros_like(cost), torch.zeros_like(cost)
    carry = None
    for start, stop, reset in layout.blocks(chunk):
        actual[start:stop], carry = suffix_credit(
            cost[start:stop], times[start:stop], config, carry=None if reset else carry
        )
    offset = 0
    for n in layout.counts:
        for t in range(offset, offset + n):
            weights = gamma ** (times[t + 1 : offset + n] - times[t]).double()
            value = (cost.detach()[t + 1 : offset + n] * weights).sum()
            if mode in ("mean", "discounted_mean") and len(weights):
                value /= weights.sum().clamp_min(torch.finfo(cost.dtype).tiny)
            expected[t] = value if mode != "none" else 0
        offset += n
    torch.testing.assert_close(actual, expected)
    assert not actual.requires_grad


@pytest.mark.parametrize(
    "mode", ["token-mean", "token-sum", "seq-mean-token-mean", "seq-mean-token-sum", "seq-mean-token-sum-norm"]
)
def test_aggregate_factorization(mode):
    from verl.trainer.ppo.core_algos import agg_loss

    values = torch.arange(12, dtype=torch.float64).reshape(3, 4).requires_grad_()
    mask = torch.tensor([[1, 0, 1, 1], [0, 0, 0, 0], [1, 0, 0, 0]])
    layout = PackedReduction((3, 0, 1), torch.arange(4), torch.zeros(4, dtype=torch.long), mode)
    weights = layout.row_weights(values.dtype)
    packed = values[mask.bool()]
    actual = (packed if weights is None else packed * weights).sum() * reduction_scale(
        mode, dp_size=2, num_tokens=11, num_sequences=5, fixed_scale=7
    )
    expected = agg_loss(values, mask, mode, dp_size=2, batch_num_tokens=11, global_batch_size=5, loss_scale_factor=7)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, values)[0], torch.autograd.grad(expected, values)[0])


@pytest.mark.parametrize(
    "mode,gamma", [("bad", 1), ("sum", 0.9), ("discounted_sum", 0), ("discounted_sum", float("nan"))]
)
def test_invalid_future_config(mode, gamma):
    with pytest.raises(ValueError):
        FutureCreditConfig(mode=mode, gamma=gamma)


def test_fixed_normalization_requires_explicit_positive_constant():
    for constant in (None, 0, -1, float("inf")):
        with pytest.raises(ValueError, match="explicit positive"):
            reduction_scale("seq-mean-token-sum-norm", num_sequences=3, fixed_scale=constant)
