# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent AdamW replay must explain both fresh and resumed optimizer steps."""

import copy

import pytest
import torch

from examples.mirroropd.benchmarks.full_vocab.analyze_optimizer_drift import adam


@pytest.mark.parametrize("preceding_steps", [0, 2])
def test_adam_replay_matches_float64_optimizer(preceding_steps):
    """Exercise tiny moments near epsilon as well as ordinary parameter scales."""
    parameter = torch.nn.Parameter(torch.tensor([0.003, -0.04, 1.0], dtype=torch.float64))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3, weight_decay=0.01)
    for _ in range(preceding_steps):
        parameter.grad = torch.tensor([1e-9, -2e-8, 0.1], dtype=torch.float64)
        optimizer.step()
    parameter.grad = torch.tensor([-3e-9, 1e-8, -0.02], dtype=torch.float64)
    snapshot = {
        "parameter_names": ["weight"],
        "parameters_before": {"weight": parameter.detach().clone()},
        "gradients_after_clip": {"weight": parameter.grad.clone()},
        "optimizer_before": copy.deepcopy(optimizer.state_dict()),
    }
    expected = adam(snapshot, "weight")
    optimizer.step()
    torch.testing.assert_close(expected, parameter, atol=1e-16, rtol=1e-14)
