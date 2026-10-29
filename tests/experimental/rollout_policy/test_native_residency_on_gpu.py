# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-style optimizer offload must release cached GPU storage before fusion startup."""

import inspect
from types import SimpleNamespace

import pytest
import torch

from verl.utils.device import get_device_name, get_torch_device
from verl.utils.fsdp_utils import offload_fsdp_optimizer
from verl.workers.engine_workers import ActorRolloutRefWorker


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires the explicitly allocated GPU")
def test_native_prepare_releases_optimizer_allocator_cache():
    device = get_torch_device()
    device.empty_cache()
    before = device.memory_reserved()
    parameter = torch.nn.Parameter(torch.ones(2048, 2048, device=get_device_name()))
    optimizer = torch.optim.AdamW([parameter])
    optimizer.state[parameter] = {
        "step": torch.tensor(1.0),
        "exp_avg": torch.zeros_like(parameter),
        "exp_avg_sq": torch.zeros_like(parameter),
    }
    parameter.data = parameter.data.cpu()
    offload_fsdp_optimizer(optimizer)
    device.synchronize()
    assert device.memory_reserved() > before

    class Engine:
        is_param_offload_enabled = True
        is_optimizer_offload_enabled = True

        def to(self, target, model, optimizer, grad):
            assert target == "cpu" and model and optimizer and grad

    worker = SimpleNamespace(actor=SimpleNamespace(engine=Engine()))
    inspect.unwrap(ActorRolloutRefWorker.park_fusion_actor)(worker)
    assert device.memory_reserved() <= before
    assert parameter.device.type == "cpu"
    assert all(value.device.type == "cpu" for value in optimizer.state[parameter].values())
