# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Verify optimizer execution facts at the actual engine boundary."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from tensordict import TensorDict

from verl.workers.engine.base import BaseEngine
from verl.workers.engine.fsdp import transformer_impl
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine


def engine():
    module = torch.nn.Linear(2, 1, bias=False)
    return SimpleNamespace(
        module=module,
        optimizer=torch.optim.AdamW(module.parameters(), lr=0.01),
        optimizer_config=SimpleNamespace(clip_grad=1.0),
        scaler=None,
        _qat_enabled=False,
    )


@pytest.mark.parametrize("gradient,expected", [(1.0, True), (float("inf"), False), (float("nan"), False), (0.0, True)])
def test_fsdp_reports_actual_adamw_execution(gradient, expected):
    worker = engine()
    for parameter in worker.module.parameters():
        parameter.grad = torch.full_like(parameter, gradient)
    original = worker.optimizer.step
    worker.optimizer.step = MagicMock(wraps=original)
    FSDPEngine.optimizer_step(worker)
    assert worker._optimizer_step_applied is expected
    assert worker.optimizer.step.call_count == int(expected)


def test_optimizer_exception_never_reports_success():
    worker = engine()
    worker._optimizer_step_applied = None
    for parameter in worker.module.parameters():
        parameter.grad = torch.ones_like(parameter)
    worker.optimizer.step = MagicMock(side_effect=RuntimeError("optimizer failed"))
    with pytest.raises(RuntimeError, match="optimizer failed"):
        FSDPEngine.optimizer_step(worker)
    assert worker._optimizer_step_applied is None


def test_base_resets_status_and_scaler_hook_observes_no_update():
    worker = SimpleNamespace(
        optimizer_zero_grad=lambda: None,
        is_mp_src_rank_with_outputs=lambda: True,
        forward_backward_batch=lambda *a, **k: {"metrics": {}},
        _optimizer_step_applied=True,
    )
    worker.optimizer_step = lambda: 2.0
    result = BaseEngine.train_batch(worker, TensorDict({}, batch_size=[]), lambda: None)
    assert result["metrics"] == {"grad_norm": 2.0}
    assert worker._optimizer_step_applied is None
    scaled = engine()
    scaled.scaler = MagicMock()
    scaled._optimizer_step_applied = None
    for parameter in scaled.module.parameters():
        parameter.grad = torch.ones_like(parameter)
    FSDPEngine.optimizer_step(scaled)
    assert scaled._optimizer_step_applied is False


def test_global_empty_training_batch_is_rejected_before_forward(monkeypatch):
    monkeypatch.setattr(transformer_impl, "get_device_id", lambda: "cpu")
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *a, **k: None)
    worker = SimpleNamespace(ulysses_sequence_parallel_size=1, get_data_parallel_group=lambda: None)
    data = TensorDict({"loss_mask": torch.zeros(2, 3)}, batch_size=[2])
    with pytest.raises(ValueError, match="zero globally valid"):
        FSDPEngine.forward_backward_batch(worker, data, lambda: None)


@pytest.mark.parametrize("overflow", [False, True])
@pytest.mark.parametrize("fused", [False, True])
def test_actual_grad_scaler_reports_applied_and_skipped_steps(overflow, fused):
    worker = engine()
    worker.optimizer = torch.optim.AdamW(worker.module.parameters(), lr=0.01, fused=fused)
    worker.scaler = torch.amp.GradScaler("cpu", init_scale=32.0)
    before = [p.detach().clone() for p in worker.module.parameters()]
    loss = worker.module(torch.ones(1, 2)).square().sum()
    worker.scaler.scale(loss).backward()
    if overflow:
        for p in worker.module.parameters():
            p.grad.fill_(float("inf"))
    worker._optimizer_step_applied = None
    FSDPEngine.optimizer_step(worker)
    assert worker._optimizer_step_applied is (not overflow)
    changed = any(not torch.equal(p, old) for p, old in zip(worker.module.parameters(), before, strict=True))
    assert changed is (not overflow)
    assert worker.scaler.get_scale() == (16.0 if overflow else 32.0)
