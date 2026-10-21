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


def test_base_resets_status_every_attempt_and_does_not_guess_scaler_or_unknown_backend():
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
    assert scaled._optimizer_step_applied is None


def test_global_empty_training_batch_is_rejected_before_forward(monkeypatch):
    monkeypatch.setattr(transformer_impl, "get_device_id", lambda: "cpu")
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *a, **k: None)
    worker = SimpleNamespace(ulysses_sequence_parallel_size=1, get_data_parallel_group=lambda: None)
    data = TensorDict({"loss_mask": torch.zeros(2, 3)}, batch_size=[2])
    with pytest.raises(ValueError, match="zero globally valid"):
        FSDPEngine.forward_backward_batch(worker, data, lambda: None)
