# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Model readouts must preserve optimizer/RNG and runtime-checkpoint retention."""

from types import SimpleNamespace

import pytest
import torch

from verl.workers.engine.fsdp.transformer_impl import FSDPEngine


@pytest.mark.parametrize("fail", [False, True])
def test_model_only_save_restores_runtime_contents_and_retention(monkeypatch, fail):
    manager = SimpleNamespace(
        checkpoint_save_contents=["model", "optimizer", "extra", "hf_model"],
        previous_global_step=5,
        previous_saved_paths=["runtime-5"],
    )

    def save(**kwargs):
        assert manager.checkpoint_save_contents == ["model", "hf_model"]
        assert kwargs["max_ckpt_to_keep"] is None
        manager.previous_global_step = 8
        manager.previous_saved_paths.append("readout-8")
        if fail:
            raise RuntimeError("injected export failure")

    manager.save_checkpoint = save
    engine = SimpleNamespace(
        module=torch.nn.Linear(2, 2),
        checkpoint_manager=manager,
        _is_offload_param=False,
        _uses_fsdp2_cpu_offload_policy=True,
    )
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    if fail:
        with pytest.raises(RuntimeError, match="injected"):
            FSDPEngine.save_checkpoint(engine, "readout-8", global_step=8, max_ckpt_to_keep=1, model_only=True)
    else:
        FSDPEngine.save_checkpoint(engine, "readout-8", global_step=8, max_ckpt_to_keep=1, model_only=True)
    assert manager.checkpoint_save_contents == ["model", "optimizer", "extra", "hf_model"]
    assert manager.previous_global_step == 5 and manager.previous_saved_paths == ["runtime-5"]


@pytest.mark.parametrize("fail", [False, True])
def test_model_only_reload_never_requests_optimizer_or_rng(monkeypatch, fail):
    manager = SimpleNamespace(checkpoint_load_contents=["model", "optimizer", "extra"])

    def load(**kwargs):
        assert manager.checkpoint_load_contents == ["model"]
        if fail:
            raise RuntimeError("injected reload failure")

    manager.load_checkpoint = load
    engine = SimpleNamespace(checkpoint_manager=manager, _is_offload_param=False, _is_offload_optimizer=False)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    if fail:
        with pytest.raises(RuntimeError, match="injected"):
            FSDPEngine.load_checkpoint(engine, "readout-8", model_only=True)
    else:
        FSDPEngine.load_checkpoint(engine, "readout-8", model_only=True)
    assert manager.checkpoint_load_contents == ["model", "optimizer", "extra"]
