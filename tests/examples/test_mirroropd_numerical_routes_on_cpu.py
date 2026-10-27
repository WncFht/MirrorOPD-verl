# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Resolve actual launcher overrides and lock the repaired default routes."""

import pytest
from hydra.errors import InstantiationException
from omegaconf import OmegaConf
from test_mirroropd_math_launcher_on_cpu import SCRIPT, _config

from verl.utils.config import omega_conf_to_dataclass, validate_config


@pytest.mark.parametrize("preset", ["E15", "E35", "E41", "E44", "E45", "E46", "E47"])
def test_registered_defaults_keep_project_numerics(tmp_path, preset):
    config = _config(tmp_path, command=["bash", str(SCRIPT), preset])
    OmegaConf.resolve(config)
    validate_config(config, use_reference_policy=False, use_critic=False)
    model = config.actor_rollout_ref.model
    assert not model.use_fused_kernels and not model.use_liger
    assert not config.actor_rollout_ref.actor.use_torch_compile
    assert config.distillation.execution_mode == "shared_sync"
    distillation = omega_conf_to_dataclass(config.distillation)
    assert not distillation.shared_teacher.use_fused_kernels
    loss = distillation.distillation_loss
    if loss.loss_settings.payload_kind == "hidden":
        assert loss.full_vocab.backend in ("precompute", "precompute_fused")


@pytest.mark.parametrize("preset", ["E15", "E44", "E45"])
def test_shared_sync_rejects_general_fused_head_override(tmp_path, preset):
    config = _config(tmp_path, command=["bash", str(SCRIPT), preset, "actor_rollout_ref.model.use_fused_kernels=True"])
    with pytest.raises(ValueError, match="use_fused_kernels=False"):
        validate_config(config, use_reference_policy=False, use_critic=False)


@pytest.mark.parametrize("preset", ["E45", "E46", "E47"])
def test_nonendpoint_mirror_rejects_external_liger_loss(tmp_path, preset):
    config = _config(tmp_path, command=["bash", str(SCRIPT), preset], FULL_KL_BACKEND="liger")
    with pytest.raises(InstantiationException, match="mirror full_vocab supports"):
        omega_conf_to_dataclass(config.distillation)


def test_endpoint_external_backend_is_an_explicit_bypass(tmp_path):
    config = _config(tmp_path, command=["bash", str(SCRIPT), "E44"], FULL_KL_BACKEND="liger")
    validate_config(config, use_reference_policy=False, use_critic=False)
    loss = omega_conf_to_dataclass(config.distillation).distillation_loss
    assert loss.loss_settings.teacher_only_kl and loss.full_vocab.backend == "liger"


@pytest.mark.parametrize("flash_available", [False, True])
@pytest.mark.parametrize("disable_flash", ["0", "1"])
def test_sampled_dispatch_and_environment_keep_stable_vjp(monkeypatch, flash_available, disable_flash):
    import torch

    import verl.utils.torch_functional as functional

    monkeypatch.setattr(functional, "FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE", flash_available)
    monkeypatch.setattr(functional, "NPU_CROSS_ENTROPY_LOSS_AVAILABLE", False)
    monkeypatch.setenv("VERL_DISABLE_FLASH_ATTN_CE", disable_flash)
    logits = torch.tensor([[9.0, -9.0, -9.0]], requires_grad=True)
    loss = -functional.logprobs_from_logits(logits, torch.tensor([0])).sum()
    gradient = torch.autograd.grad(loss, logits)[0]
    assert loss > 0 and gradient[0, 0] < 0
    torch.testing.assert_close(gradient[0, 0], -gradient[0, 1:].sum(), atol=0, rtol=1e-6)
