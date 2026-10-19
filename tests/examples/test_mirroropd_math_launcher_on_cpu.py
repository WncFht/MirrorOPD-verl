# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Compose the exact shell overrides without model loading or Ray startup."""

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from verl.utils.config import omega_conf_to_dataclass, validate_config

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples/mirroropd/run_deepseek_r1_1_5b_fsdp2.sh"


def _launch(tmp_path, **overrides):
    env = os.environ.copy()
    for name in ("CUDA_VISIBLE_DEVICES", "STUDENT_GPUS", "TEACHER_GPUS", "ROLLOUT_TP", "TEACHER_TP"):
        env.pop(name, None)
    env.update(
        DRY_RUN="1",
        VERL_USE_UV="0",
        PYTHON_BIN=sys.executable,
        MODEL_ROOT=str(tmp_path / "models with spaces"),
        DATA_ROOT=str(tmp_path / "data"),
        RUN_DIR=str(tmp_path / "run"),
        EXPERIMENT_NAME="test",
        TRAIN_BATCH_SIZE="64",
    )
    env.update(overrides)
    return subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True)


def _config(tmp_path, **overrides):
    result = _launch(tmp_path, **overrides)
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout)
    with initialize_config_dir(config_dir=str(ROOT / "verl/trainer/config"), version_base=None):
        return compose(config_name="ppo_trainer", overrides=command[command.index("verl.trainer.main_ppo") + 1 :])


def test_launcher_composes_native_single_teacher_baseline(tmp_path):
    cfg = _config(tmp_path)
    OmegaConf.resolve(cfg)
    validate_config(cfg, use_reference_policy=False, use_critic=False)
    assert cfg.trainer.use_v1 and cfg.trainer.v1.trainer_mode == "sync"
    assert cfg.trainer.n_gpus_per_node == 4 and cfg.distillation.n_gpus_per_node == 4
    assert cfg.actor_rollout_ref.actor.strategy == "fsdp2"
    assert cfg.data.train_batch_size == cfg.actor_rollout_ref.actor.ppo_mini_batch_size == 64
    assert cfg.actor_rollout_ref.actor.ppo_epochs == 1
    assert cfg.actor_rollout_ref.rollout.response_length == 7168
    assert cfg.actor_rollout_ref.rollout.val_kwargs.max_tokens == 31744
    assert cfg.actor_rollout_ref.rollout.val_kwargs.n == 16
    assert cfg.actor_rollout_ref.rollout.max_model_len == 35841
    assert cfg.distillation.teacher_models.teacher_model.inference.max_model_len == 11265
    assert cfg.distillation.distillation_loss.loss_mode == "k1"
    assert not cfg.distillation.distillation_loss.use_task_rewards
    assert cfg.distillation.distillation_loss.loss_max_clamp is None
    assert [Path(p).parent.name for p in cfg.data.val_files] == ["MATH-500", "AMC23", "AIME25"]
    assert cfg.actor_rollout_ref.model.path.endswith("models with spaces/DeepSeek-R1-Distill-Qwen-1.5B")
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("teacher_gpus,teacher_tp,replicas", [(4, 1, 4), (4, 2, 2), (1, 1, 1)])
def test_native_teacher_config_uses_entire_requested_pool(tmp_path, teacher_gpus, teacher_tp, replicas):
    cfg = _config(
        tmp_path,
        TEACHER_GPUS=str(teacher_gpus),
        TEACHER_TP=str(teacher_tp),
        CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
    )
    distillation = omega_conf_to_dataclass(cfg.distillation)
    assert list(distillation.teacher_models) == ["default"]
    teacher = distillation.teacher_models["default"]
    assert teacher.num_replicas == replicas
    assert teacher.per_replica_world_size == teacher_tp
    assert teacher.world_size == teacher_gpus


@pytest.mark.parametrize(
    "overrides",
    [
        {"STUDENT_GPUS": "3"},
        {"TEACHER_TP": "3"},
        {"MAX_VAL_RESPONSE_LENGTH": "0"},
        {"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6"},
    ],
)
def test_launcher_rejects_invalid_resource_or_budget_settings(tmp_path, overrides):
    assert _launch(tmp_path, **overrides).returncode == 2


def test_validation_context_checked_before_worker_startup(tmp_path):
    cfg = _config(tmp_path)
    cfg.actor_rollout_ref.rollout.max_model_len = 11265
    with pytest.raises(ValueError, match="prompt_length"):
        validate_config(cfg, use_reference_policy=False, use_critic=False)


def test_validation_budget_rejects_default_multiturn_agent(tmp_path):
    cfg = _config(tmp_path)
    cfg.actor_rollout_ref.rollout.agent.default_agent_loop = "tool_agent"
    with pytest.raises(ValueError, match="single_turn_agent"):
        validate_config(cfg, use_reference_policy=False, use_critic=False)


def test_validation_budget_rejects_backend_that_overwrites_request_limit(tmp_path):
    cfg = _config(tmp_path)
    cfg.actor_rollout_ref.rollout.name = "trtllm"
    with pytest.raises(ValueError, match="vllm or sglang"):
        validate_config(cfg, use_reference_policy=False, use_critic=False)


def test_uv_launch_includes_math_dependencies_for_driver_and_workers(tmp_path):
    result = _launch(tmp_path, VERL_USE_UV="1", DEVICE="gpu")
    assert result.returncode == 0
    command = shlex.split(result.stdout)
    assert command[:2] == ["uv", "run"]
    assert "math" in command[: command.index("-m")]
    worker_python = next(arg for arg in command if arg.startswith("ray_kwargs.ray_init.runtime_env.py_executable="))
    assert "--extra math" in worker_python


def test_real_launch_requires_explicit_gpu_allocation(tmp_path):
    result = _launch(tmp_path, DRY_RUN="0", CONFIG_ONLY="0")
    assert result.returncode == 2
    assert "allocated to this run" in result.stderr
    assert not (tmp_path / "run").exists()
