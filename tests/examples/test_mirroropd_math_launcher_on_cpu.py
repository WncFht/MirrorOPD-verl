# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Compose the exact shell overrides without model loading or Ray startup."""

import os
import shlex
import shutil
import subprocess
import sys
from configparser import ConfigParser
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from verl.utils.config import omega_conf_to_dataclass, validate_config

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples/mirroropd/run_math_deepseek_r1_1_5b_fsdp2.sh"


def _launch(tmp_path, script=SCRIPT, **overrides):
    env = os.environ.copy()
    for name in ("CUDA_VISIBLE_DEVICES", "STUDENT_GPUS", "TEACHER_GPUS", "ROLLOUT_TP", "TEACHER_TP", "DOMAIN"):
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
    return subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)


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
    assert not cfg.trainer.v1.opd_metrics
    assert cfg.trainer.n_gpus_per_node == 4 and cfg.distillation.n_gpus_per_node == 4
    assert cfg.actor_rollout_ref.actor.strategy == "fsdp2"
    assert cfg.data.train_batch_size == cfg.actor_rollout_ref.actor.ppo_mini_batch_size == 64
    assert cfg.data.continuous_token.model_family == "deepseek"
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


@pytest.mark.parametrize(
    "domain,teacher,train,validation",
    [
        ("code", "DeepCoder-1.5B-Preview", "code/TACO/train.parquet", ["LCBv6"]),
        ("if", "UltraData-IF-1.5B", "if/MultiIF/train.parquet", []),
    ],
)
def test_domain_launchers_select_teacher_and_dataset_protocol(tmp_path, domain, teacher, train, validation):
    script = ROOT / f"examples/mirroropd/run_{domain}_deepseek_r1_1_5b_fsdp2.sh"
    cfg = _config(tmp_path, script=script)
    OmegaConf.resolve(cfg)
    validate_config(cfg, use_reference_policy=False, use_critic=False)
    assert cfg.distillation.teacher_models.teacher_model.model_path.endswith(teacher)
    assert cfg.data.train_files == [str(tmp_path / "data" / train)]
    assert [Path(p).parent.name for p in cfg.data.val_files] == validation
    assert cfg.data.continuous_token.model_family == "deepseek"
    assert cfg.trainer.n_gpus_per_node == cfg.distillation.n_gpus_per_node == 4
    if domain == "if":
        assert cfg.trainer.test_freq == -1
        assert not cfg.trainer.val_before_train
        assert cfg.actor_rollout_ref.rollout.max_model_len == 11265


def test_performance_controls_reach_student_and_teacher(tmp_path):
    cfg = _config(
        tmp_path,
        OPD_METRICS="True",
        ROLLOUT_ENFORCE_EAGER="False",
        TEACHER_ENFORCE_EAGER="False",
        PARAM_OFFLOAD="False",
        OPTIMIZER_OFFLOAD="False",
        GRADIENT_CHECKPOINTING="False",
        ROLLOUT_TOKEN_BUDGET="16384",
        TEACHER_TOKEN_BUDGET="16384",
    )
    assert cfg.trainer.v1.opd_metrics
    assert not cfg.actor_rollout_ref.rollout.enforce_eager
    assert not cfg.distillation.teacher_models.teacher_model.inference.enforce_eager
    assert not cfg.actor_rollout_ref.actor.fsdp_config.param_offload
    assert not cfg.actor_rollout_ref.actor.fsdp_config.optimizer_offload
    assert not cfg.actor_rollout_ref.model.enable_gradient_checkpointing
    assert cfg.actor_rollout_ref.rollout.max_num_batched_tokens == 16384
    assert cfg.distillation.teacher_models.teacher_model.inference.max_num_batched_tokens == 16384


def test_if_checker_environment_reaches_ray_workers(tmp_path):
    cfg = _config(
        tmp_path,
        script=ROOT / "examples/mirroropd/run_if_deepseek_r1_1_5b_fsdp2.sh",
        OPENCOMPASS_ROOT=str(tmp_path / "opencompass source"),
        NLTK_DATA=str(tmp_path / "nltk_data"),
    )
    variables = cfg.ray_kwargs.ray_init.runtime_env.env_vars
    assert variables.OPENCOMPASS_ROOT == str(tmp_path / "opencompass source")
    assert variables.NLTK_DATA == str(tmp_path / "nltk_data")


def test_if_rejects_single_turn_validation(tmp_path):
    result = _launch(tmp_path, script=ROOT / "examples/mirroropd/run_if_deepseek_r1_1_5b_fsdp2.sh", TEST_FREQ="20")
    assert result.returncode == 2
    assert "three-turn" in result.stderr


def test_if_checks_checker_source_before_starting_training(tmp_path):
    train = tmp_path / "data/if/MultiIF/train.parquet"
    train.parent.mkdir(parents=True)
    train.touch()
    result = _launch(
        tmp_path,
        script=ROOT / "examples/mirroropd/run_if_deepseek_r1_1_5b_fsdp2.sh",
        DRY_RUN="0",
        CONFIG_ONLY="0",
        CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
        OPENCOMPASS_ROOT=str(tmp_path / "missing"),
    )
    assert result.returncode == 2
    assert "OPENCOMPASS_ROOT" in result.stderr
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


def test_default_launch_uses_absolute_native_interpreter():
    """Default launch selection must not depend on activation or the ambient Python PATH."""
    env = os.environ.copy()
    for name in (
        "PYTHON_BIN",
        "VERL_USE_UV",
        "CUDA_VISIBLE_DEVICES",
        "MODEL_ROOT",
        "DATA_ROOT",
        "STUDENT_MODEL",
        "TEACHER_MODEL",
        "TRAIN_FILE",
        "MATH_FILE",
        "AMC_FILE",
        "AIME_FILE",
    ):
        env.pop(name, None)
    env["DRY_RUN"] = "1"
    result = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout)
    expected = str(ROOT / ".venv-native/bin/python")
    assert command[0] == expected
    assert f"ray_kwargs.ray_init.runtime_env.py_executable='{expected}'" in command
    assert f"actor_rollout_ref.model.path='{ROOT.parent / 'model/DeepSeek-R1-Distill-Qwen-1.5B'}'" in command
    assert any(str(ROOT.parent / "data/math/DAPO/train.parquet") in argument for argument in command)


def test_explicit_native_interpreter_reaches_ray_workers(tmp_path):
    """The Ray runtime override retains the requested absolute environment path."""
    interpreter = str(tmp_path / "custom-env/bin/python")
    cfg = _config(tmp_path, PYTHON_BIN=interpreter)
    assert cfg.ray_kwargs.ray_init.runtime_env.py_executable == interpreter


def test_launcher_fixes_offline_wandb_location_for_ray(tmp_path):
    """The training driver and Ray task runner share a checkout-relative offline log root."""
    cfg = _config(tmp_path, WANDB_DIR=str(tmp_path / "unrelated"), WANDB_MODE="offline")
    assert cfg.trainer.logger == ["console", "wandb"]
    assert cfg.ray_kwargs.ray_init.runtime_env.env_vars.WANDB_DIR == str(ROOT)
    assert cfg.ray_kwargs.ray_init.runtime_env.env_vars.WANDB_MODE == "offline"


def test_real_launch_requires_explicit_gpu_allocation(tmp_path):
    result = _launch(tmp_path, DRY_RUN="0", CONFIG_ONLY="0")
    assert result.returncode == 2
    assert "allocated to this run" in result.stderr
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("exit_code", [0, 23])
@pytest.mark.parametrize("domain", ["math", "code", "if"])
def test_source_archive_launch_saves_both_log_streams_and_training_exit_status(tmp_path, exit_code, domain):
    """An unpacked tree logs stdout/stderr while retaining the trainer's success or failure."""
    checkout = tmp_path / "source checkout"
    entry = SCRIPT if domain == "math" else SCRIPT.with_name(f"run_{domain}_deepseek_r1_1_5b_fsdp2.sh")
    script = checkout / "examples/mirroropd" / entry.name
    script.parent.mkdir(parents=True)
    shutil.copy2(entry, script)
    shutil.copy2(SCRIPT.parent / "_run_single_teacher_fsdp2.sh", script.parent)
    datasets = {
        "math": ["math/DAPO/train", "math/MATH-500/test", "math/AMC23/test", "math/AIME25/test"],
        "code": ["code/TACO/train", "code/LCBv6/test"],
        "if": ["if/MultiIF/train"],
    }
    for name in datasets[domain]:
        path = tmp_path / "data" / f"{name}.parquet"
        path.parent.mkdir(parents=True)
        path.touch()
    checker = tmp_path / "checker source"
    (checker / "opencompass/datasets/MultiIF").mkdir(parents=True)
    interpreter = tmp_path / "mock-python"
    interpreter.write_text(
        '#!/bin/sh\ncase " $* " in *" -c "*) exit 0 ;; *" --cfg "*) printf "mock: true\\n"; exit 0 ;; esac\n'
        'printf "mock training stdout\\n"\nprintf "mock training stderr\\n" >&2\n'
        f"exit {exit_code}\n"
    )
    interpreter.chmod(0o755)
    result = _launch(
        tmp_path,
        script=script,
        DRY_RUN="0",
        CONFIG_ONLY="0",
        PYTHON_BIN=str(interpreter),
        CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
        OPENCOMPASS_ROOT=str(checker),
    )
    assert result.returncode == exit_code, result.stderr
    assert "source archive without Git metadata" in (tmp_path / "run/source_commit.txt").read_text()
    assert (tmp_path / "run/source_changes.patch").read_bytes() == b""
    assert "verl.trainer.main_ppo" in (tmp_path / "run/command.sh").read_text()
    log = (tmp_path / "run/train.log").read_text()
    for message in ("mock training stdout", "mock training stderr"):
        assert message in log
        assert message in result.stdout


@pytest.mark.parametrize("domain", ["math", "code", "if"])
def test_hope_submission_and_worker_preview_match_eight_gpu_allocation(tmp_path, domain):
    """Exercise the real Make submission and worker config while stubbing only the Hope CLI."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    hope = fake_bin / "hope"
    record = tmp_path / "submission.txt"
    hope.write_text('#!/bin/sh\nprintf "%s\\n" "$PWD" "$@" > "$HOPE_SUBMIT_RECORD"\n')
    hope.chmod(0o755)
    env = os.environ.copy()
    env.update(PATH=str(fake_bin) + os.pathsep + env["PATH"], HOPE_SUBMIT_RECORD=str(record))
    submitted = subprocess.run(["make", "opd-submit"], cwd=ROOT, env=env, capture_output=True, text=True)
    assert submitted.returncode == 0, submitted.stderr
    directory, action, filename = record.read_text().splitlines()
    assert action == "run"
    config = ConfigParser(interpolation=None)
    config.read(Path(directory) / filename)
    assert config.getint("roles", "workers") == 1
    allocated_gpus = config.getint("roles", "worker.gcores80g")

    preview = _launch(
        tmp_path,
        script=ROOT / "hope/mirroropd/run.sh",
        DOMAIN=domain,
        CUDA_VISIBLE_DEVICES="GPU-0,GPU-1,GPU-2,GPU-3,GPU-4,GPU-5,GPU-6,GPU-7",
        VERL_USE_UV="1",
    )
    assert preview.returncode == 0, preview.stderr
    command = shlex.split(preview.stdout)
    native_python = str(ROOT / ".venv-native/bin/python")
    assert command[0] == native_python
    with initialize_config_dir(config_dir=str(ROOT / "verl/trainer/config"), version_base=None):
        cfg = compose(config_name="ppo_trainer", overrides=command[command.index("verl.trainer.main_ppo") + 1 :])
    assert cfg.trainer.n_gpus_per_node + cfg.distillation.n_gpus_per_node == allocated_gpus == 8
    assert cfg.ray_kwargs.ray_init.runtime_env.py_executable == native_python
    assert cfg.ray_kwargs.ray_init.runtime_env.env_vars.WANDB_MODE == "offline"
    teachers = {"math": "JustRL-DeepSeek-1.5B", "code": "DeepCoder-1.5B-Preview", "if": "UltraData-IF-1.5B"}
    assert cfg.distillation.teacher_models.teacher_model.model_path.endswith(teachers[domain])
    assert not (tmp_path / "run").exists()
