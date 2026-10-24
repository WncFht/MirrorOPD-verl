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


@pytest.mark.parametrize("domain", ["math", "code", "if"])
def test_full_reverse_kl_launcher(tmp_path, domain):
    cfg = _config(
        tmp_path,
        script=SCRIPT.parent / "run_full_reverse_kl.sh",
        DOMAIN=domain,
        FULL_KL_BACKEND="streaming",
        FULL_KL_ASYNC_WRITE="True",
        FULL_KL_PREFETCH="True",
    )
    OmegaConf.resolve(cfg)
    validate_config(cfg, use_reference_policy=False, use_critic=False)
    loss = omega_conf_to_dataclass(cfg.distillation).distillation_loss
    assert loss.loss_mode == "full_reverse_kl" and not loss.use_policy_gradient
    assert loss.full_vocab.backend == "streaming"
    assert loss.full_vocab.async_write and loss.full_vocab.prefetch
    assert not cfg.actor_rollout_ref.rollout.calculate_log_probs


def test_full_reverse_kl_smoke_configuration(tmp_path):
    cfg = _config(tmp_path, script=SCRIPT.parent / "smoke/smoke_full_reverse_kl.sh")
    OmegaConf.resolve(cfg)
    validate_config(cfg, use_reference_policy=False, use_critic=False)
    assert cfg.trainer.total_training_steps == 2
    assert cfg.trainer.n_gpus_per_node == cfg.ray_kwargs.ray_init.num_gpus == 4
    assert cfg.distillation.distillation_loss.full_vocab.backend == "precompute"
    assert cfg.actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config.backend == "TRITON_ATTN"
    assert cfg.distillation.distillation_loss.full_vocab.async_write
    assert cfg.distillation.distillation_loss.full_vocab.prefetch


def _launch(tmp_path, script=SCRIPT, command=None, **overrides):
    env = os.environ.copy()
    for name in (
        "CUDA_VISIBLE_DEVICES",
        "STUDENT_GPUS",
        "TEACHER_GPUS",
        "ROLLOUT_TP",
        "TEACHER_TP",
        "DOMAIN",
        "DISTILLATION_EXECUTION_MODE",
        "TEACHER_FSDP_TOKEN_BUDGET",
        "TEACHER_FSDP_DTYPE",
        "ROLLOUT_ENFORCE_EAGER",
        "ROLLOUT_MAX_NUM_SEQS",
        "ROLLOUT_GPU_MEMORY_UTILIZATION",
        "ROLLOUT_TOKEN_BUDGET",
        "VAL_BATCH_SIZE",
        "PPO_EPOCHS",
        "TOTAL_TRAINING_STEPS",
        "TOTAL_EPOCHS",
        "STUDENT_MODEL",
        "TEACHER_MODEL",
        "TEST_FREQ",
        "SAVE_FREQ",
        "LEARNING_RATE",
        "SEED",
        "FULL_KL_BACKEND",
        "FULL_KL_TOKEN_CHUNK",
        "FULL_KL_VOCAB_CHUNK",
        "FULL_KL_ASYNC_WRITE",
        "FULL_KL_PREFETCH",
    ):
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
    return subprocess.run(command or ["bash", str(script)], env=env, capture_output=True, text=True)


def _config(tmp_path, **overrides):
    result = _launch(tmp_path, **overrides)
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout)
    with initialize_config_dir(config_dir=str(ROOT / "verl/trainer/config"), version_base=None):
        return compose(config_name="ppo_trainer", overrides=command[command.index("verl.trainer.main_ppo") + 1 :])


def test_launcher_composes_native_single_teacher_baseline(tmp_path):
    cfg = _config(tmp_path, DISTILLATION_EXECUTION_MODE="standalone")
    OmegaConf.resolve(cfg)
    validate_config(cfg, use_reference_policy=False, use_critic=False)
    assert cfg.trainer.use_v1 and cfg.trainer.v1.trainer_mode == "sync"
    assert "opd_metrics" not in cfg.trainer.v1
    assert cfg.trainer.n_gpus_per_node == 4 and cfg.distillation.n_gpus_per_node == 4
    assert cfg.actor_rollout_ref.actor.strategy == "fsdp2"
    assert cfg.data.train_batch_size == cfg.actor_rollout_ref.actor.ppo_mini_batch_size == 64
    assert cfg.data.continuous_token.model_family == "deepseek"
    assert cfg.actor_rollout_ref.actor.ppo_epochs == 1
    assert cfg.actor_rollout_ref.rollout.response_length == 7168
    assert cfg.actor_rollout_ref.rollout.val_kwargs.max_tokens == 31744
    assert cfg.actor_rollout_ref.rollout.val_kwargs.n == 16
    assert cfg.actor_rollout_ref.rollout.max_model_len == 35840
    assert cfg.distillation.teacher_models.teacher_model.inference.max_model_len == 11265
    assert cfg.distillation.distillation_loss.loss_mode == "k1"
    assert not cfg.distillation.distillation_loss.use_task_rewards
    assert cfg.distillation.distillation_loss.loss_max_clamp is None
    assert [Path(p).parent.name for p in cfg.data.val_files] == ["MATH-500", "AMC23", "AIME25"]
    assert cfg.actor_rollout_ref.model.path.endswith("models with spaces/DeepSeek-R1-Distill-Qwen-1.5B")
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("gpus", [4, 8])
@pytest.mark.parametrize("entry", ["run_math_deepseek_r1_1_5b_fsdp2.sh", "launchers/_run_single_teacher_fsdp2.sh"])
def test_default_graph_and_validation_window_keep_training_budget(tmp_path, gpus, entry):
    cfg = _config(tmp_path, script=SCRIPT.parent / entry, STUDENT_GPUS=str(gpus))
    assert not cfg.actor_rollout_ref.rollout.enforce_eager
    assert list(cfg.actor_rollout_ref.rollout.cudagraph_capture_sizes) == [1, 2, 4, 8, 16, 32, 64, 128]
    assert cfg.actor_rollout_ref.rollout.max_num_seqs == 128
    assert cfg.actor_rollout_ref.rollout.gpu_memory_utilization == 0.65
    assert cfg.actor_rollout_ref.rollout.max_num_batched_tokens == 8192
    assert cfg.data.val_batch_size is None
    assert cfg.trainer.n_gpus_per_node == gpus
    assert cfg.distillation.n_gpus_per_node == 0
    assert cfg.actor_rollout_ref.rollout.val_kwargs.n == 16
    assert cfg.data.train_batch_size == cfg.actor_rollout_ref.actor.ppo_mini_batch_size == 64


def test_validation_environment_and_final_hydra_overrides(tmp_path):
    cfg = _config(tmp_path, VAL_BATCH_SIZE="null", ROLLOUT_MAX_NUM_SEQS="96", ROLLOUT_GPU_MEMORY_UTILIZATION="0.5")
    assert cfg.data.val_batch_size is None
    assert cfg.actor_rollout_ref.rollout.max_num_seqs == 96
    assert cfg.actor_rollout_ref.rollout.gpu_memory_utilization == 0.5
    assert cfg.actor_rollout_ref.rollout.cudagraph_capture_sizes[-1] == 96
    cfg = _config(
        tmp_path,
        command=["bash", str(SCRIPT), "data.val_batch_size=64", "actor_rollout_ref.rollout.gpu_memory_utilization=0.6"],
        VAL_BATCH_SIZE="null",
        ROLLOUT_GPU_MEMORY_UTILIZATION="0.5",
    )
    assert cfg.data.val_batch_size == 64
    assert cfg.actor_rollout_ref.rollout.gpu_memory_utilization == 0.6


@pytest.mark.parametrize("window", ["0", "-1", "1.5", "None", "false"])
def test_invalid_validation_window_is_rejected_before_startup(tmp_path, window):
    result = _launch(tmp_path, VAL_BATCH_SIZE=window)
    assert result.returncode == 2
    assert "VAL_BATCH_SIZE must be null or a positive integer" in result.stderr


def test_graph_capture_covers_concurrency_override_and_eager_fallback(tmp_path):
    cfg = _config(tmp_path, ROLLOUT_MAX_NUM_SEQS="64", ROLLOUT_ENFORCE_EAGER="True", VAL_BATCH_SIZE="16")
    assert cfg.actor_rollout_ref.rollout.enforce_eager
    assert list(cfg.actor_rollout_ref.rollout.cudagraph_capture_sizes) == [1, 2, 4, 8, 16, 32, 64]
    assert cfg.data.val_batch_size == 16


def test_zero_rollout_concurrency_is_rejected_before_capture_derivation(tmp_path):
    result = _launch(tmp_path, ROLLOUT_MAX_NUM_SEQS="0")
    assert result.returncode == 2
    assert "ROLLOUT_MAX_NUM_SEQS must be a positive integer" in result.stderr


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
    assert cfg.trainer.n_gpus_per_node == 8 and cfg.distillation.n_gpus_per_node == 0
    assert cfg.data.val_batch_size == 32
    assert cfg.actor_rollout_ref.rollout.max_num_seqs == 32
    assert cfg.actor_rollout_ref.rollout.gpu_memory_utilization == 0.5
    if domain == "if":
        assert cfg.trainer.test_freq == -1
        assert not cfg.trainer.val_before_train
        assert cfg.actor_rollout_ref.rollout.max_model_len == 11264


def test_performance_controls_reach_student_and_teacher(tmp_path):
    cfg = _config(
        tmp_path,
        DISTILLATION_EXECUTION_MODE="standalone",
        ROLLOUT_ENFORCE_EAGER="False",
        TEACHER_ENFORCE_EAGER="False",
        PARAM_OFFLOAD="False",
        OPTIMIZER_OFFLOAD="False",
        GRADIENT_CHECKPOINTING="False",
        ROLLOUT_TOKEN_BUDGET="16384",
        TEACHER_TOKEN_BUDGET="16384",
    )
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
        DISTILLATION_EXECUTION_MODE="standalone",
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
        {"DISTILLATION_EXECUTION_MODE": "standalone", "TEACHER_TP": "3"},
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


def test_qwen3_base_32k_validation_window_fits_exactly(tmp_path):
    cfg = _config(tmp_path, MAX_PROMPT_LENGTH="1024", MAX_VAL_RESPONSE_LENGTH="31744")
    assert cfg.actor_rollout_ref.rollout.max_model_len == 32768
    assert cfg.distillation.teacher_models.teacher_model.inference.max_model_len == 8193


def test_validation_budget_rejects_default_multiturn_agent(tmp_path):
    cfg = _config(tmp_path)
    cfg.actor_rollout_ref.rollout.agent.default_agent_loop = "tool_agent"
    with pytest.raises(ValueError, match="single_turn_agent"):
        validate_config(cfg, use_reference_policy=False, use_critic=False)


def test_validation_budget_rejects_backend_that_overwrites_request_limit(tmp_path):
    cfg = _config(tmp_path, DISTILLATION_EXECUTION_MODE="standalone")
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
@pytest.mark.parametrize("append_source", [False, True])
def test_source_archive_launch_saves_both_log_streams_and_training_exit_status(
    tmp_path, exit_code, domain, append_source
):
    """An unpacked tree logs stdout/stderr while retaining the trainer's success or failure."""
    checkout = tmp_path / "source checkout"
    entry = SCRIPT if domain == "math" else SCRIPT.with_name(f"run_{domain}_deepseek_r1_1_5b_fsdp2.sh")
    script = checkout / "examples/mirroropd" / entry.name
    script.parent.mkdir(parents=True)
    shutil.copy2(entry, script)
    (script.parent / "launchers").mkdir()
    shutil.copy2(SCRIPT.parent / "launchers/_run_single_teacher_fsdp2.sh", script.parent / "launchers")
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
        # A running shell may resume reading the file after its child exits.
        # Updating the launcher must not execute newly appended source text.
        'if [ "$MUTATE_LAUNCHER" = 1 ]; then\n'
        '  printf \'\\nprintf "unexpected" > "$RUN_DIR/unexpected"\\n\' >> "$LAUNCHER_TO_CHANGE"\n'
        "fi\n"
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
        MUTATE_LAUNCHER="1" if append_source else "0",
        LAUNCHER_TO_CHANGE=str(script.parent / "launchers/_run_single_teacher_fsdp2.sh"),
    )
    assert result.returncode == exit_code, result.stderr
    assert not (tmp_path / "run/unexpected").exists()
    assert "source archive without Git metadata" in (tmp_path / "run/source_commit.txt").read_text()
    assert (tmp_path / "run/source_changes.patch").read_bytes() == b""
    assert "verl.trainer.main_ppo" in (tmp_path / "run/command.sh").read_text()
    log = (tmp_path / "run/train.log").read_text()
    for message in ("mock training stdout", "mock training stderr"):
        assert message in log
        assert message in result.stdout


def _local_hope_command(worker_command=None):
    if worker_command is None:
        config = ConfigParser(interpolation=None)
        config.read(ROOT / "hope/mirroropd/mirroropd.hope")
        worker_command = config.get("roles", "worker.script")
    remote_root = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/MirrorOPD-verl"
    assert remote_root in worker_command
    return shlex.split(worker_command.replace(remote_root, str(ROOT)))


@pytest.mark.parametrize("devices,expected_code", [(None, 0), ("GPU-a,GPU-b", 2)])
def test_direct_hope_command_preserves_allocation_or_uses_job_local_fallback(tmp_path, devices, expected_code):
    overrides = {} if devices is None else {"CUDA_VISIBLE_DEVICES": devices}
    result = _launch(tmp_path, command=_local_hope_command(), **overrides)
    assert result.returncode == expected_code, result.stderr
    if expected_code:
        assert "CUDA_VISIBLE_DEVICES does not cover" in result.stderr


@pytest.mark.parametrize("domain", ["math", "code", "if"])
@pytest.mark.parametrize("mode", ["shared_sync", "standalone"])
def test_hope_submission_and_worker_preview_match_eight_gpu_allocation(tmp_path, domain, mode):
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

    # Execute the configured worker command itself, translating only the remote mount.
    worker_command = config.get("roles", "worker.script")
    assert "hope/mirroropd/run.sh" not in worker_command
    assert "STUDENT_GPUS=" not in worker_command and "TEACHER_GPUS=" not in worker_command
    preview = _launch(
        tmp_path,
        command=_local_hope_command(worker_command),
        PYTHON_BIN="",
        DISTILLATION_EXECUTION_MODE=mode,
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
    assert cfg.distillation.execution_mode == mode
    assert cfg.trainer.n_gpus_per_node == (8 if mode == "shared_sync" else 4)
    assert cfg.distillation.n_gpus_per_node == (0 if mode == "shared_sync" else 4)
    assert cfg.ray_kwargs.ray_init.runtime_env.py_executable == native_python
    assert cfg.ray_kwargs.ray_init.runtime_env.env_vars.WANDB_MODE == "offline"
    teachers = {"math": "JustRL-DeepSeek-1.5B", "code": "DeepCoder-1.5B-Preview", "if": "UltraData-IF-1.5B"}
    assert cfg.distillation.teacher_models.teacher_model.model_path.endswith(teachers[domain])
    assert not (tmp_path / "run").exists()


def test_default_shared_pipeline_and_teacher_forward_controls(tmp_path):
    cfg = _config(tmp_path, TEACHER_FSDP_TOKEN_BUDGET="16384", TEACHER_FSDP_DTYPE="bfloat16")
    OmegaConf.resolve(cfg)
    validate_config(cfg, use_reference_policy=False, use_critic=False)
    distillation = omega_conf_to_dataclass(cfg.distillation)
    assert distillation.execution_mode == "shared_sync"
    assert cfg.trainer.n_gpus_per_node == 8
    assert distillation.n_gpus_per_node == distillation.nnodes == 0
    assert distillation.shared_teacher.infer_max_token_len_per_gpu == 16384
    assert distillation.shared_teacher.model_dtype == "bfloat16"
    assert distillation.shared_teacher.mixed_precision["param_dtype"] == "bfloat16"
    assert distillation.teacher_models["default"].num_replicas == 0
    # Shared configuration does not rewrite the vLLM prompt/response budget.
    assert distillation.teacher_models["default"].inference.response_length == 7168


@pytest.mark.parametrize(
    "overrides",
    [
        {"DISTILLATION_EXECUTION_MODE": "invalid"},
        {"TEACHER_GPUS": "4"},
        {"ROLLOUT_TP": "2"},
        {"PARAM_OFFLOAD": "False"},
        {"OPTIMIZER_OFFLOAD": "False"},
        {"TEACHER_FSDP_TOKEN_BUDGET": "8192"},
    ],
)
def test_shared_launcher_rejects_incompatible_settings(tmp_path, overrides):
    assert _launch(tmp_path, **overrides).returncode == 2


@pytest.mark.parametrize(
    "group,source,mode",
    [
        ("E1", "rollout", "token_ppo"),
        ("E2", "rollout", "token_is"),
        ("E3", "current", "token_ppo"),
        ("E4", "current", "token_is"),
        ("E5", "current", "reinforce"),
        ("E6", "rollout", "reinforce"),
        ("E7", "rollout", "token_ppo"),
        ("E8", "rollout", "token_is"),
        ("E9", "current", "token_ppo"),
        ("E10", "current", "token_is"),
        ("E11", "current", "reinforce"),
        ("E12", "rollout", "reinforce"),
    ],
)
@pytest.mark.parametrize("execution_mode", ["shared_sync", "standalone"])
def test_ablation_launcher_composes_supported_modes(tmp_path, group, source, mode, execution_mode):
    script = ROOT / "examples/mirroropd/run_math_opd_ablation_fsdp2.sh"
    result = _launch(tmp_path, command=["bash", str(script), group], DISTILLATION_EXECUTION_MODE=execution_mode)
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout)
    with initialize_config_dir(config_dir=str(ROOT / "verl/trainer/config"), version_base=None):
        cfg = compose(config_name="ppo_trainer", overrides=command[command.index("verl.trainer.main_ppo") + 1 :])
    OmegaConf.resolve(cfg)
    validate_config(cfg, use_reference_policy=False, use_critic=False)
    assert cfg.distillation.distillation_loss.advantage_source == source
    assert cfg.distillation.distillation_loss.policy_loss_mode == mode
    if int(group[1:]) <= 6:
        assert cfg.actor_rollout_ref.actor.ppo_epochs == 4
        assert cfg.trainer.total_training_steps == cfg.trainer.save_freq == cfg.trainer.test_freq == 32
    else:
        assert cfg.actor_rollout_ref.actor.ppo_epochs == 16
        assert cfg.trainer.total_training_steps == 200
        assert cfg.trainer.test_freq == 50
        assert cfg.trainer.save_freq == 100
    assert not cfg.trainer.val_before_train
    assert cfg.trainer.v1.trainer_mode == "sync"
    assert cfg.trainer.v1.sync.get("parameter_sync_step", 1) == 1
    assert cfg.actor_rollout_ref.actor.optim.lr_warmup_steps == 0
    assert not cfg.actor_rollout_ref.actor.checkpoint.async_save


@pytest.mark.parametrize("group", ["E1", "E10"])
def test_ablation_launcher_accepts_budget_clip_epochs_and_final_overrides(tmp_path, group):
    script = ROOT / "examples/mirroropd/run_math_opd_ablation_fsdp2.sh"
    result = _launch(
        tmp_path,
        command=["bash", str(script), group, "trainer.total_training_steps=7"],
        TOTAL_TRAINING_STEPS="200",
        PPO_EPOCHS="2",
        TEST_FREQ="200",
        SAVE_FREQ="200",
        OPD_CLIP_EPS="0.1",
    )
    command = shlex.split(result.stdout)
    with initialize_config_dir(config_dir=str(ROOT / "verl/trainer/config"), version_base=None):
        cfg = compose(config_name="ppo_trainer", overrides=command[command.index("verl.trainer.main_ppo") + 1 :])
    assert cfg.trainer.total_training_steps == 7
    assert cfg.trainer.save_freq == cfg.trainer.test_freq == 200
    assert cfg.actor_rollout_ref.actor.ppo_epochs == 2
    loss = cfg.distillation.distillation_loss
    assert loss.clip_ratio == loss.clip_ratio_low == loss.clip_ratio_high == 0.1


@pytest.mark.parametrize(
    "group,lr,epochs",
    [("E13", 1e-6, 1), ("E14", 5e-7, 1), ("E15", 1e-6, 1), ("E16", 1e-6, 4), ("E17", 1e-6, 16)],
)
def test_full_kl_and_vanilla_presets(tmp_path, group, lr, epochs):
    script = SCRIPT.parent / "run_math_opd_ablation_fsdp2.sh"
    cfg = _config(tmp_path, command=["bash", str(script), group], DOMAIN="code")
    OmegaConf.resolve(cfg)
    validate_config(cfg, use_reference_policy=False, use_critic=False)
    distillation = omega_conf_to_dataclass(cfg.distillation)
    loss = distillation.distillation_loss
    assert cfg.trainer.n_gpus_per_node == 8 and distillation.execution_mode == "shared_sync"
    assert cfg.data.train_batch_size == cfg.actor_rollout_ref.actor.ppo_mini_batch_size == 64
    assert cfg.actor_rollout_ref.actor.ppo_epochs == epochs
    assert cfg.trainer.total_training_steps == 200
    assert cfg.trainer.test_freq == 20 and cfg.trainer.save_freq == 100
    assert cfg.trainer.max_actor_ckpt_to_keep == 2 and not cfg.trainer.val_before_train
    assert cfg.actor_rollout_ref.actor.optim.lr == pytest.approx(lr)
    assert cfg.actor_rollout_ref.actor.optim.lr_scheduler_type == "constant"
    assert cfg.actor_rollout_ref.actor.optim.lr_warmup_steps == 0
    assert cfg.data.seed == cfg.actor_rollout_ref.rollout.seed == cfg.actor_rollout_ref.actor.fsdp_config.seed == 42
    assert cfg.actor_rollout_ref.actor.data_loader_seed == 42
    assert cfg.data.max_response_length == 7168 and cfg.actor_rollout_ref.rollout.val_kwargs.n == 16
    assert cfg.distillation.teacher_models.teacher_model.model_path.endswith("JustRL-DeepSeek-1.5B")
    assert loss.advantage_source == "current" and loss.policy_loss_mode == "vanilla"
    assert not loss.use_task_rewards
    assert cfg.algorithm.rollout_correction.rollout_is is None
    assert not cfg.algorithm.rollout_correction.bypass_mode
    if group == "E15":
        assert loss.loss_mode == "k1" and loss.use_policy_gradient
        assert cfg.actor_rollout_ref.rollout.calculate_log_probs
    else:
        assert loss.loss_mode == "full_reverse_kl" and not loss.use_policy_gradient
        assert loss.full_vocab.backend == "precompute"
        assert loss.full_vocab.async_write and loss.full_vocab.prefetch
        assert loss.full_vocab.token_chunk_size == 512 and loss.full_vocab.vocab_chunk_size == 8192
        assert not cfg.actor_rollout_ref.rollout.calculate_log_probs


@pytest.mark.parametrize("group,field", [("E14", "optim.lr"), ("E16", "ppo_epochs"), ("E17", "ppo_epochs")])
def test_full_kl_pairs_only_change_the_ablation_factor(tmp_path, group, field):
    script = SCRIPT.parent / "run_math_opd_ablation_fsdp2.sh"
    first = _config(tmp_path, command=["bash", str(script), "E13"])
    second = _config(tmp_path, command=["bash", str(script), group])
    key = "actor_rollout_ref.actor." + field
    OmegaConf.update(second, key, OmegaConf.select(first, key))
    assert OmegaConf.to_container(first, resolve=True) == OmegaConf.to_container(second, resolve=True)


@pytest.mark.parametrize("group", ["E13", "E14", "E15", "E16", "E17"])
def test_full_kl_and_vanilla_presets_preserve_explicit_overrides(tmp_path, group):
    script = SCRIPT.parent / "run_math_opd_ablation_fsdp2.sh"
    cfg = _config(
        tmp_path,
        command=["bash", str(script), group, "actor_rollout_ref.actor.optim.lr=8e-7", "trainer.test_freq=7"],
        LEARNING_RATE="3e-7",
        PPO_EPOCHS="2",
        TOTAL_TRAINING_STEPS="12",
        TEST_FREQ="6",
        SAVE_FREQ="12",
        SEED="43",
        FULL_KL_BACKEND="streaming",
        FULL_KL_ASYNC_WRITE="False",
        FULL_KL_PREFETCH="False",
    )
    assert cfg.actor_rollout_ref.actor.optim.lr == pytest.approx(8e-7)
    assert cfg.actor_rollout_ref.actor.ppo_epochs == 2 and cfg.trainer.total_training_steps == 12
    assert cfg.trainer.test_freq == 7 and cfg.trainer.save_freq == 12
    assert cfg.data.seed == cfg.actor_rollout_ref.rollout.seed == cfg.actor_rollout_ref.actor.fsdp_config.seed == 43
    if group != "E15":
        assert cfg.distillation.distillation_loss.full_vocab.backend == "streaming"
        assert not cfg.distillation.distillation_loss.full_vocab.async_write
        assert not cfg.distillation.distillation_loss.full_vocab.prefetch


@pytest.mark.parametrize("group", ["E13", "E14", "E16", "E17"])
def test_full_kl_presets_require_shared_teacher(tmp_path, group):
    result = _launch(
        tmp_path,
        command=["bash", str(SCRIPT.parent / "run_math_opd_ablation_fsdp2.sh"), group],
        DISTILLATION_EXECUTION_MODE="standalone",
    )
    assert result.returncode == 2 and "shared_sync" in result.stderr


def test_shared_launcher_remains_vanilla_and_validates_epoch_input(tmp_path):
    cfg = _config(tmp_path)
    assert cfg.distillation.distillation_loss.policy_loss_mode == "vanilla"
    assert cfg.distillation.distillation_loss.advantage_source == "current"
    assert cfg.actor_rollout_ref.actor.ppo_epochs == 1
    result = _launch(tmp_path, PPO_EPOCHS="0")
    assert result.returncode == 2 and "PPO_EPOCHS" in result.stderr


def test_code_launcher_forwards_lcb_policy_and_audit_directory(tmp_path):
    script = ROOT / "examples/mirroropd/run_code_deepseek_r1_1_5b_fsdp2.sh"
    cfg = _config(tmp_path, script=script, MIRROROPD_LCB_MEMORY_MIB="2048", MIRROROPD_LCB_MAX_CONCURRENT="2")
    env = cfg.ray_kwargs.ray_init.runtime_env.env_vars
    assert env.MIRROROPD_LCB_MEMORY_MIB == "2048"
    assert env.MIRROROPD_LCB_MAX_CONCURRENT == "2"
    assert env.MIRROROPD_LCB_RUN_ID == "test"
    assert env.MIRROROPD_LCB_AUDIT_DIR == str(tmp_path / "run" / "lcb-audit")


@pytest.mark.parametrize(
    "group,epochs,outer,test,save",
    [
        ("E18", 64, 64, 8, 16),
        ("E19", 32, 128, 16, 32),
        ("E20", 16, 256, 32, 64),
    ],
)
def test_equal_update_presets_use_one_original_data_pass(tmp_path, group, epochs, outer, test, save):
    cfg = _config(tmp_path, command=["bash", str(SCRIPT.parent / "run_math_opd_ablation_fsdp2.sh"), group])
    OmegaConf.resolve(cfg)
    validate_config(cfg, use_reference_policy=False, use_critic=False)
    loss = omega_conf_to_dataclass(cfg.distillation).distillation_loss
    assert cfg.data.train_files[0].endswith("/math/DAPO/train.parquet")
    assert cfg.data.train_batch_size == cfg.actor_rollout_ref.actor.ppo_mini_batch_size == 64
    assert cfg.data.train_max_samples == -1 and cfg.data.shuffle and cfg.data.seed == 42
    assert cfg.trainer.total_epochs == 1 and cfg.trainer.total_training_steps == outer
    assert cfg.actor_rollout_ref.actor.ppo_epochs == epochs and outer * epochs == 4096
    assert cfg.trainer.test_freq == test and test * epochs == 512
    assert cfg.trainer.save_freq == save and save * epochs == 1024
    assert cfg.trainer.max_actor_ckpt_to_keep == 4
    assert not cfg.trainer.val_before_train and cfg.trainer.resume_mode == "disable"
    assert not cfg.algorithm.filter_groups.enable and not cfg.trainer.v1.sampler.sync_refill_failed_groups
    assert outer * 64 <= 17917
    assert loss.loss_mode == "full_reverse_kl" and not loss.use_task_rewards
    assert loss.full_vocab.backend == "precompute" and loss.full_vocab.async_write and loss.full_vocab.prefetch
    assert cfg.actor_rollout_ref.actor.optim.lr == 1e-6


@pytest.mark.parametrize(
    "target,model",
    [
        ("student", "DeepSeek-R1-Distill-Qwen-1.5B"),
        ("teacher", "JustRL-DeepSeek-1.5B"),
    ],
)
def test_standalone_validation_uses_selected_generator_without_distillation(tmp_path, target, model):
    cfg = _config(tmp_path, command=["bash", str(SCRIPT.parent / "run_math_validation_fsdp2.sh"), target])
    OmegaConf.resolve(cfg)
    validate_config(cfg, use_reference_policy=False, use_critic=False)
    assert cfg.actor_rollout_ref.model.path.endswith("/" + model)
    assert not cfg.distillation.enabled and cfg.distillation.n_gpus_per_node == 0
    assert cfg.trainer.val_only and cfg.trainer.val_before_train
    assert cfg.trainer.resume_mode == "disable" and cfg.trainer.save_freq == -1
    assert not cfg.actor_rollout_ref.rollout.calculate_log_probs
    assert cfg.actor_rollout_ref.rollout.seed == cfg.data.seed == 42
    assert cfg.actor_rollout_ref.rollout.val_kwargs.n == 16
    assert cfg.actor_rollout_ref.rollout.val_kwargs.max_tokens == 31744
    assert cfg.data.val_batch_size is None
    assert cfg.actor_rollout_ref.rollout.max_num_seqs == 128
    assert cfg.actor_rollout_ref.rollout.gpu_memory_utilization == 0.65
    assert not cfg.actor_rollout_ref.rollout.enforce_eager


def test_teacher_validation_selects_teacher_override_and_retains_hydra_precedence(tmp_path):
    script = SCRIPT.parent / "run_math_validation_fsdp2.sh"
    cfg = _config(
        tmp_path,
        command=["bash", str(script), "teacher"],
        STUDENT_MODEL="/unused/student",
        TEACHER_MODEL="/teacher with spaces",
        STUDENT_GPUS="4",
    )
    assert cfg.actor_rollout_ref.model.path == "/teacher with spaces"
    assert cfg.trainer.n_gpus_per_node == 4
    cfg = _config(
        tmp_path,
        command=[
            "bash",
            str(script),
            "student",
            "actor_rollout_ref.model.path=/checkpoint",
            "actor_rollout_ref.rollout.val_kwargs.n=2",
        ],
        SEED="43",
    )
    assert cfg.actor_rollout_ref.model.path == "/checkpoint"
    assert cfg.actor_rollout_ref.rollout.val_kwargs.n == 2 and cfg.actor_rollout_ref.rollout.seed == 43
    result = _launch(tmp_path, command=["bash", str(script), "unknown"])
    assert result.returncode == 2


@pytest.mark.parametrize("group", ["E18", "E19", "E20"])
def test_equal_update_presets_allow_explicit_smoke_overrides(tmp_path, group):
    cfg = _config(
        tmp_path,
        command=["bash", str(SCRIPT.parent / "run_math_opd_ablation_fsdp2.sh"), group, "trainer.test_freq=1"],
        PPO_EPOCHS="2",
        TOTAL_TRAINING_STEPS="2",
        TRAIN_BATCH_SIZE="8",
        TEST_FREQ="2",
        SAVE_FREQ="-1",
    )
    assert cfg.actor_rollout_ref.actor.ppo_epochs == cfg.trainer.total_training_steps == 2
    assert cfg.data.train_batch_size == cfg.actor_rollout_ref.actor.ppo_mini_batch_size == 8
    assert cfg.trainer.test_freq == 1 and cfg.trainer.save_freq == -1
