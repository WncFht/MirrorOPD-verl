# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Existing native full-vocabulary launch defaults and config-only generation."""

import os
import subprocess

import yaml


def environment(repo, source, study, python, devices=(0, 1)):
    env = dict(os.environ)
    # Never inherit unrelated experiment overrides from the launching session.
    for key in list(env):
        if key.startswith(("PREFIX_", "FUSION_", "QWEN_")) or key in {
            "CONFIG_ONLY",
            "DRY_RUN",
            "TRAIN_BATCH_SIZE",
            "PPO_EPOCHS",
            "TOTAL_TRAINING_STEPS",
            "STUDENT_MODEL",
            "TEACHER_MODEL",
            "TRAIN_FILE",
            "RUN_DIR",
            "CHECKPOINT_DIR",
            "EXPERIMENT_NAME",
        }:
            env.pop(key)
    env.update(
        PYTHON_BIN=str(python),
        PYTHONPATH=str(source),
        MIRROROPD_WORKSPACE=str(repo.parent),
        MODEL_ROOT=str(repo.parent / "model"),
        DATA_ROOT=str(repo.parent / "data"),
        QWEN_EVAL_ROOT=str(study),
        STUDENT_GPUS=str(len(devices)),
        CUDA_VISIBLE_DEVICES=",".join(map(str, devices)),
        TRAIN_BATCH_SIZE="128",
        PPO_EPOCHS="16",
        TOTAL_TRAINING_STEPS="100",
        LEARNING_RATE="1e-6",
        SEED="42",
        TOTAL_EPOCHS="1",
        MAX_PROMPT_LENGTH="4096",
        MAX_RESPONSE_LENGTH="7168",
        MAX_VAL_RESPONSE_LENGTH="31744",
        FULL_KL_BACKEND="precompute_fused",
        FULL_KL_TERMINATION_TOKEN_IDS="[]",
        VERL_USE_UV="0",
        WANDB_MODE="offline",
        OMP_NUM_THREADS="1",
    )
    return env


def config_only(source, study, python, env, experiment, destination):
    local = dict(env, CONFIG_ONLY="1", RUN_DIR=str(destination), EXPERIMENT_NAME=destination.name)
    completed = subprocess.run(
        ["bash", str(source / "examples/mirroropd/run_math_opd_ablation_fsdp2.sh"), experiment],
        cwd=source,
        env=local,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"Configuration {experiment} failed: {completed.stderr[-4000:]}")
    config = yaml.safe_load(completed.stdout)
    loggers = config["trainer"]["logger"]
    if "file" not in loggers:
        loggers.append("file")
    destination.mkdir(parents=True, exist_ok=True)
    ray = config["ray_kwargs"]["ray_init"]
    ray["address"] = "local"
    ray["_temp_dir"] = "${oc.env:SMOKE_RAY_TMPDIR}"
    variables = ray.setdefault("runtime_env", {}).setdefault("env_vars", {})
    variables["VERL_ROLLOUT_IPC_NAMESPACE"] = "${oc.env:VERL_ROLLOUT_IPC_NAMESPACE}"
    variables["VERL_FILE_LOGGER_PATH"] = str(destination / "metrics.jsonl")
    variables["WANDB_DIR"] = str(destination)
    path = study / "configs" / f"{experiment}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    return config, path
