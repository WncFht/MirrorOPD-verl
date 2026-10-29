#!/usr/bin/env bash
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
# Qwen3-8B non-thinking reference: 570 questions x 16 samples; no training.
set -eo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../../../.." && pwd)
WORKSPACE=$(dirname -- "$REPO_ROOT")
cd "$REPO_ROOT"
if [[ -z ${PYTHON_BIN:-} ]]; then
    if [[ -x $REPO_ROOT/.venv-native/bin/python ]]; then
        PYTHON_BIN=$REPO_ROOT/.venv-native/bin/python
    elif [[ -x /root/.local/share/mirroropd/venvs/profile-cu128/bin/python ]]; then
        PYTHON_BIN=/root/.local/share/mirroropd/venvs/profile-cu128/bin/python
    else
        echo 'Set PYTHON_BIN to the absolute Python path of the existing verl environment.' >&2
        exit 2
    fi
fi
export PYTHON_BIN VERL_USE_UV=0
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
IFS=',' read -r -a DEVICES <<< "$CUDA_VISIBLE_DEVICES"
export STUDENT_GPUS=${#DEVICES[@]}
export MODEL_ROOT=${MODEL_ROOT:-$WORKSPACE/model}
export TEACHER_MODEL="$MODEL_ROOT/Qwen3-8B"
if [[ -z ${DATA_ROOT:-} ]]; then
    if [[ -f $WORKSPACE/data/math/MATH-500/test.parquet ]]; then
        DATA_ROOT=$WORKSPACE/data
    else
        DATA_ROOT=$WORKSPACE/MD-OPD/data
    fi
fi
export DATA_ROOT
export MAX_PROMPT_LENGTH=1024 MAX_RESPONSE_LENGTH=7168 MAX_VAL_RESPONSE_LENGTH=31744
export VAL_N=16 VAL_BATCH_SIZE=null SEED=42 TRAIN_BATCH_SIZE=64
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-16}
export ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.8}
export ROLLOUT_TOKEN_BUDGET=8192 ROLLOUT_ENFORCE_EAGER=False ROLLOUT_TP=1
export WANDB_MODE=${WANDB_MODE:-online} WANDB_ENTITY=${WANDB_ENTITY:-wncfht}
export NETRC=${NETRC:-$WORKSPACE/.secrets/netrc}
# Prefer the explicitly configured credential file over a stale shell API key.
if [[ -f $NETRC ]]; then unset WANDB_API_KEY; fi
unset EXPERIMENT_METHOD FULL_KL_TERMINATION_TOKEN_IDS VERL_E41_TERMINATION_EQUIV
export RAY_ADDRESS=local RAY_USAGE_STATS_ENABLED=0 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3-8b-nonthinking-valid-$(date +%Y%m%d-%H%M%S)-$$}
export RUN_DIR=${RUN_DIR:-$REPO_ROOT/outputs/$EXPERIMENT_NAME}
export VERL_FILE_LOGGER_PATH="$RUN_DIR/metrics.jsonl"
exec bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E39-valid-teacher \
    trainer.logger='[console,file,wandb]' \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.9 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    data.filter_overlong_prompts_workers=1 \
    "++ray_kwargs.ray_init.num_gpus=$STUDENT_GPUS" \
    ++ray_kwargs.ray_init.num_cpus=32 \
    ++ray_kwargs.ray_init.include_dashboard=False \
    ++ray_kwargs.ray_init.object_store_memory=8589934592 \
    "++ray_kwargs.ray_init._temp_dir=/tmp/qwen3-8b-valid-$$" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.NETRC=$NETRC" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.WANDB_ENTITY=$WANDB_ENTITY" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_PATH=$VERL_FILE_LOGGER_PATH" \
    "$@"
