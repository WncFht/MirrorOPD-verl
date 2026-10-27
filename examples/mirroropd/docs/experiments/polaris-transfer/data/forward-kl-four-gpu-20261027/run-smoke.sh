#!/usr/bin/env bash
set -eo pipefail
FKL_REPO=$(git rev-parse --show-toplevel)
cd "$FKL_REPO"
export CUDA_VISIBLE_DEVICES=0,1,2,3 STUDENT_GPUS=4
export PYTHON_BIN=${PYTHON_BIN:-/root/.local/share/mirroropd/venvs/profile-cu128/bin/python}
export VERL_USE_UV=0 OMP_NUM_THREADS=1 RAY_ADDRESS=local
export MODEL_ROOT=${MODEL_ROOT:-$FKL_REPO/../model}
export DATA_ROOT=${DATA_ROOT:-$FKL_REPO/examples/mirroropd/docs/experiments/polaris-transfer/data/forward-kl-four-gpu-20261027/inputs}
export TRAIN_FILE="$DATA_ROOT/train.parquet"
export MATH_FILE="$DATA_ROOT/math.parquet"
export AMC_FILE="$DATA_ROOT/amc.parquet"
export AIME_FILE="$DATA_ROOT/aime.parquet"
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-full-fkl-four-gpu-20261027-validation-v2}
export RUN_DIR=${RUN_DIR:-$FKL_REPO/outputs/$EXPERIMENT_NAME}
export TRAIN_BATCH_SIZE=64 PPO_EPOCHS=16 SMOKE_STEPS=2
export MAX_PROMPT_LENGTH=512 MAX_RESPONSE_LENGTH=512 MAX_VAL_RESPONSE_LENGTH=512
export VAL_N=1 FULL_KL_TOKEN_CHUNK=512
export SMOKE_RAY_TMPDIR=${SMOKE_RAY_TMPDIR:-/tmp/fkl4-$$}
bash examples/mirroropd/smoke/smoke_full_forward_kl.sh trainer.total_epochs=2 trainer.test_freq=2 "$@"
