#!/usr/bin/env bash
# Small current-code training + evaluation run for metrics inspection.
# Defaults: GPU 0-3, E3, 2 outer steps x 4 updates; 2 prompts/source x 4 responses.
# DRY_RUN=1 prints the command without creating subsets or starting Ray.
set -eo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXAMPLE_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(cd -- "$EXAMPLE_DIR/../.." && pwd)
WORKSPACE=$(dirname -- "$REPO_ROOT")
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export STUDENT_GPUS=${STUDENT_GPUS:-4}
export TEACHER_GPUS=0 DISTILLATION_EXECUTION_MODE=shared_sync VERL_USE_UV=0
export PYTHON_BIN=${PYTHON_BIN:-$REPO_ROOT/.venv-native/bin/python}
if [[ ! -x "$PYTHON_BIN" && -x /root/.local/share/mirroropd/venvs/profile-cu128/bin/python ]]; then
    export PYTHON_BIN=/root/.local/share/mirroropd/venvs/profile-cu128/bin/python
fi
export MODEL_ROOT=${MODEL_ROOT:-$WORKSPACE/model}
export DATA_ROOT=${DATA_ROOT:-$WORKSPACE/data}
if [[ ! -f "$DATA_ROOT/math/DAPO/train.parquet" && -f "$WORKSPACE/MD-OPD/data/math/DAPO/train.parquet" ]]; then
    export DATA_ROOT="$WORKSPACE/MD-OPD/data"
fi
METRICS_SMOKE_MODE=${METRICS_SMOKE_MODE:-e3}
SMOKE_OVERRIDES=()
case "$METRICS_SMOKE_MODE" in
    e3) DEFAULT_STEPS=2; DEFAULT_PPO_EPOCHS=4 ;;
    is-rs)
        DEFAULT_STEPS=1; DEFAULT_PPO_EPOCHS=2
        SMOKE_OVERRIDES=(
            distillation.distillation_loss.policy_loss_mode=vanilla
            algorithm.rollout_correction.rollout_is=token
            algorithm.rollout_correction.rollout_rs=token_k1
            "algorithm.rollout_correction.rollout_rs_threshold='0.99_1.01'"
            trainer.val_before_train=False trainer.test_freq=-1 trainer.log_val_generations=0
        )
        ;;
    *) echo "METRICS_SMOKE_MODE must be e3 or is-rs" >&2; exit 2 ;;
esac
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-metrics-v3-$METRICS_SMOKE_MODE-smoke-$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)-$$}
METRICS_RUN_ROOT=${METRICS_RUN_ROOT:-$EXAMPLE_DIR/outputs/$EXPERIMENT_NAME}
export RUN_DIR="$METRICS_RUN_ROOT/training"
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-$DEFAULT_STEPS}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8} PPO_EPOCHS=${PPO_EPOCHS:-$DEFAULT_PPO_EPOCHS}
export TEST_FREQ=1 SAVE_FREQ=-1
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}
export MAX_VAL_RESPONSE_LENGTH=${MAX_VAL_RESPONSE_LENGTH:-2048}
export VAL_N=${VAL_N:-4} VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-6}
export PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}
export TEACHER_FSDP_TOKEN_BUDGET=${TEACHER_FSDP_TOKEN_BUDGET:-8192}
export ROLLOUT_TOKEN_BUDGET=${ROLLOUT_TOKEN_BUDGET:-4096}
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-8}
export ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.25}
export ROLLOUT_ENFORCE_EAGER=True
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_ENTITY=${WANDB_ENTITY:-wncfht}
export WANDB_BASE_URL=${WANDB_BASE_URL:-https://api.forge.coreweave.com}
export RAY_ADDRESS=local RAY_USAGE_STATS_ENABLED=0 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export NVTX_INJECTION64_PATH=${NVTX_INJECTION64_PATH:-/nonexistent/verl-disabled-nvtx-injection.so}
if [[ ${DRY_RUN:-0} != 1 && ${CONFIG_ONLY:-0} != 1 ]]; then
    [[ ! -e "$RUN_DIR" ]] || { echo "Use a new EXPERIMENT_NAME: $RUN_DIR already exists" >&2; exit 2; }
    "$PYTHON_BIN" - "$DATA_ROOT" "$METRICS_RUN_ROOT/subsets" <<'PY'
from pathlib import Path
import sys
import pyarrow.parquet as pq
source, target = map(Path, sys.argv[1:])
target.mkdir(parents=True, exist_ok=True)
for name in ('MATH-500', 'AMC23', 'AIME25'):
    table = pq.read_table(source / 'math' / name / 'test.parquet').slice(0, 2)
    pq.write_table(table, target / (name + '.parquet'))
PY
fi
export MATH_FILE="$METRICS_RUN_ROOT/subsets/MATH-500.parquet"
export AMC_FILE="$METRICS_RUN_ROOT/subsets/AMC23.parquet"
export AIME_FILE="$METRICS_RUN_ROOT/subsets/AIME25.parquet"
exec bash "$EXAMPLE_DIR/run_math_opd_ablation_fsdp2.sh" E3 \
    data.train_max_samples=64 trainer.val_before_train=True trainer.log_val_generations=3 \
    trainer.metrics_quantile_interval=1 trainer.eval_metrics_resamples=256 \
    'trainer.logger=[console,wandb,file]' \
    "++ray_kwargs.ray_init.num_gpus=$STUDENT_GPUS" \
    ++ray_kwargs.ray_init.num_cpus=24 ++ray_kwargs.ray_init.include_dashboard=False \
    ++ray_kwargs.ray_init.object_store_memory=2147483648 \
    "++ray_kwargs.ray_init._temp_dir=/tmp/mm3-$$" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.WANDB_DIR='$METRICS_RUN_ROOT'" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.WANDB_ENTITY='$WANDB_ENTITY'" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.WANDB_BASE_URL='$WANDB_BASE_URL'" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_PATH='$METRICS_RUN_ROOT/metrics.jsonl'" \
    "${SMOKE_OVERRIDES[@]}" "$@"
