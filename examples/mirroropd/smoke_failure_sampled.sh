#!/usr/bin/env bash
# C/D sampled-token integration check; short responses are not a failure reproduction.
set -eo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
WORKSPACE=$(dirname -- "$REPO_ROOT")
SETTING=${1:?Usage: smoke_failure_sampled.sh C|D [Hydra overrides...]}
shift
export MODEL_ROOT=${MODEL_ROOT:-$WORKSPACE/model}
case "$SETTING" in
    C)
        export STUDENT_MODEL=${STUDENT_MODEL:-$MODEL_ROOT/DeepSeek-R1-Distill-Qwen-1.5B}
        export TEACHER_MODEL=${TEACHER_MODEL:-$MODEL_ROOT/DeepSeek-R1-Distill-Qwen-7B}
        export CONTINUOUS_TOKEN_MODEL_FAMILY=deepseek
        ;;
    D)
        export STUDENT_MODEL=${STUDENT_MODEL:-$MODEL_ROOT/Qwen3-1.7B-Base}
        export TEACHER_MODEL=${TEACHER_MODEL:-$MODEL_ROOT/Qwen3-8B}
        export CONTINUOUS_TOKEN_MODEL_FAMILY=qwen3
        ;;
    *) echo 'Expected C or D' >&2; exit 2 ;;
esac
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export STUDENT_GPUS=4 TEACHER_GPUS=0 DISTILLATION_EXECUTION_MODE=shared_sync VERL_USE_UV=0
export PYTHON_BIN=${PYTHON_BIN:-$REPO_ROOT/.venv-native/bin/python}
export DATA_ROOT=${DATA_ROOT:-$WORKSPACE/data}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-failure-$SETTING-sampled-smoke-$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)-$$}
export RUN_DIR=${RUN_DIR:-$REPO_ROOT/outputs/$EXPERIMENT_NAME}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8} PPO_EPOCHS=1 LEARNING_RATE=1e-6
export OPD_ADVANTAGE_SOURCE=current OPD_POLICY_LOSS_MODE=vanilla
export MAX_PROMPT_LENGTH=1024 MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}
export MAX_VAL_RESPONSE_LENGTH=512 VAL_N=1 VAL_BATCH_SIZE=3
export TEST_FREQ=2 SAVE_FREQ=-1
export PPO_MAX_TOKEN_LEN_PER_GPU=8192 TEACHER_FSDP_TOKEN_BUDGET=8192
export ROLLOUT_TOKEN_BUDGET=8192 ROLLOUT_MAX_NUM_SEQS=8
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.25 ROLLOUT_ENFORCE_EAGER=True
export RAY_ADDRESS=local RAY_USAGE_STATS_ENABLED=0 WANDB_MODE=offline
export NVTX_INJECTION64_PATH=/nonexistent/verl-disabled-nvtx-injection.so
SUBSETS="$RUN_DIR-subsets"
if [[ ${DRY_RUN:-0} != 1 && ${CONFIG_ONLY:-0} != 1 ]]; then
    "$PYTHON_BIN" - "$DATA_ROOT" "$SUBSETS" <<'PY'
from pathlib import Path
import sys
import pyarrow.parquet as pq
source, target = map(Path, sys.argv[1:])
target.mkdir(parents=True, exist_ok=False)
for name in ('MATH-500', 'AMC23', 'AIME25'):
    pq.write_table(pq.read_table(source / 'math' / name / 'test.parquet').slice(0, 1), target / (name + '.parquet'))
PY
fi
export MATH_FILE="$SUBSETS/MATH-500.parquet" AMC_FILE="$SUBSETS/AMC23.parquet" AIME_FILE="$SUBSETS/AIME25.parquet"
EXTRA=()
if [[ $SETTING == D ]]; then
    EXTRA+=(+data.apply_chat_template_kwargs.enable_thinking=False)
fi
exec bash "$SCRIPT_DIR/run_math_deepseek_r1_1_5b_fsdp2.sh" \
    trainer.total_training_steps=2 data.train_max_samples=64 data.filter_overlong_prompts_workers=1 \
    trainer.metrics_quantile_interval=1 trainer.eval_metrics_resamples=16 \
    'trainer.logger=[console,file]' \
    ++ray_kwargs.ray_init.num_gpus=4 ++ray_kwargs.ray_init.num_cpus=24 \
    ++ray_kwargs.ray_init.include_dashboard=False ++ray_kwargs.ray_init.object_store_memory=2147483648 \
    "++ray_kwargs.ray_init._temp_dir=/tmp/cd-smoke-$$" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_PATH='$RUN_DIR/metrics.jsonl'" \
    "${EXTRA[@]}" "$@"
