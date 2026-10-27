#!/usr/bin/env bash
# Real student rollout -> Polaris hidden states -> native full-FKL updates.
set -eo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXAMPLE_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
SMOKE_REPO_ROOT=$(cd -- "$EXAMPLE_DIR/../.." && pwd)
export STUDENT_GPUS=${STUDENT_GPUS:-4}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-$((2 * STUDENT_GPUS))}
export PPO_EPOCHS=${PPO_EPOCHS:-2}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-256}
export MAX_VAL_RESPONSE_LENGTH=${MAX_VAL_RESPONSE_LENGTH:-256}
export VAL_N=${VAL_N:-1}
export ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-True}
export ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.12}
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-4}
export ROLLOUT_TOKEN_BUDGET=${ROLLOUT_TOKEN_BUDGET:-1024}
export FULL_KL_BACKEND=${FULL_KL_BACKEND:-precompute_fused}
export TEST_FREQ=-1 SAVE_FREQ=-1 VAL_BEFORE_TRAIN=False
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-full-fkl-smoke-$(date +%Y%m%d-%H%M%S)}
export VERL_FILE_LOGGER_PATH=${VERL_FILE_LOGGER_PATH:-${RUN_DIR:-$SMOKE_REPO_ROOT/outputs/$EXPERIMENT_NAME}/history.jsonl}
export NVTX_INJECTION64_PATH=${NVTX_INJECTION64_PATH:-/nonexistent/verl-disabled-nvtx-injection.so}
SMOKE_RAY_TMPDIR=${SMOKE_RAY_TMPDIR:-/tmp/mirroropd-full-fkl-smoke-$$}
exec bash "$EXAMPLE_DIR/run_math_opd_ablation_fsdp2.sh" E49 \
    "trainer.total_training_steps=${SMOKE_STEPS:-2}" \
    "trainer.val_before_train=${SMOKE_INITIAL_VALIDATION:-False}" \
    'trainer.logger=[console,file]' \
    "++ray_kwargs.ray_init.num_gpus=$STUDENT_GPUS" \
    "++ray_kwargs.ray_init.num_cpus=${SMOKE_RAY_CPUS:-32}" \
    ++ray_kwargs.ray_init.include_dashboard=False \
    ++ray_kwargs.ray_init.object_store_memory=4294967296 \
    "++ray_kwargs.ray_init._temp_dir=$SMOKE_RAY_TMPDIR" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_PATH='$VERL_FILE_LOGGER_PATH'" \
    "++actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config.backend=${SMOKE_ATTN_BACKEND:-TRITON_ATTN}" \
    "$@"
