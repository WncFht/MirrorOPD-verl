#!/usr/bin/env bash
# Small real-model rollout -> teacher hidden -> full-KL update check.
set -eo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export STUDENT_GPUS=${STUDENT_GPUS:-4}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-$((2 * STUDENT_GPUS))}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-256}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-128}
export MAX_VAL_RESPONSE_LENGTH=${MAX_VAL_RESPONSE_LENGTH:-128}
export PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}
export TEACHER_FSDP_TOKEN_BUDGET=${TEACHER_FSDP_TOKEN_BUDGET:-$PPO_MAX_TOKEN_LEN_PER_GPU}
export ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-True}
export ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.25}
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-8}
export ROLLOUT_TOKEN_BUDGET=${ROLLOUT_TOKEN_BUDGET:-2048}
export FULL_KL_BACKEND=${FULL_KL_BACKEND:-precompute}
export FULL_KL_ASYNC_WRITE=${FULL_KL_ASYNC_WRITE:-True}
export FULL_KL_PREFETCH=${FULL_KL_PREFETCH:-True}
export TEST_FREQ=-1 SAVE_FREQ=-1
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-full-rkl-smoke-$(date +%Y%m%d-%H%M%S)}
export NVTX_INJECTION64_PATH=${NVTX_INJECTION64_PATH:-/nonexistent/verl-disabled-nvtx-injection.so}
SMOKE_RAY_TMPDIR=${SMOKE_RAY_TMPDIR:-/tmp/mirroropd-full-kl-smoke-$$}
exec bash "$SCRIPT_DIR/run_full_reverse_kl.sh" \
    "trainer.total_training_steps=${SMOKE_STEPS:-2}" \
    'trainer.logger=[console]' \
    "++ray_kwargs.ray_init.num_gpus=$STUDENT_GPUS" \
    "++ray_kwargs.ray_init.num_cpus=${SMOKE_RAY_CPUS:-32}" \
    ++ray_kwargs.ray_init.include_dashboard=False \
    ++ray_kwargs.ray_init.object_store_memory=4294967296 \
    "++ray_kwargs.ray_init._temp_dir=$SMOKE_RAY_TMPDIR" \
    "++actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config.backend=${SMOKE_ATTN_BACKEND:-TRITON_ATTN}" \
    "$@"
