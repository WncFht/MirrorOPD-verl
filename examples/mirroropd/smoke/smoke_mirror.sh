#!/usr/bin/env bash
# Real-model mirror smoke: choose objective/estimator through the unified launcher.
set -eo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export STUDENT_GPUS=${STUDENT_GPUS:-2}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-$((2 * STUDENT_GPUS))}
export PPO_EPOCHS=${PPO_EPOCHS:-2}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-64}
export MAX_VAL_RESPONSE_LENGTH=${MAX_VAL_RESPONSE_LENGTH:-64}
export VAL_N=${VAL_N:-1}
export PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}
export TEACHER_FSDP_TOKEN_BUDGET=${TEACHER_FSDP_TOKEN_BUDGET:-$PPO_MAX_TOKEN_LEN_PER_GPU}
export ROLLOUT_ENFORCE_EAGER=True
export ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.15}
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-4}
export ROLLOUT_TOKEN_BUDGET=${ROLLOUT_TOKEN_BUDGET:-1024}
export FULL_KL_BACKEND=${FULL_KL_BACKEND:-precompute_fused}
export FULL_KL_TOKEN_CHUNK=${FULL_KL_TOKEN_CHUNK:-32}
export FULL_KL_ASYNC_WRITE=True FULL_KL_PREFETCH=True
export TEST_FREQ=-1 SAVE_FREQ=-1
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-mirror-smoke-${MIRROR_OBJECTIVE:-mdpo}-${MIRROR_ESTIMATOR:-full_vocab}}
export NVTX_INJECTION64_PATH=${NVTX_INJECTION64_PATH:-/nonexistent/verl-disabled-nvtx-injection.so}
exec bash "$SCRIPT_DIR/../run_math_opd_ablation_fsdp2.sh" custom \
    --method "${MIRROR_OBJECTIVE:-mdpo}" --estimator "${MIRROR_ESTIMATOR:-full_vocab}" \
    "trainer.total_training_steps=${SMOKE_STEPS:-2}" \
    'trainer.logger=[console,file]' \
    trainer.val_before_train=False \
    "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=${SMOKE_WEIGHT_BUCKET_MB:-512}" \
    data.train_max_samples=32 data.shuffle=False data.filter_overlong_prompts_workers=1 \
    "++ray_kwargs.ray_init.num_gpus=$STUDENT_GPUS" \
    "++ray_kwargs.ray_init.num_cpus=${SMOKE_RAY_CPUS:-32}" \
    ++ray_kwargs.ray_init.include_dashboard=False \
    ++ray_kwargs.ray_init.object_store_memory=536870912 \
    "++ray_kwargs.ray_init._temp_dir=${SMOKE_RAY_TMPDIR:-/tmp/mirror-smoke-$$}" \
    "$@"
