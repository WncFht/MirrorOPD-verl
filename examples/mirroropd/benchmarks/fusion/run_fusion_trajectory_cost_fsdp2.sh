#!/usr/bin/env bash
# Cost reference only: two native-vLLM trajectory batches, with actor updates disabled.
set -eo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$HERE/../../../.." && pwd)
WORKSPACE=$(dirname -- "$REPO")
PAIR=${1:?Choose dense or moe}
export EXPERIMENT_NAME=${2:-fusion-trajectory-cost-$PAIR-$(date +%Y%m%d-%H%M%S)}
case "$PAIR" in
    dense) export TEACHER_MODEL=$WORKSPACE/model/Qwen3-8B ;;
    moe) export TEACHER_MODEL=$WORKSPACE/model/Qwen3-30B-A3B-Instruct-2507 ;;
    *) exit 2 ;;
esac
export PYTHONPATH=$REPO
export TRAIN_FILE=$WORKSPACE/runs/rollout-fusion-20261028/default-regression-prompts.parquet
export TRAIN_BATCH_SIZE=64 MAX_PROMPT_LENGTH=4096 MAX_RESPONSE_LENGTH=7168 MAX_VAL_RESPONSE_LENGTH=128
export PPO_MAX_TOKEN_LEN_PER_GPU=11264 TEACHER_FSDP_TOKEN_BUDGET=11264 PPO_EPOCHS=1 VAL_N=1
export ROLLOUT_ENFORCE_EAGER=False ROLLOUT_MAX_NUM_SEQS=32 ROLLOUT_GPU_MEMORY_UTILIZATION=0.65
export ROLLOUT_TOKEN_BUDGET=8192
cd "$REPO"
exec bash examples/mirroropd/smoke/smoke_rollout_fusion.sh trajectory_mixture current \
    'trainer.total_training_steps=2' 'data.shuffle=False' 'trainer.critic_warmup=100' \
    'trainer.test_freq=-1' 'trainer.save_freq=-1'
