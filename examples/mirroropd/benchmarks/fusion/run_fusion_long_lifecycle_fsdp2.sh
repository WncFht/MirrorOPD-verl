#!/usr/bin/env bash
# Full batch64/cap7168 lifecycle costs: reference once, optimized train/save/resume.
set -eo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$HERE/../../../.." && pwd)
WORKSPACE=$(dirname -- "$REPO")
checkpoint_root() {
    "$WORKSPACE/envs/mirroropd-fusion/bin/python" "$HERE/record_schema.py" \
        --checkpoint-root "$REPO/outputs/$1"
}

PAIR=${1:?Choose dense or moe}
PREFIX=${2:-fusion-long-$(date +%Y%m%d-%H%M%S)}
case "$PAIR" in
    dense) export TEACHER_MODEL=$(python3 "$REPO/examples/mirroropd/assets/paths.py" --alias Qwen3-8B --model-root "${MODEL_ROOT:-$WORKSPACE/model}") FUSION_MICRO_BATCH_SIZE=16 ;;
    moe) export TEACHER_MODEL=$(python3 "$REPO/examples/mirroropd/assets/paths.py" --alias Qwen3-30B-A3B-Instruct-2507 --model-root "${MODEL_ROOT:-$WORKSPACE/model}") FUSION_MICRO_BATCH_SIZE=8 ;;
    *) exit 2 ;;
esac
export PYTHONPATH=$REPO
export TRAIN_FILE=$WORKSPACE/runs/rollout-fusion-20261028/default-regression-prompts.parquet
export TRAIN_BATCH_SIZE=64 MAX_PROMPT_LENGTH=4096 MAX_RESPONSE_LENGTH=7168 MAX_VAL_RESPONSE_LENGTH=128
export PPO_MAX_TOKEN_LEN_PER_GPU=11264 TEACHER_FSDP_TOKEN_BUDGET=11264 PPO_EPOCHS=1 VAL_N=1
LOGS=$WORKSPACE/runs/rollout-fusion-20261028/$PREFIX
mkdir -p "$LOGS"
cd "$REPO"
run() {
    local name=$1
    shift
    if [[ -e $REPO/outputs/$name ]]; then
        echo "Refusing to overwrite $REPO/outputs/$name" >&2
        exit 2
    fi
    EXPERIMENT_NAME=$name bash examples/mirroropd/smoke/smoke_rollout_fusion.sh token_mixture current \
        'rollout_policy.execution.backend=hf_fusion' 'data.shuffle=False' 'trainer.test_freq=1' 'trainer.save_freq=2' 'data.val_max_samples=-1' "$@" \
        > "$LOGS/$name.log" 2>&1
}
{
run "$PREFIX-$PAIR-reference" 'trainer.total_training_steps=1' 'trainer.save_freq=-1' \
    '++actor_rollout_ref.rollout.engine_kwargs.hf_fusion.reference=true'
run "$PREFIX-$PAIR-optimized" 'trainer.total_training_steps=2'
run "$PREFIX-$PAIR-optimized-resume" 'trainer.total_training_steps=3' 'trainer.resume_mode=resume_path' \
    "trainer.resume_from_path=$(checkpoint_root "$PREFIX-$PAIR-optimized")/global_step_2"
exit
}
