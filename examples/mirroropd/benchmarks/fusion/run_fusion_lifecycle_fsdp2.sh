#!/usr/bin/env bash
# Real updates, all five validation sets (n=1, short response cap), save and restore.
set -eo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$HERE/../../../.." && pwd)
WORKSPACE=$(dirname -- "$REPO")
PREFIX=${1:-fusion-final-$(date +%Y%m%d-%H%M%S)}
export PYTHONPATH=$REPO
export PPO_EPOCHS=2 VAL_N=1 MAX_PROMPT_LENGTH=4096 MAX_RESPONSE_LENGTH=128 MAX_VAL_RESPONSE_LENGTH=128
export TRAIN_BATCH_SIZE=8 FUSION_MICRO_BATCH_SIZE=2
LOGS=$WORKSPACE/runs/rollout-fusion-20261028/$PREFIX
mkdir -p "$LOGS"
cd "$REPO"

run() {
    local name=$1 mode=$2 binding=$3
    shift 3
    if [[ -e $REPO/outputs/$name ]]; then
        echo "Refusing to overwrite $REPO/outputs/$name" >&2
        exit 2
    fi
    EXPERIMENT_NAME=$name bash examples/mirroropd/smoke/smoke_rollout_fusion.sh "$mode" "$binding" \
        'trainer.test_freq=1' 'trainer.save_freq=2' 'data.train_max_samples=64' 'data.val_max_samples=-1' \
        "$@" > "$LOGS/$name.log" 2>&1
}

current_pair() {
    local name=$1
    run "$name" token_mixture current 'trainer.total_training_steps=2'
    run "$name-resume" token_mixture current 'trainer.total_training_steps=3' \
        'trainer.resume_mode=resume_path' "trainer.resume_from_path=$REPO/outputs/$name/checkpoints/global_step_2"
}

{
export TEACHER_MODEL=$(python3 "$REPO/examples/mirroropd/assets/paths.py" --alias Qwen3-8B --model-root "${MODEL_ROOT:-$WORKSPACE/model}")
current_pair "$PREFIX-dense-current"
schedule='rollout_policy.teacher_weight={kind:linear,start:1.0,end:0.0,steps:4,clock:optimizer_step}'
run "$PREFIX-initial-schedule" token_mixture initial 'trainer.total_training_steps=2' "$schedule"
run "$PREFIX-initial-schedule-resume" token_mixture initial 'trainer.total_training_steps=3' "$schedule" \
    'trainer.resume_mode=resume_path' \
    "trainer.resume_from_path=$REPO/outputs/$PREFIX-initial-schedule/checkpoints/global_step_2"
TRAIN_BATCH_SIZE=64 run "$PREFIX-trajectory-current" trajectory_mixture current 'trainer.total_training_steps=1'
run "$PREFIX-trajectory-initial-n2" trajectory_mixture initial 'trainer.total_training_steps=1' \
    'actor_rollout_ref.rollout.n=2'
export TEACHER_MODEL=$(python3 "$REPO/examples/mirroropd/assets/paths.py" --alias Qwen3-30B-A3B-Instruct-2507 --model-root "${MODEL_ROOT:-$WORKSPACE/model}")
current_pair "$PREFIX-moe-current"
exit
}
