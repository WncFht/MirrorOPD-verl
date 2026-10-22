#!/usr/bin/env bash
# Math OPD: sampled-token reuse ablations and single-update full-KL comparisons.
# Usage: CUDA_VISIBLE_DEVICES=0,1,2,3 STUDENT_GPUS=4 bash "$0" E1 [Hydra overrides...]
# DRY_RUN=1 prints the resolved launch command without starting training.
set -eo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ ${1:-} == --help || ${1:-} == -h || $# == 0 ]]; then
    cat <<'HELP'
Usage: bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E1..E15 [Hydra overrides...]
  Short  Long   Algorithm
  E1     E7     rollout advantage + token PPO
  E2     E8     rollout advantage + token IS (no PPO/dual clip)
  E3     E9     current advantage + token PPO
  E4     E10    current advantage + token IS (no PPO/dual clip)
  E5     E11    current advantage + NoIS (reinforce)
  E6     E12    rollout advantage + NoIS (reinforce)
  E13          full-vocab reverse KL, LR=1e-6
  E14          full-vocab reverse KL, LR=5e-7 (learning-rate ablation)
  E15          original sampled k1 + current signal + vanilla PPO, LR=1e-6
E1-E6 defaults:  PPO_EPOCHS=4, TOTAL_TRAINING_STEPS=32,
                 TEST_FREQ=SAVE_FREQ=TOTAL_TRAINING_STEPS.
E7-E12 defaults: PPO_EPOCHS=16, TOTAL_TRAINING_STEPS=200,
                 TEST_FREQ=50, SAVE_FREQ=100 (outer steps).
E13-E15 defaults: PPO_EPOCHS=1, TOTAL_TRAINING_STEPS=200,
                  TEST_FREQ=20, SAVE_FREQ=100, keep the latest 2 checkpoints.
E13/E14 use shared_sync, precompute, async write and prefetch by default.
FULL_KL_BACKEND/FULL_KL_* environment variables can override those choices.
E15 keeps the legacy vanilla ratio against FSDP old_log_probs; no rollout IS.
Common: D=0 (no delayed queue), batch=mini-batch=64, no initial validation,
        OPD_CLIP_EPS=0.2, constant LR, zero warmup, synchronous checkpoint.
E7-E12 reuse each fresh batch 16 times; they do not set delay D=16.
Without skipped updates, 200 outer steps give 3200 updates in E7-E12,
and 200 updates in E13-E15. E13-E15 also align rollout/actor seeds with SEED.
Set CUDA_VISIBLE_DEVICES and STUDENT_GPUS together for fewer than eight shared GPUs.
Set PYTHON_BIN to the native Python executable if .venv-native is not present.
Environment variables override preset defaults; final Hydra arguments take precedence.
Use DRY_RUN=1 to inspect the resolved command without starting training.
Each invocation starts a fresh run unless explicit trainer.resume_* overrides are supplied.
HELP
    exit 0
fi
ABLATION=$1
shift
ENTRY="$SCRIPT_DIR/run_math_deepseek_r1_1_5b_fsdp2.sh"
PRESET_OVERRIDES=()
case "$ABLATION" in
    E1|E7) export OPD_ADVANTAGE_SOURCE=rollout OPD_POLICY_LOSS_MODE=token_ppo ;;
    E2|E8) export OPD_ADVANTAGE_SOURCE=rollout OPD_POLICY_LOSS_MODE=token_is ;;
    E3|E9) export OPD_ADVANTAGE_SOURCE=current OPD_POLICY_LOSS_MODE=token_ppo ;;
    E4|E10) export OPD_ADVANTAGE_SOURCE=current OPD_POLICY_LOSS_MODE=token_is ;;
    E5|E11) export OPD_ADVANTAGE_SOURCE=current OPD_POLICY_LOSS_MODE=reinforce ;;
    E6|E12) export OPD_ADVANTAGE_SOURCE=rollout OPD_POLICY_LOSS_MODE=reinforce ;;
    E13|E14|E15) export OPD_ADVANTAGE_SOURCE=current OPD_POLICY_LOSS_MODE=vanilla ;;
    *) echo "Unknown ablation: $ABLATION; expected E1 through E15" >&2; exit 2 ;;
esac
case "$ABLATION" in
    E7|E8|E9|E10|E11|E12)
        export PPO_EPOCHS=${PPO_EPOCHS:-16}
        TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-200}
        export TEST_FREQ=${TEST_FREQ:-50}
        export SAVE_FREQ=${SAVE_FREQ:-100}
        ;;
    E13|E14|E15)
        export PPO_EPOCHS=${PPO_EPOCHS:-1}
        TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-200}
        export TEST_FREQ=${TEST_FREQ:-20}
        export SAVE_FREQ=${SAVE_FREQ:-100}
        export SEED=${SEED:-42}
        if [[ $ABLATION == E14 ]]; then
            export LEARNING_RATE=${LEARNING_RATE:-5e-7}
        else
            export LEARNING_RATE=${LEARNING_RATE:-1e-6}
        fi
        PRESET_OVERRIDES=(
            trainer.max_actor_ckpt_to_keep=2
            "actor_rollout_ref.actor.fsdp_config.seed=$SEED"
            "actor_rollout_ref.actor.data_loader_seed=$SEED"
            "actor_rollout_ref.rollout.seed=$SEED"
        )
        if [[ $ABLATION != E15 ]]; then
            export DOMAIN=math
            export FULL_KL_BACKEND=${FULL_KL_BACKEND:-precompute}
            export FULL_KL_ASYNC_WRITE=${FULL_KL_ASYNC_WRITE:-True}
            export FULL_KL_PREFETCH=${FULL_KL_PREFETCH:-True}
            ENTRY="$SCRIPT_DIR/run_full_reverse_kl.sh"
        fi
        ;;
esac
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-32}
if [[ ! $TOTAL_TRAINING_STEPS =~ ^[1-9][0-9]*$ ]]; then
    echo "TOTAL_TRAINING_STEPS must be a positive integer" >&2
    exit 2
fi
export PPO_EPOCHS=${PPO_EPOCHS:-4}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
export TEST_FREQ=${TEST_FREQ:-$TOTAL_TRAINING_STEPS}
export SAVE_FREQ=${SAVE_FREQ:-$TOTAL_TRAINING_STEPS}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-math-opd-$ABLATION-$(date +%Y%m%d-%H%M%S)-$$}
exec bash "$ENTRY" \
    "trainer.total_training_steps=$TOTAL_TRAINING_STEPS" \
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.actor.checkpoint.async_save=False \
    "${PRESET_OVERRIDES[@]}" \
    "$@"
