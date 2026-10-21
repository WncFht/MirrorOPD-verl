#!/usr/bin/env bash
# Six sampled-token ablations; the shared launcher retains its vanilla default.
# Usage: CUDA_VISIBLE_DEVICES=0,1,2,3 STUDENT_GPUS=4 bash "$0" E1 [Hydra overrides...]
# DRY_RUN=1 prints the resolved launch command without starting training.
set -eo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ ${1:-} == --help || ${1:-} == -h || $# == 0 ]]; then
    cat <<'HELP'
Usage: bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E1|E2|E3|E4|E5|E6 [Hydra overrides...]
  E1  rollout advantage + token PPO
  E2  rollout advantage + token IS (no PPO/dual clip)
  E3  current advantage + token PPO
  E4  current advantage + token IS (no PPO/dual clip)
  E5  current advantage + NoIS (reinforce)
  E6  rollout advantage + NoIS (reinforce)
Defaults: batch=mini-batch=64, PPO_EPOCHS=4, TOTAL_TRAINING_STEPS=32,
          OPD_CLIP_EPS=0.2, constant LR, zero warmup, synchronous checkpoint.
Set CUDA_VISIBLE_DEVICES and STUDENT_GPUS together for fewer than eight shared GPUs.
Set PYTHON_BIN to the native Python executable if .venv-native is not present.
Use TOTAL_TRAINING_STEPS=200 for the longer comparison; DRY_RUN=1 to inspect.
Each invocation starts a fresh run unless explicit trainer.resume_* overrides are supplied.
HELP
    exit 0
fi
ABLATION=$1
shift
case "$ABLATION" in
    E1) export OPD_ADVANTAGE_SOURCE=rollout OPD_POLICY_LOSS_MODE=token_ppo ;;
    E2) export OPD_ADVANTAGE_SOURCE=rollout OPD_POLICY_LOSS_MODE=token_is ;;
    E3) export OPD_ADVANTAGE_SOURCE=current OPD_POLICY_LOSS_MODE=token_ppo ;;
    E4) export OPD_ADVANTAGE_SOURCE=current OPD_POLICY_LOSS_MODE=token_is ;;
    E5) export OPD_ADVANTAGE_SOURCE=current OPD_POLICY_LOSS_MODE=reinforce ;;
    E6) export OPD_ADVANTAGE_SOURCE=rollout OPD_POLICY_LOSS_MODE=reinforce ;;
    *) echo "Unknown ablation: $ABLATION; expected E1 through E6" >&2; exit 2 ;;
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
exec bash "$SCRIPT_DIR/run_math_deepseek_r1_1_5b_fsdp2.sh" \
    "trainer.total_training_steps=$TOTAL_TRAINING_STEPS" \
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.actor.checkpoint.async_save=False \
    "$@"
