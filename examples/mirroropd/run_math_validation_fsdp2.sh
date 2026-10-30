#!/usr/bin/env bash
# Evaluate either original model through the same optimized math validation path.
set -eo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
if [[ $# == 0 || ${1:-} == --help || ${1:-} == -h ]]; then
    cat <<'HELP'
Usage: bash examples/mirroropd/run_math_validation_fsdp2.sh student|teacher [Hydra overrides...]
  student: STUDENT_MODEL, or MODEL_ROOT/DeepSeek-R1-Distill-Qwen-1.5B
  teacher: TEACHER_MODEL, or MODEL_ROOT/JustRL-DeepSeek-1.5B
Both use the selected model as the generator; distillation is disabled.
Math validation defaults: 630 prompts x 16 (MATH-500, AMC23, AIME24, AIME25, AIME26), temperature=0.7, top-p=0.9,
max response=31744, full submission, Graph, TP1, concurrency128/util0.65.
Validation repetition stop defaults to 16-128 token blocks repeated 8 times.
Set VAL_REPETITION_DETECTION=false to restore natural stopping without this heuristic.
These concurrency defaults were tuned for R1-1.5B. For Qwen3-Base, set
MAX_PROMPT_LENGTH=1024, CONTINUOUS_TOKEN_MODEL_FAMILY=qwen3,
ROLLOUT_MAX_NUM_SEQS=64 and ROLLOUT_GPU_MEMORY_UTILIZATION=0.8.
Add +data.apply_chat_template_kwargs.enable_thinking=False
and actor_rollout_ref.rollout.stop_token_ids=[151645] for the ChatML-stop arm.
Set CUDA_VISIBLE_DEVICES and STUDENT_GPUS to the allocated devices (default 8).
MODEL_ROOT/DATA_ROOT/PYTHON_BIN and execution overrides use the common launcher.
Use DRY_RUN=1 or CONFIG_ONLY=1 without starting Ray or GPU work.
Final Hydra arguments take precedence, including model and validation options.
HELP
    exit 0
fi
TARGET=$1
shift
MODEL_ROOT=${MODEL_ROOT:-$WORKSPACE/model}
case "$TARGET" in
    student) export STUDENT_MODEL=${STUDENT_MODEL:-$MODEL_ROOT/DeepSeek-R1-Distill-Qwen-1.5B} ;;
    teacher) export STUDENT_MODEL=${TEACHER_MODEL:-$MODEL_ROOT/JustRL-DeepSeek-1.5B} ;;
    *) echo "Expected student or teacher; got $TARGET" >&2; exit 2 ;;
esac
export DOMAIN=math DISTILLATION_EXECUTION_MODE=shared_sync TEACHER_GPUS=0 EXPERIMENT_METHOD=sampled
# Validation has no distillation objective, even if the caller exported a training preset.
unset FULL_KL_TERMINATION_TOKEN_IDS
export SEED=${SEED:-42}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-math-valid-$TARGET-$(date +%Y%m%d-%H%M%S)-$$}
exec bash "$SCRIPT_DIR/launchers/_run_single_teacher_fsdp2.sh" \
    distillation.enabled=False \
    actor_rollout_ref.rollout.calculate_log_probs=False \
    "actor_rollout_ref.rollout.seed=$SEED" \
    "actor_rollout_ref.actor.fsdp_config.seed=$SEED" \
    "actor_rollout_ref.actor.data_loader_seed=$SEED" \
    trainer.val_only=True \
    trainer.val_before_train=True \
    trainer.resume_mode=disable \
    trainer.save_freq=-1 \
    "$@"
