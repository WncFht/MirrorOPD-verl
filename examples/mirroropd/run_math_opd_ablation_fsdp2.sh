#!/usr/bin/env bash
# Math OPD: algorithm/reuse ablations and numbered sampled/full failure comparisons.
# Usage: CUDA_VISIBLE_DEVICES=0,1,2,3 STUDENT_GPUS=4 bash "$0" E1 [Hydra overrides...]
# DRY_RUN=1 prints the resolved launch command without starting training.
set -eo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ ${1:-} == --help || ${1:-} == -h || $# == 0 ]]; then
    cat <<'HELP'
Usage: bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh PRESET [Hydra overrides...]

E21-E32 paired failure experiments:
  Sampled Full  Student                    Teacher                       Train cap
  E21     E22   R1-Distill-Qwen-1.5B       JustRL-DeepSeek-1.5B            7168
  E23     E24   R1-Distill-Qwen-1.5B       JustRL-DeepSeek-1.5B           15360
  E25     E26   R1-Distill-Qwen-1.5B       R1-Distill-Qwen-7B              7168
  E27     E28   Qwen3-1.7B-Base           Qwen3-8B                        7168
  E29     E30   Qwen3-1.7B-Base           Qwen3-4B                        7168
  E31     E32   Qwen3-1.7B                Qwen3-4B-Instruct-2507          4096
  Odd IDs use sampled k1 + current + vanilla; even IDs use full reverse KL.
  Full KL is already implemented: these presets select the existing backend.
  Train: DAPO, batch=mini=64, n=1, PPO_EPOCHS=1, LR=1e-6, 250 steps.
  Validate before training, every 20 steps, and at the final step; save every 20.
  Valid: MATH-500/AMC23/AIME25, n=16, temperature=0.7, top-p=0.9, cap31744.
  Add -valid-student or -valid-teacher to independently evaluate that model,
  e.g. E25-valid-teacher. Both IDs in a sampled/full pair share the same models.
  Qwen3 uses non-thinking. Base student prompt cap1024 + runtime stop151645;
  other generators use prompt cap4096 and their native EOS configuration.
  Per-engine concurrency starting points: R1 student128, Base12, Qwen student32;
  JustRL teacher64, other teachers32. These are bounded-pilot choices, not optima.
  E22 shares E13's model pair, loss and train cap, but uses 250 steps and initial
  validation; E13 retains 200 steps and no initial validation. E21 likewise
  repeats E15 with the new budget. Existing E1-E20 defaults are unchanged.
  Environment/Hydra overrides remain supported, including FULL_KL_*.

Legacy E1..E20 algorithm/reuse ablations:
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
  E16          full-vocab current + IS, 4 updates per batch, LR=1e-6
  E17          full-vocab current + IS, 16 updates per batch, LR=1e-6
  E18          full-vocab, 64 updates x 64 outer steps = 4096 updates
  E19          full-vocab, 32 updates x 128 outer steps = 4096 updates
  E20          full-vocab, 16 updates x 256 outer steps = 4096 updates
E1-E6 defaults:  PPO_EPOCHS=4, TOTAL_TRAINING_STEPS=32,
                 TEST_FREQ=SAVE_FREQ=TOTAL_TRAINING_STEPS.
E7-E12 defaults: PPO_EPOCHS=16, TOTAL_TRAINING_STEPS=200,
                 TEST_FREQ=50, SAVE_FREQ=100 (outer steps).
E13-E15 defaults: PPO_EPOCHS=1, TOTAL_TRAINING_STEPS=200,
                  TEST_FREQ=20, SAVE_FREQ=100, keep the latest 2 checkpoints.
E16/E17 defaults: PPO_EPOCHS=4/16; other settings match E13 (200 outer steps,
                  TEST_FREQ=20, SAVE_FREQ=100, keep the latest 2 checkpoints).
E18-E20 defaults: one data epoch, original DAPO data, LR=1e-6;
                  test every 512 updates, save every 1024 updates.
                  TEST_FREQ=8/16/32, SAVE_FREQ=16/32/64 (outer steps).
                  Batch=mini-batch=64; retain the last 4 checkpoints.
E13/E14/E16-E20 use shared_sync, precompute, async write and prefetch by default.
FULL_KL_BACKEND/FULL_KL_* environment variables can override those choices.
E15 keeps the legacy vanilla ratio against FSDP old_log_probs; no rollout IS.
E16/E17 use the existing exact full reverse-KL loss: current + untruncated IS
at fixed prefixes, with no extra sampled-token ratio or PPO clipping.
E1-E20 common: D=0 (no delayed queue), batch=mini-batch=64, no initial validation,
        OPD_CLIP_EPS=0.2, constant LR, zero warmup, synchronous checkpoint.
E7-E12 reuse each fresh batch 16 times; they do not set delay D=16.
Without skipped updates, 200 outer steps give 3200 updates in E7-E12,
and 200 updates in E13-E15. E13-E15 also align rollout/actor seeds with SEED.
E16/E17 give 800/3200 updates by default, with the same seed alignment as E13.
PPO_EPOCHS can override these reuse counts; batch reuse does not add delay D.
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
run_failure_preset() {
    local experiment=${ABLATION%%-*} mode=sampled
    local student=DeepSeek-R1-Distill-Qwen-1.5B teacher=JustRL-DeepSeek-1.5B
    local family=deepseek response=7168 prompt=4096 student_seqs=128 teacher_seqs=64
    local entry="$SCRIPT_DIR/run_math_deepseek_r1_1_5b_fsdp2.sh"
    local workspace
    workspace=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
    case "$experiment" in E22|E24|E26|E28|E30|E32) mode=full ;; esac
    if [[ $ABLATION != "$experiment" ]]; then
        mode=${ABLATION#*-}
        case "$mode" in
            valid-student|valid-teacher) ;;
            *) echo "Expected E21-E32, optionally followed by -valid-student or -valid-teacher" >&2; exit 2 ;;
        esac
    fi
    case "$experiment" in
        E21|E22) ;;
        E23|E24) response=15360 ;;
        E25|E26) teacher=DeepSeek-R1-Distill-Qwen-7B; teacher_seqs=32 ;;
        E27|E28|E29|E30)
            student=Qwen3-1.7B-Base
            if [[ $experiment == E27 || $experiment == E28 ]]; then teacher=Qwen3-8B; else teacher=Qwen3-4B; fi
            family=qwen3; prompt=1024; student_seqs=12; teacher_seqs=32
            ;;
        E31|E32)
            student=Qwen3-1.7B; teacher=Qwen3-4B-Instruct-2507
            family=qwen3; response=4096; student_seqs=32; teacher_seqs=32
            ;;
    esac
    if [[ $mode == valid-teacher ]]; then
        # The independently generating teacher has its own context and capacity.
        prompt=4096
        student_seqs=$teacher_seqs
    fi
    export MODEL_ROOT=${MODEL_ROOT:-$workspace/model}
    export STUDENT_MODEL=${STUDENT_MODEL:-$MODEL_ROOT/$student}
    export TEACHER_MODEL=${TEACHER_MODEL:-$MODEL_ROOT/$teacher}
    export CONTINUOUS_TOKEN_MODEL_FAMILY=${CONTINUOUS_TOKEN_MODEL_FAMILY:-$family}
    export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-$prompt}
    export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-$response}
    export MAX_VAL_RESPONSE_LENGTH=${MAX_VAL_RESPONSE_LENGTH:-31744}
    export VAL_N=${VAL_N:-16} VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-null}
    # Measured starting points, not claimed optima. The larger-concurrency sweep is paused.
    export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-$student_seqs}
    export SEED=${SEED:-42}
    export EXPERIMENT_NAME=${EXPERIMENT_NAME:-math-opd-$ABLATION-$(date +%Y%m%d-%H%M%S)-$$}
    local -a overrides=(
        "actor_rollout_ref.actor.fsdp_config.seed=$SEED"
        "actor_rollout_ref.actor.data_loader_seed=$SEED"
        "actor_rollout_ref.rollout.seed=$SEED"
    )
    if [[ $family == qwen3 ]]; then
        overrides+=(+data.apply_chat_template_kwargs.enable_thinking=False)
    fi
    if [[ $student == Qwen3-1.7B-Base && $mode != valid-teacher ]]; then
        overrides+=("actor_rollout_ref.rollout.stop_token_ids=[151645]")
    fi
    case "$mode" in
        valid-*)
            exec bash "$SCRIPT_DIR/run_math_validation_fsdp2.sh" "${mode#valid-}" "${overrides[@]}" "$@"
            ;;
    esac
    export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64} PPO_EPOCHS=${PPO_EPOCHS:-1}
    export TOTAL_EPOCHS=${TOTAL_EPOCHS:-1} LEARNING_RATE=${LEARNING_RATE:-1e-6}
    export OPD_ADVANTAGE_SOURCE=current OPD_POLICY_LOSS_MODE=vanilla
    export TEST_FREQ=${TEST_FREQ:-20} SAVE_FREQ=${SAVE_FREQ:-20}
    local steps=${TOTAL_TRAINING_STEPS:-250}
    if [[ ! $steps =~ ^[1-9][0-9]*$ ]]; then
        echo "TOTAL_TRAINING_STEPS must be a positive integer" >&2
        exit 2
    fi
    if [[ $mode == full ]]; then
        export DOMAIN=math
        export FULL_KL_BACKEND=${FULL_KL_BACKEND:-precompute}
        export FULL_KL_ASYNC_WRITE=${FULL_KL_ASYNC_WRITE:-True}
        export FULL_KL_PREFETCH=${FULL_KL_PREFETCH:-True}
        entry="$SCRIPT_DIR/run_full_reverse_kl.sh"
    fi
    exec bash "$entry" \
        "trainer.total_training_steps=$steps" \
        trainer.val_before_train=True \
        trainer.max_actor_ckpt_to_keep=4 \
        actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
        actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
        actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
        actor_rollout_ref.actor.checkpoint.async_save=False \
        "${overrides[@]}" "$@"
}

case "$ABLATION" in
    E2[1-9]|E3[0-2]|E2[1-9]-*|E3[0-2]-*) run_failure_preset "$@" ;;
esac

ENTRY="$SCRIPT_DIR/run_math_deepseek_r1_1_5b_fsdp2.sh"
PRESET_OVERRIDES=()
case "$ABLATION" in
    E1|E7) export OPD_ADVANTAGE_SOURCE=rollout OPD_POLICY_LOSS_MODE=token_ppo ;;
    E2|E8) export OPD_ADVANTAGE_SOURCE=rollout OPD_POLICY_LOSS_MODE=token_is ;;
    E3|E9) export OPD_ADVANTAGE_SOURCE=current OPD_POLICY_LOSS_MODE=token_ppo ;;
    E4|E10) export OPD_ADVANTAGE_SOURCE=current OPD_POLICY_LOSS_MODE=token_is ;;
    E5|E11) export OPD_ADVANTAGE_SOURCE=current OPD_POLICY_LOSS_MODE=reinforce ;;
    E6|E12) export OPD_ADVANTAGE_SOURCE=rollout OPD_POLICY_LOSS_MODE=reinforce ;;
    E13|E14|E15|E16|E17|E18|E19|E20) export OPD_ADVANTAGE_SOURCE=current OPD_POLICY_LOSS_MODE=vanilla ;;
    *) echo "Unknown ablation: $ABLATION; expected E1 through E32 (E21-E32 also support -valid-student/-valid-teacher)" >&2; exit 2 ;;
esac
# Equal-update presets use the same file and seeded sampler, stopping at nested
# prefixes of one data epoch. Explicit overrides remain supported for smoke runs.
case "$ABLATION" in
    E18|E19|E20)
        case "$ABLATION" in
            E18) ABLATION_REUSE=64; ABLATION_OUTER=64; ABLATION_TEST=8; ABLATION_SAVE=16 ;;
            E19) ABLATION_REUSE=32; ABLATION_OUTER=128; ABLATION_TEST=16; ABLATION_SAVE=32 ;;
            E20) ABLATION_REUSE=16; ABLATION_OUTER=256; ABLATION_TEST=32; ABLATION_SAVE=64 ;;
        esac
        export PPO_EPOCHS=${PPO_EPOCHS:-$ABLATION_REUSE}
        TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-$ABLATION_OUTER}
        export TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
        export TEST_FREQ=${TEST_FREQ:-$ABLATION_TEST}
        export SAVE_FREQ=${SAVE_FREQ:-$ABLATION_SAVE}
        ;;
esac
case "$ABLATION" in
    E7|E8|E9|E10|E11|E12)
        export PPO_EPOCHS=${PPO_EPOCHS:-16}
        TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-200}
        export TEST_FREQ=${TEST_FREQ:-50}
        export SAVE_FREQ=${SAVE_FREQ:-100}
        ;;
    E13|E14|E15|E16|E17|E18|E19|E20)
        if [[ $ABLATION == E16 ]]; then
            export PPO_EPOCHS=${PPO_EPOCHS:-4}
        elif [[ $ABLATION == E17 ]]; then
            export PPO_EPOCHS=${PPO_EPOCHS:-16}
        else
            export PPO_EPOCHS=${PPO_EPOCHS:-1}
        fi
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
        case "$ABLATION" in
            E18|E19|E20)
                PRESET_OVERRIDES+=(
                    trainer.max_actor_ckpt_to_keep=4
                    data.train_max_samples=-1
                    algorithm.filter_groups.enable=False
                    trainer.v1.sampler.sync_refill_failed_groups=False
                )
                ;;
        esac
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
