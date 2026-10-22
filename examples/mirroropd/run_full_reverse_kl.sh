#!/usr/bin/env bash
# Single-teacher full reverse KL; DOMAIN=math|code|if selects the existing recipe.
set -eo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export DOMAIN=${DOMAIN:-math}
if [[ ${DISTILLATION_EXECUTION_MODE:-shared_sync} != shared_sync ]]; then
    echo "Full reverse KL currently requires DISTILLATION_EXECUTION_MODE=shared_sync" >&2
    exit 2
fi
FULL_KL_BACKEND=${FULL_KL_BACKEND:-recompute}
FULL_KL_TOKEN_CHUNK=${FULL_KL_TOKEN_CHUNK:-512}
FULL_KL_VOCAB_CHUNK=${FULL_KL_VOCAB_CHUNK:-8192}
FULL_KL_ASYNC_WRITE=${FULL_KL_ASYNC_WRITE:-False}
FULL_KL_PREFETCH=${FULL_KL_PREFETCH:-False}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-full-rkl-${DOMAIN}-${FULL_KL_BACKEND}}
exec bash "$SCRIPT_DIR/_run_single_teacher_fsdp2.sh" \
    distillation.execution_mode=shared_sync \
    distillation.distillation_loss.loss_mode=full_reverse_kl \
    distillation.distillation_loss.use_policy_gradient=False \
    distillation.distillation_loss.use_task_rewards=False \
    distillation.distillation_loss.policy_loss_mode=vanilla \
    distillation.distillation_loss.loss_max_clamp=null \
    distillation.distillation_loss.log_prob_min_clamp=null \
    "distillation.distillation_loss.full_vocab.backend=$FULL_KL_BACKEND" \
    "distillation.distillation_loss.full_vocab.token_chunk_size=$FULL_KL_TOKEN_CHUNK" \
    "distillation.distillation_loss.full_vocab.vocab_chunk_size=$FULL_KL_VOCAB_CHUNK" \
    "distillation.distillation_loss.full_vocab.async_write=$FULL_KL_ASYNC_WRITE" \
    "distillation.distillation_loss.full_vocab.prefetch=$FULL_KL_PREFETCH" \
    actor_rollout_ref.rollout.calculate_log_probs=False \
    actor_rollout_ref.actor.calculate_entropy=False \
    "$@"
