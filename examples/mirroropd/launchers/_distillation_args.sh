#!/usr/bin/env bash
# Sourced by the shared launcher. Single owner of algorithm -> Hydra settings.
# Resources, datasets and budgets belong to the preset/shared launcher.
DISTILLATION_OBJECTIVE_ARGS=(
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.loss_max_clamp=null
    distillation.distillation_loss.log_prob_min_clamp=null
    "distillation.distillation_loss.clip_ratio=$OPD_CLIP_EPS"
    "distillation.distillation_loss.clip_ratio_low=$OPD_CLIP_EPS"
    "distillation.distillation_loss.clip_ratio_high=$OPD_CLIP_EPS"
    "distillation.distillation_loss.full_vocab.termination_token_ids=${FULL_KL_TERMINATION_TOKEN_IDS:-[]}"
)
OPD_CALCULATE_LOG_PROBS=True
case "${EXPERIMENT_METHOD:-sampled}" in
    sampled)
        DISTILLATION_OBJECTIVE_ARGS+=(
            distillation.distillation_loss.loss_mode=k1
            distillation.distillation_loss.mirror=null
            distillation.distillation_loss.use_policy_gradient=True
            "distillation.distillation_loss.advantage_source=$OPD_ADVANTAGE_SOURCE"
            "distillation.distillation_loss.policy_loss_mode=$OPD_POLICY_LOSS_MODE"
        )
        ;;
    full_kl|mdpo|bpo)
        if [[ $DISTILLATION_EXECUTION_MODE != shared_sync ]]; then
            echo 'Full-KL and mirror currently require DISTILLATION_EXECUTION_MODE=shared_sync' >&2
            exit 2
        fi
        _opd_objective=${EXPERIMENT_METHOD}
        _opd_estimator=${MIRROR_ESTIMATOR:-full_vocab}
        _opd_kappa=${MIRROR_KAPPA:-0.5}
        _opd_eta=${MIRROR_ETA:-0.3}
        if [[ $EXPERIMENT_METHOD == full_kl ]]; then
            _opd_objective=mdpo; _opd_estimator=full_vocab; _opd_kappa=0; _opd_eta=1
        fi
        case "$_opd_objective/$_opd_estimator" in
            mdpo/sampled) _opd_pg=True; _opd_policy=token_is ;;
            mdpo/full_vocab|bpo/full_vocab) _opd_pg=False; _opd_policy=vanilla; OPD_CALCULATE_LOG_PROBS=False ;;
            *) echo 'Supported mirror settings: mdpo/sampled, mdpo/full_vocab, bpo/full_vocab' >&2; exit 2 ;;
        esac
        DISTILLATION_OBJECTIVE_ARGS+=(
            distillation.distillation_loss.loss_mode=mirror
            "distillation.distillation_loss.mirror={objective:$_opd_objective,estimator:$_opd_estimator,kappa:$_opd_kappa,eta:$_opd_eta,diagnostics:${MIRROR_DIAGNOSTICS:-False}}"
            "distillation.distillation_loss.use_policy_gradient=$_opd_pg"
            "distillation.distillation_loss.policy_loss_mode=$_opd_policy"
            distillation.distillation_loss.advantage_source=current
            "distillation.distillation_loss.full_vocab.backend=${FULL_KL_BACKEND:-precompute}"
            "distillation.distillation_loss.full_vocab.token_chunk_size=${FULL_KL_TOKEN_CHUNK:-512}"
            "distillation.distillation_loss.full_vocab.vocab_chunk_size=${FULL_KL_VOCAB_CHUNK:-8192}"
            "distillation.distillation_loss.full_vocab.async_write=${FULL_KL_ASYNC_WRITE:-False}"
            "distillation.distillation_loss.full_vocab.prefetch=${FULL_KL_PREFETCH:-False}"
            actor_rollout_ref.actor.calculate_entropy=False
        )
        ;;
    *) echo "Unknown EXPERIMENT_METHOD: $EXPERIMENT_METHOD" >&2; exit 2 ;;
esac
