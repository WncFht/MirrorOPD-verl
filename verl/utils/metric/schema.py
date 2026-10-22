# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Name and route observations without a global field allowlist."""

import logging
import math
from functools import cache

from .evaluation import dataset_slug, evaluation_metrics  # noqa: F401

logger = logging.getLogger(__name__)
FAMILIES = ("optim/", "policy/", "advantage/", "correction/", "distill/", "rollout/", "eval/", "perf/", "step/")
OUTER_RENAMES = {
    "actor/reward_kl_penalty": "policy/anchor_reference/reward_kl",
    "actor/reward_kl_penalty_coeff": "rollout/reward_kl/coefficient",
    "validation/timing_s/rollout_batch": "perf/timing/validation_rollout_seconds",
    "actor/entropy": "policy/entropy/pre_update",
    "critic/score/mean": "rollout/score_mean",
    "response/aborted_ratio": "rollout/aborted_fraction",
    "response_length/mean": "rollout/response_length_mean",
    "response_length/clip_ratio": "rollout/response_length/padded_width_fraction",
    "memory/fsdp_worker_peak_allocated_gib_max": "perf/memory/fsdp_allocated_peak_gib",
    "perf/throughput": "perf/throughput/tokens_per_gpu_second",
    "perf/total_num_tokens": "rollout/total_tokens",
    "perf/time_per_step": "perf/timing/outer_step_seconds",
    "training/global_step": "step/outer",
    "training/epoch": "rollout/data_epoch",
    "training/actor_optimizer_step": "step/optimizer",
    "training/actor_optimizer_step_attempts": "step/update",
    "training/actor_optimizer_skipped_updates": "optim/actor/skipped_updates",
    "training/rollout_probs_diff_valid": "policy/anchor_behavior/has_valid_tokens",
    "training/rollout_probs_diff_mean": "policy/anchor_behavior/prob_diff_mean",
    "training/rollout_probs_diff_max": "policy/anchor_behavior/prob_diff_max",
    "training/rollout_probs_diff_std": "policy/anchor_behavior/prob_diff_std",
    "training/rollout_actor_probs_pearson_corr": "policy/anchor_behavior/prob_pearson",
    "teacher/request_latency_s/mean": "perf/teacher/request_latency_mean_seconds",
    "rollout/spec_accept_rate": "perf/spec_decode/accept_rate",
    "rollout/spec_accept_length": "perf/spec_decode/accept_length",
}
TIMERS = {"gen": "rollout_pipeline", "step": "outer_step", "testing": "validation"}
INTERNAL_COUNTERS = {
    "distillation/token_observations",
    "distillation/advantage_sign_flip_count",
    "distillation/log_ratio_guard_count",
    "distillation/ppo_clip_count",
}


@cache
def _unknown(name):
    logger.warning("Metric %s has no naming rule; preserving it instead of silently discarding it", name)


def canonical_name(name, *, role=None):
    """Translate producer names once; dynamic names keep their complete suffix."""
    if name in OUTER_RENAMES:
        return OUTER_RENAMES[name]
    if role is not None and name in ("loss", "grad_norm", "lr", "lr_used"):
        stat = {"grad_norm": "grad_norm_preclip", "lr": "lr_next", "lr_used": "lr"}.get(name, name)
        return f"optim/{role}/{stat}"
    if role is not None and name in (
        "perf/max_memory_allocated_gb",
        "perf/max_memory_reserved_gb",
        "perf/cpu_memory_used_gb",
    ):
        return canonical_name(f"{role}/{name}")
    if role is not None and name == "mfu":
        return f"perf/{role}/mfu"
    for who in ("actor", "critic"):
        if name.startswith(who + "/perf/"):
            stat = name.split("/", 2)[2]
            stat = {
                "max_memory_allocated_gb": "allocated_peak_gib",
                "max_memory_reserved_gb": "reserved_peak_gib",
                "cpu_memory_used_gb": "cpu_used_gib",
            }.get(stat, stat)
            return f"perf/{who}/{stat}"
        if name in (f"perf/mfu/{who}", f"{who}/mfu"):
            return f"perf/{who}/mfu"
    if name.startswith("validation/"):
        sub = name[len("validation/") :]
        if sub.startswith("timing_s/"):
            return "perf/timing/validation_" + sub.split("/", 1)[1] + "_seconds"
        if sub == "rollout/preemptions":
            return "eval/preemptions"
        return "eval/system/" + sub
    if name.startswith("timing_s/"):
        stage = name.split("/", 1)[1]
        return "perf/timing/" + TIMERS.get(stage, stage) + "_seconds"
    if name.startswith("timing_per_token_ms/"):
        return "perf/time_per_token/" + name.split("/", 1)[1] + "_ms"
    if name.startswith("global_seqlen/"):
        return name.replace("global_seqlen/", "perf/load_balance/", 1)
    if name.startswith(("num_turns/", "training/num_turns/", "tool_call_counts/")):
        obj = "tool_calls" if name.startswith("tool_call") else "num_turns"
        return f"rollout/{obj}/" + name.rsplit("/", 1)[1]
    for kind in ("prompt_length", "response_length", "response_length_non_aborted"):
        if name.startswith(kind + "/"):
            stat = name.split("/", 1)[1].replace("clip_ratio", "padded_width_fraction")
            return f"rollout/{kind}/{stat}"
    if name.startswith("training/off_policy/"):
        return name.replace("training/off_policy/", "rollout/staleness/", 1)
    if name.startswith("training/filter_groups/"):
        return name.replace("training/filter_groups/", "rollout/filter/", 1)
    if name.startswith("training/rollout_failure/"):
        return name.replace("training/rollout_failure/", "rollout/failures/", 1)
    if name.startswith("separate_async/"):
        return name.replace("separate_async/", "perf/async/", 1)
    if name.startswith("rollout/moe/"):
        return name.replace("rollout/moe/", "perf/moe/", 1)
    if name.startswith("variance_proxy/"):
        return "advantage/" + name
    if name.startswith("rollout_corr/"):
        # Driver helpers see the full batch. In-loss legacy helpers see local
        # micro-batches; their nonlinear summaries are not global raw moments.
        # Keep that scope explicit alongside the mergeable applied_* observations.
        sub = name.split("/", 1)[1]
        if sub.startswith("rollout_is_"):
            sub = sub[len("rollout_is_") :]
            return ("correction/is/local_summary/" if role == "actor" else "correction/is/batch/") + {
                "eff_sample_size": "applied_ess_fraction",
                "batch_norm_factor": "normalization_factor",
            }.get(sub, sub)
        if sub.startswith("rollout_rs_"):
            return ("correction/rejection/local_summary/" if role == "actor" else "correction/rejection/batch/") + sub[
                len("rollout_rs_") :
            ]
        return ("policy/current_behavior/local_summary/" if role == "actor" else "policy/anchor_behavior/") + {
            "k3_kl": "k3",
            "kl": "sampled_reverse_log_ratio_mean",
        }.get(sub, sub)
    sub = name.removeprefix("actor/")
    if sub.startswith("distillation/"):
        stat = sub.split("/", 1)[1]
        if stat == "loss":
            return "optim/actor/distillation_loss"
        if stat in ("pg_clipfrac", "pg_clipfrac_lower"):
            return (
                "correction/anchor_ppo/"
                + {"pg_clipfrac": "active_fraction", "pg_clipfrac_lower": "dual_clip_fraction"}[stat]
            )
        if stat == "ppo_kl":
            return "policy/current_anchor/distillation_ppo_kl"
        if stat in ("abs_loss", "loss_min", "loss_max"):
            return "distill/loss/" + {"abs_loss": "abs_mean", "loss_min": "min", "loss_max": "max"}[stat]
        if stat.startswith(("student_mass", "teacher_mass", "overlap")):
            return "distill/topk/" + stat
        return "distill/" + stat
    if name in ("kl_loss", "kl_coef"):
        return f"optim/{role or 'actor'}/{name}"
    if name.startswith("actor/"):
        stat = name[len("actor/") :]
        if stat.startswith("sc_"):
            return "correction/score_centering/" + stat[3:]
        if stat in ("pg_clipfrac", "pg_clipfrac_lower"):
            return (
                "correction/task_ppo/"
                + {"pg_clipfrac": "active_fraction", "pg_clipfrac_lower": "dual_clip_fraction"}[stat]
            )
        if stat == "ppo_kl":
            return "policy/current_anchor/task_ppo_kl"
        if role is None and stat in ("loss", "grad_norm", "lr"):
            return "optim/actor/outer_" + stat + "_mean"
        return "optim/actor/" + stat
    if name.startswith("critic/"):
        stat = name[len("critic/") :]
        for old, new in [
            ("score/", "rollout/score/"),
            ("rewards/", "rollout/reward/"),
            ("returns/", "rollout/returns/"),
            ("advantages/", "advantage/task/batch_"),
            ("values/", "policy/critic/value_"),
        ]:
            if stat.startswith(old):
                return new + stat[len(old) :]
        if stat in ("vf_explained_var", "vpred_mean"):
            return "policy/critic/" + {"vf_explained_var": "explained_variance", "vpred_mean": "prediction_mean"}[stat]
        if stat == "vf_clipfrac":
            return "correction/value/active_fraction"
        return "optim/critic/" + {"vf_loss": "value_loss", "grad_norm": "grad_norm_preclip"}.get(stat, stat)
    if name.startswith(FAMILIES):
        return name
    _unknown(name)
    return name


def worker_outer_name(name, role):
    """Already-namespaced records and legacy worker-local performance gauges."""
    if name == "mfu" or name in (
        "perf/max_memory_allocated_gb",
        "perf/max_memory_reserved_gb",
        "perf/cpu_memory_used_gb",
    ):
        return canonical_name(name, role=role)
    if name.startswith(FAMILIES) or name.startswith("memory/"):
        return canonical_name(name)
    return canonical_name(f"{role}/{name}")


def metric_axis(name):
    if name.startswith("step/"):
        return None
    if name.startswith(("optim/critic/", "correction/value/")) or name == "policy/critic/prediction_mean":
        return "step/critic_update"
    if name.startswith(("correction/is/batch/", "correction/rejection/batch/")):
        return "step/outer"
    if "/batch_" in name:
        return "step/outer"
    if name.startswith(("advantage/task/batch_", "optim/actor/outer_", "advantage/variance_proxy/")):
        return "step/outer"
    if name in ("eval/samples", "rollout/filtered_reward_counts"):
        return "step/outer"
    if name.startswith("eval/"):
        return "step/optimizer"
    if (
        name.startswith(("optim/", "advantage/", "correction/", "distill/", "rollout/data_reuse/"))
        or name == "rollout/training_tokens"
    ):
        return "step/update"
    if name.startswith("policy/") and not name.startswith(
        ("policy/anchor_", "policy/entropy/pre_update", "policy/critic/")
    ):
        return "step/update"
    return "step/outer"


def is_update_metric(name):
    return (
        name.startswith(("policy/", "advantage/", "correction/", "distill/", "optim/", "rollout/data_reuse/"))
        or name == "rollout/training_tokens"
    )


def finite_metrics(metrics):
    """Undefined/nonfinite values become explicit event flags, never fake zeroes."""
    result = {}
    for key, value in metrics.items():
        if value is None:
            result[key + "/undefined"] = 1.0
            continue
        value = float(value)
        if math.isfinite(value):
            result[key] = value
        else:
            result[key + "/nonfinite"] = 1.0
            logger.warning("Nonfinite metric %s omitted; its /nonfinite event is recorded", key)
    return result


def outer_metrics(metrics):
    result = {}
    # Explicit canonical producer values win over equivalent legacy helper values.
    for key, value in metrics.items():
        name = canonical_name(key)
        if name not in result or name == key:
            result[name] = value
    return finite_metrics(result)
