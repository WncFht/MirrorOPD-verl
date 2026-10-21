# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Canonical metric names and reporting axes, independent of algorithm math."""

import math
import re
from collections import defaultdict

OUTER_RENAMES = {
    "actor/entropy": "policy/entropy/pre_update",
    "actor/perf/cpu_memory_used_gb": "perf/memory/cpu_used_gib",
    "actor/perf/max_memory_reserved_gb": "perf/actor/reserved_peak_gib",
    "critic/score/mean": "rollout/score_mean",
    "memory/fsdp_worker_peak_allocated_gib_max": "perf/memory/fsdp_allocated_peak_gib",
    "perf/mfu/actor": "perf/actor/mfu",
    "perf/throughput": "perf/throughput/tokens_per_gpu_second",
    "response/aborted_ratio": "rollout/aborted_fraction",
    "response_length/clip_ratio": "rollout/length_limit_fraction",
    "response_length/mean": "rollout/response_length_mean",
    "timing_s/adv": "perf/timing/adv_seconds",
    "timing_s/gen": "perf/timing/rollout_pipeline_seconds",
    "timing_s/old_log_prob": "perf/timing/old_log_prob_seconds",
    "timing_s/rollout_batch": "perf/timing/rollout_batch_seconds",
    "timing_s/save_checkpoint": "perf/timing/save_checkpoint_seconds",
    "timing_s/step": "perf/timing/outer_step_seconds",
    "timing_s/teacher_forward": "perf/timing/teacher_forward_seconds",
    "timing_s/teacher_transfer": "perf/timing/teacher_transfer_seconds",
    "timing_s/testing": "perf/timing/validation_seconds",
    "timing_s/update_actor": "perf/timing/update_actor_seconds",
    "timing_s/update_weights": "perf/timing/update_weights_seconds",
    "validation/timing_s/rollout_batch": "perf/timing/validation_rollout_seconds",
}

CANONICAL_FIELDS = frozenset(
    [
        "advantage/distillation/abs_mean",
        "advantage/distillation/mean",
        "advantage/distillation/positive_fraction",
        "advantage/distillation/rms",
        "correction/behavior_ppo/active_fraction",
        "correction/behavior_ppo/active_mass",
        "correction/is/coeff_change",
        "correction/log_ratio_guard/fraction",
        "distill/advantage/sign_flip_fraction",
        "distill/advantage/source_coeff_change",
        "eval/aime25/acc_mean",
        "eval/aime25/failed_count",
        "eval/aime25/response_count",
        "eval/amc23/acc_mean",
        "eval/amc23/failed_count",
        "eval/amc23/response_count",
        "eval/math500/acc_mean",
        "eval/math500/failed_count",
        "eval/math500/response_count",
        "optim/actor/grad_norm_preclip",
        "optim/actor/loss",
        "optim/actor/lr",
        "perf/actor/mfu",
        "perf/actor/reserved_peak_gib",
        "perf/memory/cpu_used_gib",
        "perf/memory/fsdp_allocated_peak_gib",
        "perf/throughput/tokens_per_gpu_second",
        "perf/timing/adv_seconds",
        "perf/timing/old_log_prob_seconds",
        "perf/timing/outer_step_seconds",
        "perf/timing/rollout_batch_seconds",
        "perf/timing/rollout_pipeline_seconds",
        "perf/timing/save_checkpoint_seconds",
        "perf/timing/teacher_forward_seconds",
        "perf/timing/teacher_transfer_seconds",
        "perf/timing/update_actor_seconds",
        "perf/timing/update_weights_seconds",
        "perf/timing/validation_rollout_seconds",
        "perf/timing/validation_seconds",
        "policy/anchor_behavior/k3",
        "policy/anchor_behavior/log_ratio_abs_max",
        "policy/anchor_behavior/log_ratio_abs_mean",
        "policy/anchor_behavior/nonfinite_fraction",
        "policy/current_anchor/log_ratio_abs_max",
        "policy/current_anchor/log_ratio_m2",
        "policy/current_anchor/nonfinite_fraction",
        "policy/current_behavior/ess_fraction",
        "policy/current_behavior/log_ratio_abs_max",
        "policy/current_behavior/log_ratio_m2",
        "policy/current_behavior/log_ratio_mean",
        "policy/current_behavior/nonfinite_fraction",
        "policy/current_behavior/ratio_gt_2_fraction",
        "policy/current_behavior/ratio_lt_half_fraction",
        "policy/current_behavior/ratio_p99",
        "policy/entropy/pre_update",
        "rollout/aborted_fraction",
        "rollout/length_limit_fraction",
        "rollout/response_count",
        "rollout/response_length_mean",
        "rollout/response_tokens",
        "rollout/score_mean",
        "step/optimizer",
        "step/outer",
        "step/update",
    ]
)


def metric_axis(name):
    """Return a business coordinate; the W&B transport step is never an axis."""
    if name.startswith("step/"):
        return None
    if name in ("eval/samples", "rollout/filtered_reward_counts"):
        return "step/outer"
    if name.startswith("eval/"):
        return "step/optimizer"
    if name.startswith(("optim/", "advantage/", "correction/", "distill/")):
        return "step/update"
    if name.startswith("policy/") and not name.startswith(("policy/anchor_behavior/", "policy/entropy/")):
        return "step/update"
    return "step/outer"


def is_update_metric(name):
    """Recognize already finalized update observations, including task advantages."""
    return name.startswith(("policy/", "advantage/", "correction/", "distill/", "optim/"))


def finite_metrics(metrics):
    """Omit undefined numeric observations rather than replacing them with zero."""
    return {k: float(v) for k, v in metrics.items() if v is not None and math.isfinite(float(v))}


def outer_metrics(metrics):
    """Project internal trainer diagnostics onto the new output contract."""
    result = {new: metrics[old] for old, new in OUTER_RENAMES.items() if old in metrics}
    for key, value in metrics.items():
        if key in CANONICAL_FIELDS or key.startswith("eval/"):
            result[key] = value
    return finite_metrics(result)


def dataset_slug(name):
    """Create a stable dataset component, with explicit aliases for math sets."""
    aliases = {"MATH-500": "math500", "AMC23": "amc23", "AIME25": "aime25"}
    slug = aliases.get(str(name), re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_"))
    if not slug:
        raise ValueError(f"Dataset name has no usable metric component: {name!r}")
    return slug


def evaluation_metrics(data_sources, uids, infos, expected_counts=None):
    """Preserve prompt-weighted accuracy and zero-pad failed validation sessions.

    Missing/malformed scores are absent observations, not wrong answers. Expected
    counts use native (data_source, uid) prompt identities from the trainer.
    """
    field = "acc" if "acc" in infos or not len(data_sources) else "reward"
    values = infos.get(field, [])
    if len(values) != len(data_sources) or len(uids) != len(data_sources):
        raise ValueError("Validation fields must have identical row counts")
    groups = defaultdict(list)
    for source, uid, value in zip(data_sources, uids, values, strict=True):
        key = (str(source), str(uid))
        groups[key]  # preserve prompts with only failed observations
        if value is not None:
            value = float(value)
            if not math.isfinite(value):
                raise ValueError("Validation accuracy/reward must be finite")
            groups[key].append(value)
    expected = {(str(s), str(u)): int(n) for (s, u), n in (expected_counts or {}).items()}
    for key in expected:
        groups[key]
    by_source = defaultdict(list)
    failed = defaultdict(int)
    counts = defaultdict(int)
    for key, observed in groups.items():
        count = expected.get(key, len(observed))
        if count < len(observed) or count < 0:
            raise ValueError("More validation observations than expected sessions")
        if count == 0:
            continue
        source = key[0]
        by_source[source].append(sum(observed) / count)
        failed[source] += count - len(observed)
        counts[source] += count
    result, names = {}, {}
    for source, means in by_source.items():
        slug = dataset_slug(source)
        if slug in names and names[slug] != source:
            raise ValueError(f"Dataset metric name collision: {source!r}, {names[slug]!r}")
        names[slug] = source
        prefix = f"eval/{slug}/"
        result[prefix + ("acc_mean" if field == "acc" else "score_mean")] = sum(means) / len(means)
        result[prefix + "failed_count"] = failed[source]
        result[prefix + "response_count"] = counts[source]
    return result
