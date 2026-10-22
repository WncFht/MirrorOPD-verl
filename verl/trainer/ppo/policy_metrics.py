# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Read-only policy-pair and score-coefficient observations on response tokens."""

import math

import torch

from verl.utils.metric.observations import Observation


def distribution_observations(values, prefix, *, quantile=False):
    values = values.detach().double().reshape(-1)
    result = {
        prefix + "mean": Observation.mean(values),
        prefix + "std": Observation.std(values),
        prefix + "min": Observation.minimum(values),
        prefix + "max": Observation.maximum(values),
    }
    if quantile:
        for q, label in [(0.01, "p01"), (0.5, "p50"), (0.99, "p99")]:
            result[prefix + label] = Observation.quantile(values, q)
    return result


def pair_observations(current, other, mask, pair, *, quantile=False):
    mask = mask.bool()
    full = current.detach().double() - other.detach().double()
    x, y = current.detach().double()[mask], other.detach().double()[mask]
    z = x - y
    finite = torch.isfinite(x) & torch.isfinite(y)
    z = torch.where(finite.all(), z, z.new_full((), math.nan))
    prefix = f"policy/{pair}/"
    metrics = {
        prefix + "nonfinite_fraction": Observation.mean((~finite).double()),
        prefix + "valid_token_count": Observation.total(finite.sum()),
        prefix + "log_ratio_mean": Observation.mean(z),
        prefix + "log_ratio_std": Observation.std(z),
        prefix + "log_ratio_m2": Observation.mean(z.square()),
        prefix + "log_ratio_abs_mean": Observation.mean(z.abs()),
        prefix + "log_ratio_abs_max": Observation.maximum(z.abs()),
    }
    if pair == "anchor_behavior":
        metrics[prefix + "k3"] = Observation.mean(z.expm1() - z)
    if pair in ("current_behavior", "anchor_behavior"):
        metrics[prefix + "ratio_mean"] = Observation.log_moment(z, "log_mean")
        metrics[prefix + "ratio_std"] = Observation.log_moment(z, "log_std")
        metrics[prefix + "ess_fraction"] = Observation.ess(z)
        for name, event in (("ratio_lt_half_fraction", z < math.log(0.5)), ("ratio_gt_2_fraction", z > math.log(2))):
            metrics[prefix + name] = Observation.mean(torch.where(finite.all(), event.double(), math.nan))
        if quantile:
            metrics[prefix + "ratio_p99"] = Observation.ratio_p99(z)
    if quantile:
        for q, label in [(0.01, "p01"), (0.5, "p50"), (0.99, "p99")]:
            metrics[prefix + "log_ratio_" + label] = Observation.quantile(z, q)
    if full.ndim == 1:
        full, mask = full.unsqueeze(0), mask.unsqueeze(0)
    lengths = mask.sum(-1)
    keep = lengths > 0
    sums = torch.where(mask, full, 0.0).sum(-1)[keep]
    for label, values in [("sequence_log_ratio", sums), ("sequence_mean_log_ratio", sums / lengths[keep])]:
        metrics.update(distribution_observations(values, prefix + label + "_", quantile=quantile))
        metrics[prefix + label + "_m2"] = Observation.mean(values.square())
        metrics[prefix + label + "_abs_max"] = Observation.maximum(values.abs())
    return metrics


def applied_weight_observations(weights, used_mask, observation_mask, *, quantile=False):
    raw, used = observation_mask.bool(), used_mask.bool()
    effective = torch.where(used, weights.detach().double(), 0.0)
    values = effective[raw]
    result = distribution_observations(values, "correction/is/applied_", quantile=quantile)
    result["correction/is/applied_zero_fraction"] = Observation.mean((values == 0).double())
    result["correction/is/applied_nonfinite_fraction"] = Observation.mean((~torch.isfinite(values)).double())
    logs = values.log()
    result["correction/is/applied_ess_fraction"] = Observation.ess(logs)
    absolute = Observation.ess(logs)
    absolute.kind = "ess_absolute"
    result["correction/is/applied_ess"] = absolute
    result["correction/is/applied_token_count"] = Observation.total(raw.sum())
    result["correction/is/applied_nonzero_count"] = Observation.total((values > 0).sum())
    keep = raw.any(-1)
    seq_weights = effective.sum(-1)[keep] / raw.sum(-1)[keep]
    result.update(distribution_observations(seq_weights, "correction/is/applied_sequence_weight_", quantile=quantile))
    result["correction/is/applied_sequence_ess_fraction"] = Observation.ess(seq_weights.log())
    return result


def rejection_observations(used_mask, raw_mask):
    used, raw = used_mask.bool(), raw_mask.bool()
    keep = raw.any(-1)
    rejected = raw & ~used
    return {
        "correction/rejection/token_fraction": Observation.mean(rejected[raw].double()),
        "correction/rejection/sequence_any_fraction": Observation.mean(rejected.any(-1)[keep].double()),
        "correction/rejection/sequence_all_fraction": Observation.mean((~used.any(-1))[keep].double()),
        "correction/rejection/kept_token_count": Observation.total(used.sum()),
        "correction/rejection/kept_sequence_count": Observation.total(used.any(-1).sum()),
    }


def ppo_clip_observations(current, anchor, advantage, mask, low, high, dual, *, prefix, weights=None, scale=None):
    valid = mask.bool()
    ratio = (current.detach() - anchor.detach()).clamp(-20, 20).exp()
    advantage = advantage.detach()
    original = -advantage * ratio
    clipped = -advantage * ratio.clamp(1 - low, 1 + high)
    active = clipped > original
    dual_active = (advantage < 0) & (torch.maximum(original, clipped) > -advantage * dual)
    mass = (advantage * ratio).abs() * ((current.detach() - anchor.detach()).abs() <= 20)
    if scale is not None:
        mass = mass * scale
    if weights is not None:
        mass = mass * weights.detach().abs()
    inputs_finite = (
        torch.isfinite(current[valid]) & torch.isfinite(anchor[valid]) & torch.isfinite(advantage[valid])
    ).all()
    return {
        prefix + "active_fraction": Observation.mean(torch.where(inputs_finite, active[valid].double(), math.nan)),
        prefix + "dual_clip_fraction": Observation.mean(
            torch.where(inputs_finite, dual_active[valid].double(), math.nan)
        ),
        prefix + "active_mass": Observation.ratio(mass[valid & active].double().sum(), mass[valid].double().sum()),
        prefix + "dual_clip_mass": Observation.ratio(
            mass[valid & dual_active].double().sum(), mass[valid].double().sum()
        ),
    }


def update_observations(
    current,
    behavior,
    anchor,
    advantage,
    mask,
    *,
    component,
    quantile=False,
    teacher=None,
    mode=None,
    clip_active=None,
    coefficient_scale=None,
    observation_mask=None,
):
    """Construct mergeable diagnostics without changing the loss or its RNG."""
    mask = mask.bool()
    result = {}
    pair_mask = mask if observation_mask is None else observation_mask.bool()
    if behavior is not None:
        result.update(pair_observations(current, behavior, pair_mask, "current_behavior", quantile=quantile))
    if anchor is not None:
        result.update(pair_observations(current, anchor, pair_mask, "current_anchor"))
    adv = advantage.detach().double()[mask]
    prefix = f"advantage/{component}/"
    result[prefix + "mean"] = Observation.mean(adv)
    result[prefix + "min"] = Observation.minimum(adv)
    result[prefix + "max"] = Observation.maximum(adv)
    result[prefix + "std"] = Observation.std(adv)
    result[prefix + "zero_fraction"] = Observation.mean((adv == 0).double())
    result[prefix + "nonfinite_fraction"] = Observation.mean((~torch.isfinite(adv)).double())
    result[prefix + "token_count"] = Observation.total(mask.sum())
    result["rollout/training_tokens"] = Observation.total(mask.sum())
    if quantile:
        result[prefix + "abs_p99"] = Observation.quantile(adv.abs(), 0.99)
    result[prefix + "abs_mean"] = Observation.mean(adv.abs())
    # RMS is finalized from a mean-square state at the worker boundary.
    rms = Observation.mean(adv.square())
    rms.kind = "rms"
    result[prefix + "rms"] = rms
    result[prefix + "positive_fraction"] = Observation.mean(
        torch.where(torch.isfinite(adv).all(), (adv > 0).double(), math.nan)
    )
    if teacher is None or behavior is None:
        return result
    # Match the float32 arithmetic of the sampled-token objective, including
    # clamp derivative at the endpoints. These are score coefficients, not
    # norms of full parameter gradients or optimizer updates.
    cur = current.detach().float()[mask]
    beh = behavior.detach().float()[mask]
    teach = teacher.detach().float()[mask]
    a_current, a_rollout = teach - cur, teach - beh
    result.update(distribution_observations(cur - teach, "distill/student_teacher/sampled_log_gap_", quantile=quantile))
    finite_signal = (torch.isfinite(cur) & torch.isfinite(beh) & torch.isfinite(teach)).all()
    flip = (a_current * a_rollout < 0).double()
    result["distill/advantage/sign_flip_fraction"] = Observation.mean(torch.where(finite_signal, flip, math.nan))
    if mode not in ("token_ppo", "token_is", "reinforce"):
        return result
    z = cur - beh
    guarded = z.abs() > 20
    weight = torch.where(guarded, 0.0, z.clamp(-20, 20).exp()).double()
    scale = 1.0 if coefficient_scale is None else coefficient_scale.detach().double()[mask]
    c4 = weight * a_current.double() * scale
    c2 = weight * a_rollout.double() * scale
    c5 = a_current.double() * scale
    denom = c4.abs().sum()
    result["correction/is/coeff_change"] = Observation.ratio((c4 - c5).abs().sum(), denom)
    result["distill/advantage/source_coeff_change"] = Observation.ratio((c2 - c4).abs().sum(), denom)
    if mode != "reinforce":
        result["correction/log_ratio_guard/fraction"] = Observation.mean(
            torch.where(finite_signal, guarded.double(), math.nan)
        )
    if clip_active is not None:
        active = clip_active.detach().bool()[mask]
        result["correction/behavior_ppo/active_fraction"] = Observation.mean(
            torch.where(finite_signal, active.double(), math.nan)
        )
        coefficients = (weight * adv * scale).abs()
        result["correction/behavior_ppo/active_mass"] = Observation.ratio(
            coefficients[active].sum(), coefficients.sum()
        )
    return result
