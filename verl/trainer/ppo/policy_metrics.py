# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Read-only policy-pair and score-coefficient observations on response tokens."""

import math

import torch

from verl.utils.metric.observations import Observation


def pair_observations(current, other, mask, pair, *, quantile=False):
    """Observe one named probability pair without clipping its raw log ratio."""
    mask = mask.bool()
    x, y = current.detach().double()[mask], other.detach().double()[mask]
    z = x - y
    finite = torch.isfinite(x) & torch.isfinite(y)
    valid = finite.all()
    # Preserve an invalid observation instead of reporting a finite subset.
    z = torch.where(valid, z, z.new_full((), math.nan))
    prefix = f"policy/{pair}/"
    metrics = {prefix + "nonfinite_fraction": Observation.mean((~finite).double())}
    if pair == "anchor_behavior":
        metrics[prefix + "log_ratio_abs_mean"] = Observation.mean(z.abs())
        metrics[prefix + "k3"] = Observation.mean(z.expm1() - z)
    else:
        metrics[prefix + "log_ratio_m2"] = Observation.mean(z.square())
    metrics[prefix + "log_ratio_abs_max"] = Observation.maximum(z.abs())
    if pair == "current_behavior":
        metrics[prefix + "log_ratio_mean"] = Observation.mean(z)
        metrics[prefix + "ess_fraction"] = Observation.ess(z)
        for name, event in (("ratio_lt_half_fraction", z < math.log(0.5)), ("ratio_gt_2_fraction", z > math.log(2))):
            metrics[prefix + name] = Observation.mean(torch.where(valid, event.double(), math.nan))
        if quantile:
            metrics[prefix + "ratio_p99"] = Observation.ratio_p99(z)
    return metrics


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
):
    """Construct mergeable diagnostics without changing the loss or its RNG."""
    mask = mask.bool()
    result = {}
    if behavior is not None:
        result.update(pair_observations(current, behavior, mask, "current_behavior", quantile=quantile))
    if anchor is not None:
        result.update(pair_observations(current, anchor, mask, "current_anchor"))
    adv = advantage.detach().double()[mask]
    prefix = f"advantage/{component}/"
    result[prefix + "mean"] = Observation.mean(adv)
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
    finite_signal = (torch.isfinite(cur) & torch.isfinite(beh) & torch.isfinite(teach)).all()
    flip = (a_current * a_rollout < 0).double()
    result["distill/advantage/sign_flip_fraction"] = Observation.mean(torch.where(finite_signal, flip, math.nan))
    if mode not in ("token_ppo", "token_is", "reinforce"):
        return result
    z = cur - beh
    guarded = z.abs() > 20
    weight = torch.where(guarded, 0.0, z.clamp(-20, 20).exp()).double()
    c4, c2, c5 = weight * a_current.double(), weight * a_rollout.double(), a_current.double()
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
        coefficients = (weight * adv).abs()
        result["correction/behavior_ppo/active_mass"] = Observation.ratio(
            coefficients[active].sum(), coefficients.sum()
        )
    return result
