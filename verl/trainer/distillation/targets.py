# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Frozen power-mean targets, shared by sampled MDPO and full MDPO/BPO."""

import math

import torch


def mirror_signal(log_ratio, kappa, eta):
    if not math.isfinite(kappa) or not 0 <= kappa <= 1 or not math.isfinite(eta) or not 0 < eta <= 1:
        raise ValueError("Require finite kappa in [0,1], eta in (0,1]")
    if kappa == 0 or eta == 1:
        return eta * log_ratio
    x = kappa * log_ratio
    small = x.abs() < 0.5
    # Index the branches: eager torch.where would evaluate an overflowing expm1.
    result = torch.empty_like(x)
    result[small] = torch.log1p(eta * torch.expm1(x[small]))
    result[~small] = torch.logaddexp(x[~small] + math.log(eta), x.new_tensor(math.log1p(-eta)))
    return result / kappa


def build_log_weights(log_teacher, log_anchor=None, *, kappa, eta):
    """Return canonical b=log(q)+f, not normalized log(m); frozen value interface.

    A materialized b cannot retain increments below its dtype spacing. Training
    uses numerics.split_target (full vocabulary) or a separated sampled residual.
    """
    if eta == 1:
        return log_teacher.detach()
    if log_anchor is None:
        raise ValueError("eta<1 requires frozen anchor log probabilities")
    q, teacher = log_anchor.detach(), log_teacher.detach()
    return q + mirror_signal(teacher - q, kappa, eta)
