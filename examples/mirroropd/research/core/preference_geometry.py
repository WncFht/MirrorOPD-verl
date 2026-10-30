# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Fixed-p0 token-preference geometry; no subtraction of endpoint KL reductions."""

import math

import torch


def moments(log_p0, log_qa, log_qb, log_pk):
    """Per-state centered moments on the entire student support.

    Teacher arguments retain their native log normalization but are sliced to student
    support. Centering removes a support-wide log-mass shift; extra mass is reported
    separately. Token dimensions must never be split before this operation.
    """
    if len({tuple(x.shape) for x in [log_p0, log_qa, log_qb, log_pk]}) != 1 or log_p0.ndim != 2:
        raise ValueError("Aligned state-by-student-vocabulary matrices are required")
    if not all(bool(torch.isfinite(x).all()) for x in [log_p0, log_qa, log_qb, log_pk]):
        raise ValueError("Nonfinite log probabilities")
    p = log_p0.double().exp()
    if not torch.allclose(p.sum(-1), torch.ones(p.shape[0], device=p.device, dtype=p.dtype), atol=1e-5, rtol=0):
        raise ValueError("p0 must cover the full normalized student support")
    p = p / p.sum(-1, keepdim=True)
    d, u = (log_qb.double() - log_qa.double()), (log_pk.double() - log_p0.double())
    d -= (p * d).sum(-1, keepdim=True)
    u -= (p * u).sum(-1, keepdim=True)
    return dict(
        d2=(p * d.square()).sum(-1),
        u2=(p * u.square()).sum(-1),
        du=(p * d * u).sum(-1),
        teacher_a_extra_mass=1 - log_qa.double().exp().sum(-1),
        teacher_b_extra_mass=1 - log_qb.double().exp().sum(-1),
    )


def geometry(d2, u2, du, *, minimum_norm_squared=1e-12):
    """Compute ratios after aggregating moments, never average per-state ratios."""
    if not all(math.isfinite(x) for x in [d2, u2, du]) or min(d2, u2) < 0:
        raise ValueError("Invalid preference moments")
    if d2 <= minimum_norm_squared:
        return dict(projection=None, cosine=None, residual=None, residual_fraction=None, identifiable=False)
    projection = du / d2
    residual = max(0.0, u2 - du * du / d2)
    return dict(
        projection=projection,
        cosine=du / math.sqrt(d2 * u2) if u2 > minimum_norm_squared else None,
        residual=residual,
        residual_fraction=residual / u2 if u2 > minimum_norm_squared else None,
        identifiable=True,
    )
