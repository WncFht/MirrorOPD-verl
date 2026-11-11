# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Native reverse-KL logit derivatives and softmax geometry.

Requests here use the positive loss-gradient convention. Negate both source
and receiver requests to obtain descent requests; all quadratic diagnostics
are unchanged.
"""

import torch

from examples.mirroropd.research.core.metrics import native_reverse_kl_terms


def position_signal(student_logits, teacher_logits):
    """Return reverse-KL and the exact logit derivative B for one position."""
    squeeze = student_logits.ndim == 1
    if squeeze:
        student_logits, teacher_logits = student_logits[None, :], teacher_logits[None, :]
    vocab = student_logits.shape[-1]
    kl, lp, lq, p = native_reverse_kl_terms(student_logits.float(), teacher_logits.float())
    # d KL(p(z)||q) / dz = p * ((log p - log q) - KL).
    b = p * (lp - lq[..., :vocab] - kl.detach().unsqueeze(-1))
    return (kl[0], b[0]) if squeeze else (kl, b)


def fisher_diagnostics(student_logits, teacher_logits):
    """Softmax Fisher attenuation and irreducible native-support KL at fixed prefixes.

    These are logit-simplex quantities, not parameter Fisher eigenvalues.
    The gain ||F r||^2/(r^T F r) is a Fisher Rayleigh quotient in [0, 1/2].
    """
    kl, lp, lq, p = native_reverse_kl_terms(student_logits.float(), teacher_logits.float())
    return _fisher_terms(kl, lp, lq, p, teacher_logits)


def _fisher_terms(kl, lp, lq, p, teacher_logits):
    vocab = p.shape[-1]
    centered = lp - lq[..., :vocab] - kl.unsqueeze(-1)
    variance = (p * centered.square()).sum(-1)
    pressure = (p * centered).square().sum(-1)
    support_floor = torch.logsumexp(teacher_logits.float(), -1) - torch.logsumexp(
        teacher_logits[..., :vocab].float(), -1
    )
    return dict(
        logratio_variance=variance,
        pressure=pressure,
        fisher_request_gain=pressure / variance.clamp_min(1e-30),
        fisher_trace=1 - p.square().sum(-1),
        native_support_floor=support_floor,
        conditional_rkl=kl - support_floor,
        student_max_probability=p.max(-1).values,
    )


def profile_and_signal(student_logits, teacher_logits, *, mass_edges=(0.0, 0.5, 0.9, 0.99, 1.0)):
    """Partition native request energy by cumulative student probability mass.

    Assign each token by its left cumulative-mass boundary. This keeps bins
    exhaustive even when one token crosses several boundaries.
    """
    kl, lp, lq, probability = native_reverse_kl_terms(student_logits[None].float(), teacher_logits[None].float())
    derivative = (probability * (lp - lq[:, : student_logits.shape[-1]] - kl[:, None]))[0]
    diagnostics = _fisher_terms(kl, lp, lq, probability, teacher_logits[None])
    p = probability[0]
    order = p.argsort(descending=True)
    sorted_p = p[order]
    left_mass = sorted_p.cumsum(0) - sorted_p
    energy = derivative[order].double().square()
    bins = []
    for lower, upper in zip(mass_edges[:-1], mass_edges[1:], strict=True):
        mask = (left_mass >= lower) & (left_mass < upper)
        if upper == 1:
            mask |= left_mass >= 1  # FP32 cumulative-sum rounding at the tail.
        bins.append(
            dict(
                lower=lower,
                upper=upper,
                tokens=int(mask.sum()),
                probability_mass=float(sorted_p[mask].double().sum()),
                pressure=float(energy[mask].sum()),
            )
        )
    profile = dict(
        loss=float(kl[0]),
        bins=bins,
        **{k: float(v[0]) for k, v in diagnostics.items()},
    )
    return profile, derivative


def probability_profile(student_logits, teacher_logits, *, mass_edges=(0.0, 0.5, 0.9, 0.99, 1.0)):
    """JSON-ready request statistics by cumulative student probability mass."""
    return profile_and_signal(student_logits, teacher_logits, mass_edges=mass_edges)[0]
