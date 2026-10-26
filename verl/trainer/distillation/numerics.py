# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Stable row algebra. Frozen targets stay separate from their small increments.

No eta scaling, batching or parameter-gradient accumulation belongs here.
All row formulas work in FP32 or FP64; low-precision casts happen at the GEMMs.
"""

import math
from functools import lru_cache

import torch
import torch.nn.functional as F


def working(x):
    return x if x.dtype in (torch.float32, torch.float64) else x.float()


def relative(x, pivot):
    return (x - x.gather(-1, pivot)).scatter(-1, pivot, 0.0)


class _RelativeDifference(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, pivot, anchor, eta):
        from verl.trainer.distillation.full_vocab_triton import difference

        ctx.save_for_backward(pivot)
        ctx.anchored, ctx.eta = anchor is not None, eta if anchor is not None else 1.0
        return difference(x, y, pivot, anchor=anchor, eta=eta)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, upstream):
        (pivot,) = ctx.saved_tensors
        other = upstream.scatter(-1, pivot, 0.0)
        gradient = other.scatter(-1, pivot, -other.sum(-1, keepdim=True))
        return gradient, -ctx.eta * gradient, None, -(1 - ctx.eta) * gradient if ctx.anchored else None, None


def relative_difference(x, y, pivot, *, anchor=None, eta=1.0):
    # CUDA performs only the few subtractions in FP64 registers: no dense FP64
    # intermediates. The Torch path is also used for independent FP64 checks.
    dtype = working(x).dtype
    if x.is_cuda and x.dtype == y.dtype == torch.float32 and x.is_contiguous():
        if anchor is None or (anchor.dtype == torch.float32 and anchor.is_contiguous()):
            return _RelativeDifference.apply(x, y, pivot, anchor, eta)
    rx, ry = relative(x.double(), pivot), relative(y.double(), pivot)
    if anchor is None:
        result = rx - ry
    else:
        ra = relative(anchor.double(), pivot)
        result = (rx - ra) - eta * (ry - ra)
    return result.to(dtype)


def log_softmax(x):
    """Keep the largest log probability's small nonzero normalization term."""
    x = working(x)
    pivot = x.argmax(-1, keepdim=True)
    shifted = relative(x, pivot)
    others = shifted.scatter(-1, pivot, -torch.inf)
    return shifted - F.softplus(others.logsumexp(-1, keepdim=True))


def expm1_minus_x(x):
    small = x.clamp(-0.125, 0.125)
    polynomial = torch.full_like(small, 1 / math.factorial(12))
    for n in range(11, 1, -1):
        polynomial = 1 / math.factorial(n) + small * polynomial
    polynomial = small.square() * polynomial
    safe = x.clamp(-80, 80)
    return torch.where(x.abs() < 0.125, polynomial, safe.expm1() - safe)


def log1p_minus_x(x):
    small = x.clamp(-0.0625, 0.0625)
    polynomial = torch.full_like(small, -1 / 14)
    for n in range(13, 1, -1):
        polynomial = ((-1) ** (n + 1)) / n + small * polynomial
    polynomial = small.square() * polynomial
    safe = x.clamp(min=-0.5)
    return torch.where(x.abs() < 0.0625, polynomial, safe.log1p() - safe)


def mean(log_w, x):
    w = log_w.exp()
    return (w * x).sum(-1, keepdim=True) / w.sum(-1, keepdim=True)


def cumulant(log_w, x):
    """log E_w exp(x) - E_w x, retaining its quadratic near-zero term.

    The remainder expm1(x)-x is evaluated before reduction. A shifted LSE
    handles large gaps; the near branch never subtracts large normalizers.
    """
    w = log_w.exp()
    mass = w.sum(-1, keepdim=True)
    pivot = log_w.argmax(-1, keepdim=True)
    x = x - x.gather(-1, pivot)
    center = x - (w * x).sum(-1, keepdim=True) / mass
    a = (w * center).sum(-1, keepdim=True) / mass
    weighted_exp = (log_w + center).clamp(max=40).exp()
    remainder = torch.where(center.abs() < 0.125, w * expm1_minus_x(center), weighted_exp - w * (1 + center))
    b = remainder.sum(-1, keepdim=True) / mass
    near = (log_w + center).amax(-1, keepdim=True) < 40
    near = near & (a.abs() < 0.5) & (b.abs() < 0.5)
    safe_a, safe_b = a.clamp(-0.5, 0.5), b.clamp(-0.5, 0.5)
    close = torch.log1p(safe_b / (1 + safe_a)) + log1p_minus_x(safe_a)
    far = (log_w + center).logsumexp(-1, keepdim=True) - mass.log() - a
    return torch.where(near, close, far).squeeze(-1)


def centered_vjp(log_w, residual):
    """Normalize weights and reconstruct the pivot before the output dtype cast."""
    w = log_w.exp()
    mass = w.sum(-1, keepdim=True)
    pivot = log_w.argmax(-1, keepdim=True)
    residual = residual - residual.gather(-1, pivot)
    centered = residual - (w * residual).sum(-1, keepdim=True) / mass
    g = w / mass * centered
    other = g.scatter(-1, pivot, 0.0)
    return other.scatter(-1, pivot, -other.sum(-1, keepdim=True)), centered


def support_correction(teacher, vocab):
    if teacher.shape[-1] == vocab:
        return teacher.new_zeros(teacher.shape[:-1])
    pivot = teacher[..., :vocab].argmax(-1, keepdim=True)
    shifted = teacher - teacher.gather(-1, pivot)
    return F.softplus(shifted[..., vocab:].logsumexp(-1) - shifted[..., :vocab].logsumexp(-1))


def kl_rows(z, teacher):
    """Full teacher KL and analytic student logit VJP; extra teacher rows included."""
    z, teacher = working(z), working(teacher).detach()
    lp = log_softmax(z)
    pivot = z.argmax(-1, keepdim=True)
    gap = relative_difference(z, teacher[..., : z.shape[-1]], pivot)
    value = cumulant(lp, -gap) + support_correction(teacher, z.shape[-1])
    gradient, _ = centered_vjp(lp, gap)
    return value, gradient


def power_increment(x, kappa, eta):
    """Power-mean log increment; eta may also be a row tensor of mixture weights."""
    if kappa == 0:
        return eta * x
    if not isinstance(eta, torch.Tensor):
        if eta == 1:
            return x
        if eta == 0:
            return torch.zeros_like(x)
        log_eta, log_complement = math.log(eta), math.log1p(-eta)
    else:
        log_eta, log_complement = eta.log(), torch.log1p(-eta)
    eta = torch.as_tensor(eta, dtype=x.dtype, device=x.device)
    y = kappa * x
    near = torch.log1p(eta * torch.expm1(y.clamp(-0.5, 0.5)))
    far = torch.logaddexp(torch.as_tensor(log_complement, dtype=x.dtype, device=x.device), log_eta + y)
    return torch.where(y.abs() < 0.5, near, far) / kappa


def power_remainder(log_ratio, kappa, eta):
    """f(log_ratio)-eta*log_ratio without losing the second-order remainder."""
    if kappa == 0 or eta == 1:
        return torch.zeros_like(log_ratio)
    # Bernoulli cumulant symmetry avoids cancellation as eta approaches one.
    e, x = (eta, kappa * log_ratio) if eta <= 0.5 else (1 - eta, -kappa * log_ratio)
    safe = x.clamp(-0.5, 0.5)
    a = e * safe.expm1()
    near = e * expm1_minus_x(safe) + log1p_minus_x(a)
    far = torch.logaddexp(x.new_tensor(math.log1p(-e)), x + math.log(e)) - e * x
    return torch.where(x.abs() < 0.5, near, far) / kappa


@lru_cache(maxsize=128)
def power_gap_coefficients(kappa, eta):
    """Bernoulli cumulants, with 1-kappa factored before float32 conversion."""
    e = min(eta, 1 - eta)
    cumulants = [0.0, e]
    coefficients = []
    for n in range(2, 17):
        cumulants.append(e - e * sum(math.comb(n - 1, j - 1) * cumulants[j] for j in range(1, n)))
        factor = (1 - kappa) * sum(kappa**j for j in range(n - 1))
        coefficients.append(cumulants[n] * factor / math.factorial(n))
    return tuple(coefficients)


def power_gap(log_ratio, kappa, eta):
    """log(arithmetic mean)-log(power mean), preserving the small Jensen deficit."""
    if kappa == 1 or eta == 1:
        return torch.zeros_like(log_ratio)
    c = power_gap_coefficients(kappa, eta)
    x = log_ratio if eta <= 0.5 else -log_ratio
    small = x.clamp(-0.5, 0.5)
    polynomial = torch.full_like(x, c[-1])
    for coefficient in reversed(c[:-1]):
        polynomial = coefficient + small * polynomial
    polynomial = small.square() * polynomial
    wide = log_ratio.double() if kappa > 0.99 else log_ratio
    far = power_remainder(wide, 1.0, eta) - power_remainder(wide, kappa, eta)
    if kappa > 0:
        # Factor out the winning endpoint for large log ratios; neither term
        # contains the large common linear part of the log ratio.
        xabs = wide.abs()
        a = torch.where(wide >= 0, 1 - eta, eta)
        l0 = torch.where(wide >= 0, math.log(eta), math.log1p(-eta))
        l1 = torch.where(wide >= 0, math.log1p(-eta), math.log(eta))

        def h(y):
            near = torch.log1p(a * torch.expm1(-y.clamp(max=0.5)))
            return torch.where(y < 0.5, near, torch.logaddexp(l0, l1 - y))

        factored = h(xabs) - h(kappa * xabs) / kappa
        far = torch.where(xabs > 40, factored, far)
        if kappa > 0.99:
            # Midpoint divided difference; its interval in tilted log-odds is
            # at most 1e-4, making the O(interval^2) error below FP32 precision.
            midpoint = (1 + kappa) / 2
            odds = math.log(eta) - math.log1p(-eta) + midpoint * wide
            lp, l0p = -F.softplus(-odds), -F.softplus(odds)
            divergence = lp.exp() * (lp - math.log(eta)) + l0p.exp() * (l0p - math.log1p(-eta))
            divided = (1 - kappa) / (midpoint * midpoint) * divergence
            far = torch.where((1 - kappa) * xabs < 1e-4, divided, far)
    return torch.where(log_ratio.abs() < 0.5, polynomial, far.to(log_ratio.dtype))


def target_log_normalizer(lq, log_ratio, correction, *, kappa, eta):
    """log Z from an arithmetic-mixture mass minus a nonnegative Jensen deficit."""
    if eta == 1:
        return -correction
    near_mass = torch.log1p(eta * torch.expm1(-correction.clamp(max=0.5)))
    far_mass = torch.logaddexp(correction.new_tensor(math.log1p(-eta)), math.log(eta) - correction)
    log_mass = torch.where(correction < 0.5, near_mass, far_mass)
    if kappa == 1:
        return log_mass
    gap = power_gap(log_ratio, kappa, eta)
    log_mix = lq + power_increment(log_ratio, 1.0, eta)
    deficit = (log_mix.exp() * (-torch.expm1(-gap))).sum(-1)
    ratio = deficit / log_mass.exp()
    close = log_mass + torch.log1p(-ratio.clamp(max=0.5))
    far = (lq + power_increment(log_ratio, kappa, eta)).logsumexp(-1)
    return torch.where(ratio < 0.5, close, far)


def split_target(anchor, teacher, *, kappa, eta, normalizer=True):
    """Return a small relative increment and the frozen canonical log Z_m.

    b is represented as anchor logits + increment + a row constant; no small
    increment is ever added to a large log weight before forming the residual.
    """
    anchor, teacher = working(anchor).detach(), working(teacher).detach()
    pivot = anchor.argmax(-1, keepdim=True)
    lq = log_softmax(anchor)
    delta = relative_difference(teacher[..., : anchor.shape[-1]], anchor, pivot)
    if not normalizer and (kappa == 0 or eta == 1):
        return eta * delta, None, lq, None
    correction = support_correction(teacher, anchor.shape[-1])
    kl_qt = cumulant(lq, delta) + correction
    log_ratio = delta - mean(lq, delta) - kl_qt[..., None]
    if eta == 1:
        return delta, -correction, lq, None
    choose_teacher = None
    if kappa == 0:
        increment = eta * delta
    else:
        odds = math.log(eta) - math.log1p(-eta) + kappa * log_ratio.gather(-1, pivot)
        choose_teacher = odds > 0
        small_weight = torch.sigmoid(-odds.abs())
        increment = power_increment(torch.where(choose_teacher, -delta, delta), kappa, small_weight)
    log_z = target_log_normalizer(lq, log_ratio, correction, kappa=kappa, eta=eta) if normalizer else None
    return increment, log_z, lq, choose_teacher


def mirror_logits_rows(z, anchor, teacher, *, objective, kappa, eta, diagnostics=False):
    """Stable scalar and VJP from raw projected logits, including dense autograd."""
    z = working(z)
    need_kl = objective == "mdpo" or diagnostics
    increment, log_z, lq, choose_teacher = split_target(anchor, teacher, kappa=kappa, eta=eta, normalizer=need_kl)
    pivot = anchor.argmax(-1, keepdim=True)
    if eta == 1:
        residual = relative_difference(z, teacher.detach()[..., : z.shape[-1]], pivot)
    elif kappa == 0:
        residual = relative_difference(z, teacher.detach()[..., : z.shape[-1]], pivot, anchor=anchor.detach(), eta=eta)
    else:
        base = torch.where(choose_teacher, teacher.detach()[..., : z.shape[-1]], anchor.detach())
        residual = relative_difference(z, base, pivot) - increment
    if need_kl:
        lp = log_softmax(z)
        target_kl = cumulant(lp, -residual)
    lw = lp if objective == "mdpo" else lq
    gradient, centered = centered_vjp(lw, residual)
    value = target_kl - log_z if objective == "mdpo" else mean(lw, centered.square()).squeeze(-1) / 2
    info = {}
    if diagnostics:
        info = {
            "target_kl": target_kl,
            "target_log_z": log_z,
            "anchor_kl": kl_rows(z, anchor)[0],
            "teacher_kl": kl_rows(z, teacher)[0],
        }
    return value, gradient, {k: v.detach() for k, v in info.items()}
