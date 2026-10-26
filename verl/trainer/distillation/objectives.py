# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Unscaled conditional objectives; eta and global reduction belong to the loss layer."""

import torch


def sampled_policy_terms(log_p, log_beta, advantage, *, mode, log_ratio_guard=None, clip_low=0.2, clip_high=0.2):
    """Shared stopped-signal surrogate; mode/guard are explicit algorithm choices.

    Returns token terms, optional PPO-active mask, and optional log-ratio guard
    mask. The proposal and signal never receive gradients.
    """
    advantage = advantage.detach()
    if mode == "reinforce":
        return -advantage * log_p, None, None
    if mode not in ("token_is", "token_ppo"):
        raise ValueError(f"Unsupported sampled surrogate: {mode}")
    log_ratio = log_p - log_beta.detach()
    guarded = None
    if log_ratio_guard is not None:
        guarded = log_ratio.abs() > log_ratio_guard
        log_ratio = log_ratio.clamp(-log_ratio_guard, log_ratio_guard)
    ratio = log_ratio.exp()
    terms = -ratio * advantage
    active = None
    if mode == "token_ppo":
        clipped = -ratio.clamp(1 - clip_low, 1 + clip_high) * advantage
        active = clipped > terms
        terms = torch.maximum(terms, clipped)
    return terms, active, guarded


def mdpo_sampled_terms(log_p, log_beta, b=None, *, residual=None):
    """Unclipped MDPO with a stopped, refreshed mirror residual.

    Encoded b is retained for callers supplying that representation. Production
    passes residual directly so small target increments never need to enter b.
    """
    if (b is None) == (residual is None):
        raise ValueError("Provide either encoded b or a separated target residual")
    signal = b - log_p if residual is None else residual
    terms, _, _ = sampled_policy_terms(log_p, log_beta, signal, mode="token_is")
    if not torch.isfinite(terms).all():
        raise ValueError("Non-finite sampled MDPO residual/importance weight")
    return terms


def full_vocab_rows(z, b, log_q=None, *, objective, diagnostics=False, log_teacher=None):
    """Return raw objective rows, analytic first-order dZ, detached diagnostic rows.

    Dense references differentiate the returned scalar normally. Optimized kernels
    consume dZ under no_grad. q and b are always frozen. No 1/eta here.
    """
    from verl.trainer.distillation.numerics import centered_vjp, kl_rows, log_softmax, mean, relative_difference

    b = b.detach()
    target_kl, dz = kl_rows(z, b)
    log_z = b.logsumexp(-1)
    if objective == "mdpo":
        rows = target_kl - log_z
    elif objective == "bpo":
        if log_q is None:
            raise ValueError("BPO requires a frozen anchor, including eta=1")
        lq = log_softmax(log_q.detach())
        pivot = lq.argmax(-1, keepdim=True)
        residual = relative_difference(z, b, pivot)
        dz, centered = centered_vjp(lq, residual)
        rows = mean(lq, centered.square()).squeeze(-1) / 2
    else:
        raise ValueError(f"Unknown mirror objective: {objective}")
    info = {}
    if diagnostics:
        info = {"target_kl": target_kl, "target_log_z": log_z}
        if log_teacher is not None:
            teacher = log_teacher.detach()
            info["teacher_kl"] = kl_rows(z, teacher)[0] - teacher.logsumexp(-1)
        if log_q is not None:
            info["anchor_kl"] = kl_rows(z, log_q.detach())[0]
    return rows, dz, {k: v.detach() for k, v in info.items()}
