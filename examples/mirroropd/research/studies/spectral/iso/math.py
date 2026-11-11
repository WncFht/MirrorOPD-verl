# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Exact ISO paper reconstruction, transport and frame composition operators."""

import math

import torch


def polar(matrix):
    u, _, vh = torch.linalg.svd(matrix, full_matrices=False, driver="gesvd" if matrix.is_cuda else None)
    return u @ vh


def tangent(anchor, displacement):
    cross = anchor.T @ displacement
    return displacement - anchor @ ((cross + cross.T) / 2)


def spectrum_edit(original, restore, reverse, norm_ratio, kind, alpha):
    """Reset one spectral intervention from the immutable target weight."""
    if kind == "norm-only":
        return original / norm_ratio
    if kind == "shape-only":
        return (original + restore.to(original.device)) * norm_ratio
    if kind == "restore":
        return original + (1 - alpha) * restore.to(original.device)
    if kind == "reverse":
        return original + reverse.to(original.device)
    raise ValueError(f"Unknown spectral intervention: {kind}")


def transport(incoming, outgoing, fraction):
    """Paper's four rank-r reconstructions; no full-space projectors allocated."""
    u0, s0, vh0 = incoming
    u1, s1, vh1 = outgoing
    rank = max(1, math.floor(fraction * len(s0)))
    a = (u1[:, :rank] * s1[:rank]) @ vh1[:rank]
    b = (u0[:, :rank] * s0[:rank]) @ vh0[:rank]
    denominator = float((a - b).norm())
    left = u0[:, :rank] @ (u0[:, :rank].T @ a)
    right = (a @ vh0[:rank].T) @ vh0[:rank]
    mix = (left @ vh0[:rank].T) @ vh0[:rank]
    errors = dict(
        mix=float((a - mix).norm()),
        left=float((a - left).norm()),
        right=float((a - right).norm()),
        iso=float((s1[:rank] - s0[:rank]).norm()),
    )
    gaps = [float(s[rank - 1] - s[rank]) if rank < len(s) else None for s in (s0, s1)]
    admissible = all(gap is None or gap > float(s[0]) * 1e-12 for gap, s in zip(gaps, (s0, s1), strict=True))
    ratios = {k: value / denominator if denominator else None for k, value in errors.items()}
    if admissible and denominator and max(ratios.values()) > 1 + 1e-6:
        raise ValueError(f"Transport feasibility bound violated: {ratios}")
    return dict(
        rank=rank,
        fraction=fraction,
        incoming_gap=gaps[0],
        outgoing_gap=gaps[1],
        admissible=admissible,
        truncated_update_norm=denominator,
        errors=errors,
        unexplained=ratios,
        incoming_tail_energy=float(s0[rank:].square().sum()),
        outgoing_tail_energy=float(s1[rank:].square().sum()),
    )


def subspaces(incoming, outgoing, ranks=(16, 64)):
    result = {}
    for requested in ranks:
        rank = min(requested, len(incoming[1]))
        left = torch.linalg.svdvals(incoming[0][:, :rank].T @ outgoing[0][:, :rank]).clamp(0, 1)
        right = torch.linalg.svdvals(incoming[2][:rank] @ outgoing[2][:rank].T).clamp(0, 1)
        result[str(requested)] = dict(
            rank=rank,
            left_cosines=left.tolist(),
            right_cosines=right.tolist(),
            left_overlap=float(left.square().mean()),
            right_overlap=float(right.square().mean()),
        )
    return result


def compose(base, experts, keep=0.9, ridge=1e-12, clip=(0.0, 1.5), uniform=False):
    """Paper retention-aware ISO merger for one matrix, excluding task-vector baselines."""
    u0, s0, vh0 = base
    v0 = vh0.T
    deltas, proxies = [], []
    count = round(keep * len(s0))
    for u, _, vh in experts:
        signs = torch.sign((u0 * u).sum(0))
        signs[signs == 0] = 1
        du, dv = tangent(u0, u * signs - u0), tangent(v0, vh.T * signs - v0)
        du[:, count:] = 0
        dv[:, count:] = 0
        deltas.append((du, dv))
        proxies.append((du * s0) @ vh0 + (u0 * s0) @ dv.T)
    gram = torch.stack([torch.stack([(a * b).sum() for b in proxies]) for a in proxies])
    active = gram.diagonal() > 0
    coefficients = torch.zeros(len(experts), dtype=s0.dtype, device=s0.device)
    if active.any():
        sub = gram[active][:, active]
        coefficients[active] = torch.linalg.solve(
            sub + ridge * torch.eye(len(sub), device=sub.device, dtype=sub.dtype), sub.diagonal()
        ).clamp(*clip)
    if uniform:
        coefficients.fill_(1)
    du = sum(c * delta[0] for c, delta in zip(coefficients, deltas, strict=True))
    dv = sum(c * delta[1] for c, delta in zip(coefficients, deltas, strict=True))
    merged = (polar(u0 + tangent(u0, du)) * s0) @ polar(v0 + tangent(v0, dv)).T
    return merged, dict(coefficients=coefficients.tolist(), gram=gram.tolist(), keep=keep, ridge=ridge, clip=clip)
