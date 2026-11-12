# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded bundles of detached logit requests through one retained forward.

Production KL rows and projection accumulation remain the numerical source.
Bundles share vocabulary projections; only hidden/head cotangents survive a
token block. The caller controls how many full head cotangents fit in memory.
"""

import hashlib

import torch

from verl.trainer.distillation.rows import precompute_kl_rows
from verl.trainer.distillation.vjp import add_product

from .reductions import response_denominator


def split_request(live, initial):
    """Orthogonal logit projection, including zero anchors and signed scales."""
    p0 = initial.double().square().sum(-1)
    dot = (live.double() * initial.double()).sum(-1)
    alpha = torch.where(p0 > 0, dot / p0.clamp_min(1e-300), 0).to(live.dtype)
    parallel = alpha[:, None] * initial
    return parallel, live - parallel, alpha


def token_signs(row_id, length, probes, seed, *, device):
    """Stable per-question independent probes; identical across checkpoints."""
    key = hashlib.sha256(f"{seed}:{row_id}".encode()).digest()
    generator = torch.Generator(device="cpu").manual_seed(int.from_bytes(key[:8], "little") % (2**63 - 1))
    return (torch.randint(0, 2, (probes, length), generator=generator) * 2 - 1).to(device)


@torch.no_grad()
def cotangents(hs, ws, ht, wt, initial, *, fields, cap, token_chunk=128, bundle=4, signs=None, reduction="fixed_cap"):
    """Yield (field, dH, dW, token moments) for native/parallel/sign probes.

    H/W must also be supplied to autograd.backward by the caller. This preserves
    tied-head contributions through both paths without a surrogate KL formula.
    The first field owns native loss/P and anchor projection statistics.
    """
    if cap < len(hs) or min(cap, token_chunk, bundle) < 1:
        raise ValueError("positive cap/chunks/bundle and at most cap positions required")
    if not fields or len(set(fields)) != len(fields):
        raise ValueError("fields must be nonempty and unique")
    for name in fields:
        signed = name.startswith(("sign-", "frozen-sign-"))
        if name not in ("live", "parallel", "perpendicular", "frozen") and not signed:
            raise ValueError(f"Unknown request field {name}")
        if signed and (signs is None or not 0 <= int(name.rsplit("-", 1)[1]) < len(signs)):
            raise ValueError("missing sign probe")
    denominator = response_denominator(len(hs), cap, reduction)
    h0, w0 = initial
    dtype = torch.float64 if hs.dtype == torch.float64 else torch.float32
    for offset in range(0, len(fields), bundle):
        names = fields[offset : offset + bundle]
        dh = [torch.zeros_like(hs, dtype=dtype) for _ in names]
        dw = [torch.zeros_like(ws, dtype=dtype) for _ in names]
        moments = torch.zeros(7, dtype=torch.float64, device=hs.device)
        for start in range(0, len(hs), token_chunk):
            sl = slice(start, start + token_chunk)
            loss, live = precompute_kl_rows(hs[sl], ws, ht[sl], wt)
            _, anchor = precompute_kl_rows(h0[sl], w0, ht[sl], wt)
            parallel, perpendicular, alpha = split_request(live, anchor)
            if offset == 0:
                p, p0 = live.double().square().sum(-1), anchor.double().square().sum(-1)
                pp = parallel.double().square().sum(-1)
                perp = (live.double() - parallel.double()).square().sum(-1)
                moments += torch.stack(
                    (loss.double(), p, p0, pp, perp, torch.where(alpha < 0, pp, 0), alpha.double()), -1
                ).sum(0)
            for index, name in enumerate(names):
                request = {"live": live, "parallel": parallel, "perpendicular": perpendicular, "frozen": anchor}.get(
                    name
                )
                if request is None:
                    base = anchor if name.startswith("frozen-sign-") else live
                    request = base * signs[int(name.rsplit("-", 1)[1]), sl, None]
                request = request / denominator
                add_product(dh[index][sl], request, ws)
                add_product(dw[index], request.T, hs[sl])
        for index, name in enumerate(names):
            yield name, dh[index], dw[index], moments / denominator if offset + index == 0 else None


class GradientBank:
    """Device-resident group sums; export only completed group means to CPU."""

    def __init__(self, model, fields):
        self.parameters = tuple(model.parameters())
        self.sums = {name: [torch.zeros_like(p) for p in self.parameters] for name in fields}

    @torch.no_grad()
    def add(self, name):
        pairs = [(acc, p.grad) for acc, p in zip(self.sums[name], self.parameters, strict=True) if p.grad is not None]
        if pairs:
            accumulators, gradients = zip(*pairs, strict=True)
            torch._foreach_add_(accumulators, gradients)

    def export(self, name, count):
        from .runtime import flatten_tensors

        return flatten_tensors(self.parameters, self.sums[name], scale=1 / count)
