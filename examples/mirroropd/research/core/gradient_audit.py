# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Full-parameter gradient comparisons without retaining duplicate GPU models."""

import time

import torch


def snapshot(model):
    return {name: p.grad.detach().cpu().clone() for name, p in model.named_parameters() if p.grad is not None}


def compare(model, reference):
    sums = torch.zeros(4, device="cuda", dtype=torch.float64)
    for name, p in model.named_parameters():
        assert p.grad is not None and name in reference
        old, new = reference[name].reshape(-1), p.grad.reshape(-1)
        for start in range(0, len(old), 2**20):
            a = old[start : start + 2**20].to("cuda")
            b = new[start : start + 2**20]
            sums += torch.stack(
                [
                    a.square().sum(dtype=torch.float64),
                    b.square().sum(dtype=torch.float64),
                    (a * b).sum(dtype=torch.float64),
                    (a - b).square().sum(dtype=torch.float64),
                ]
            )
    aa, bb, ab, delta = sums.tolist()
    return dict(
        relative_l2=(delta / max(aa, 1e-30)) ** 0.5,
        cosine=ab / (aa * bb) ** 0.5 if aa * bb else float(delta == 0),
        reference_norm=aa**0.5,
        candidate_norm=bb**0.5,
        difference_norm=delta**0.5,
    )


def measure(model, call):
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    loss = call()
    torch.cuda.synchronize()
    return dict(loss=loss, seconds=time.perf_counter() - started, peak_gib=torch.cuda.max_memory_allocated() / 2**30)
