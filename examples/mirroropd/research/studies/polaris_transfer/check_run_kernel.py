#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Compare the exact committed run kernel to a dense CPU reference.

Also gives a labelled two-state mathematical counterexample: frozen-prefix
conditional-KL gradients need not equal the gradient through state occupancy.
This toy is not an explanation established from the run's trajectory data.
"""

import argparse
import json
import subprocess
import sys
import types
from pathlib import Path

import torch

COMMIT = "71a969c207213ea9013ad57b0dc9f0bd94c2fe0f"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.repo.resolve()))
    torch.set_num_threads(4)
    source = subprocess.check_output(
        ["git", "-C", str(a.repo), "show", COMMIT + ":verl/trainer/distillation/full_vocab.py"], text=True
    )
    mod = types.ModuleType("historical_full_vocab")
    exec(compile(source, "historical_full_vocab.py", "exec"), mod.__dict__)
    result = {"commit": COMMIT, "kernel_reference": []}
    for dtype in [torch.float32, torch.float64]:
        for backend in ["dense", "precompute", "streaming", "cached", "recompute"]:
            torch.manual_seed(42)
            hs = torch.randn(7, 3, dtype=dtype, requires_grad=True)
            ws = torch.randn(11, 3, dtype=dtype, requires_grad=True)
            ht = torch.randn(7, 5, dtype=dtype)
            wt = torch.randn(14, 5, dtype=dtype)
            lp = (hs @ ws.T).log_softmax(-1)
            lt = (ht @ wt.T).log_softmax(-1)[:, :11]
            ref = (lp.exp() * (lp - lt)).sum()
            gref = torch.autograd.grad(ref, (hs, ws))
            out = mod.full_reverse_kl(hs, ws, ht, wt, backend=backend, token_chunk=3, vocab_chunk=4)
            grad = torch.autograd.grad(out, (hs, ws))
            result["kernel_reference"].append(
                dict(
                    dtype=str(dtype),
                    backend=backend,
                    loss_abs_error=float(abs(ref - out)),
                    gradient_max_abs_error=max(float((x - y).abs().max()) for x, y in zip(grad, gref, strict=False)),
                )
            )
    theta = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    prob = theta.sigmoid()
    immediate = prob * torch.log(prob / 0.5) + (1 - prob) * torch.log((1 - prob) / 0.5)
    toy_fixed = immediate + prob.detach() * 0 + (1 - prob.detach()) * 1
    toy_total = immediate + prob * 0 + (1 - prob) * 1
    result["illustrative_two_state"] = {
        "downstream_KL_A": 0,
        "downstream_KL_B": 1,
        "prob_A": float(prob.detach()),
        "frozen_prefix_gradient": float(torch.autograd.grad(toy_fixed, theta, retain_graph=True)[0]),
        "occupancy_aware_gradient": float(torch.autograd.grad(toy_total, theta)[0]),
    }
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
