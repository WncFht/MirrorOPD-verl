# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU reverse-KL cancellation checks, separating value and first-order gradient.

The mathematical route matches full_vocab.precompute_kl_rows before its VJP
cast; this script isolates loss arithmetic and does not benchmark the backbone,
parameter gradients, distributed reduction, or optimizer updates.
"""

import argparse
import json
from pathlib import Path

import torch


def evaluate(student, teacher, dtype):
    # Fix the inputs first, then use these exact values for the FP64 reference.
    s, t = student.to(dtype), teacher.to(dtype)
    lp, lq = s.log_softmax(-1), t.log_softmax(-1)
    p, h = lp.exp(), lp - lq
    loss = (p * h).sum(-1)
    grad = p * (h - loss[:, None])
    sr, tr = s.double(), t.double()
    lpr, lqr = sr.log_softmax(-1), tr.log_softmax(-1)
    pr = lpr.exp()
    reference_loss = (pr * (lpr - lqr)).sum(-1)
    delta = sr - tr
    delta -= delta.gather(-1, sr.argmax(-1, keepdim=True))
    reference_grad = pr * (delta - (pr * delta).sum(-1, keepdim=True))
    norm = reference_grad.norm().item()
    error = (grad.double() - reference_grad).norm().item()
    return {
        "dtype": str(dtype),
        "reference_kl_mean": reference_loss.mean().item(),
        "actual_kl_mean": loss.float().mean().item(),
        "kl_per_row_mae": (loss.double() - reference_loss).abs().mean().item(),
        "kl_zero_fraction": (loss == 0).float().mean().item(),
        "kl_negative_fraction": (loss < 0).float().mean().item(),
        "gradient_relative_l2": error / norm if norm else None,
        "gradient_cosine": (
            (grad.double() * reference_grad).sum() / (grad.double().norm() * reference_grad.norm())
        ).item()
        if norm and grad.any()
        else None,
        "gradient_norm": grad.double().norm().item(),
        "reference_gradient_norm": norm,
        "gradient_max_abs_error": (grad.double() - reference_grad).abs().max().item(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    out = {"torch": torch.__version__, "device": "CPU", "scope": __doc__, "rows": []}
    teacher = torch.randn(16, 256) * 2
    noise = torch.randn_like(teacher)
    for dtype in [torch.float32, torch.bfloat16]:
        for alpha in [0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001, 0.000003, 0.0]:
            row = evaluate(teacher + alpha * noise, teacher, dtype)
            out["rows"].append({"setting": "256-class random logits", "alpha": alpha, **row})
    out["binary"] = []
    for delta in [0.02, 0.002, 0.0002, 0.00002, 0.000002]:
        row = evaluate(torch.tensor([[delta, -delta]]), torch.zeros(1, 2), torch.float32)
        out["binary"].append({"delta": delta, **row})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    for row in out["rows"]:
        print(row["dtype"], row["alpha"], row["reference_kl_mean"], row["actual_kl_mean"], row["gradient_relative_l2"])
    print("binary", json.dumps(out["binary"], indent=2))


if __name__ == "__main__":
    main()
