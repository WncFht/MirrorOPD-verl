#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Synthetic numerical audit, not a late-checkpoint/model accuracy measurement.

Hold logits fixed to isolate FP32 normalization error; separately quantize logits
to illustrate BF16 rounding. A projection probe compares FP32-output BF16-input
GEMM with FP64 on those SAME inputs, separating arithmetic from input rounding.
"""

import argparse
import json
import math
from pathlib import Path

import torch


def kl(zs, zt):
    ls, lt = zs.log_softmax(-1), zt.log_softmax(-1)
    return (ls.exp() * (ls - lt)).sum(-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    zt = torch.randn(8, 151936, dtype=torch.float64, device=device) - 18
    zt[:, :3] = torch.tensor([0.0, -1.5, -5.0], dtype=torch.float64, device=device)
    noise = torch.randn_like(zt)
    rows = []
    for target in [1.5e-3, 4e-4, 1e-5, 1e-6, 1e-7]:
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = (lo + hi) / 2
            if kl(zt + mid * noise, zt).mean().item() < target:
                lo = mid
            else:
                hi = mid
        zs = zt + (lo + hi) / 2 * noise
        ref = kl(zs, zt)
        for shift in [0.0, 16.0]:
            a, b = zs + shift, zt + shift
            variants = {
                "fp32_logits_fp32_kl": kl(a.float(), b.float()),
                "bf16_rounded_logits_fp32_kl": kl(a.bfloat16().float(), b.bfloat16().float()),
                "bf16_logits_bf16_kl": kl(a.bfloat16(), b.bfloat16()),
            }
            row = {"target": target, "common_logit_shift": shift, "fp64_mean": ref.mean().item()}
            for name, v in variants.items():
                row[name] = {
                    "mean": v.double().mean().item(),
                    "max_abs_error": (v.double() - ref).abs().max().item(),
                    "relative_mean_error": (v.double().mean() / ref.mean() - 1).item(),
                    "negative_rows": int((v < 0).sum()),
                }
            rows.append(row)
    z = zt.log_softmax(-1)
    result = {
        "scope": __doc__,
        "device": str(device),
        "torch_version": torch.__version__,
        "vocab": 151936,
        "rows": 8,
        "teacher_entropy_mean": (-z.exp() * z).sum(-1).mean().item(),
        "logit_trials": rows,
    }
    result["ulp"] = {
        str(v): {
            str(dtype): float(
                (
                    torch.nextafter(torch.tensor(v, dtype=dtype), torch.tensor(math.inf, dtype=dtype))
                    - torch.tensor(v, dtype=dtype)
                ).float()
            )
            for dtype in [torch.float32, torch.bfloat16]
        }
        for v in [0.0004, 0.01, 1.0, 16.0]
    }
    if device.type == "cuda":
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
        from verl.trainer.distillation.full_vocab import project

        h = torch.randn(8, 64, device=device).bfloat16()
        w = (torch.randn(151936, 64, device=device) / 8).bfloat16()
        s = (h.float() + 0.03 * torch.randn_like(h.float())).bfloat16()
        actual = kl(project(s, w), project(h, w))
        reference = kl(s.double() @ w.double().T, h.double() @ w.double().T)
        result["bf16_input_fp32_projection"] = {
            "reference_mean": reference.mean().item(),
            "actual_mean": actual.mean().item(),
            "max_abs_error": (actual.double() - reference).abs().max().item(),
        }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "entropy": result["teacher_entropy_mean"],
                "projection": result.get("bf16_input_fp32_projection"),
                "trials": [r for r in rows if r["target"] >= 4e-4],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
