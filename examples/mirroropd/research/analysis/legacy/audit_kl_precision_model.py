#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Test conditional KL perturbation models and arithmetic resolution.

All inputs are synthetic. Error floors are conditional expectations or measured
resolution for these workloads, never a convergence bound for a trained model.
"""

import argparse
import json
import math
from pathlib import Path

import torch


def kl(zq, zp):
    lq, lp = zq.log_softmax(-1), zp.log_softmax(-1)
    return (lq.exp() * (lq - lp)).sum(-1)


def stable_kl(zq, zp):
    """Research-only normalized KL identity; inputs here have bounded log-ratios."""
    lq, lp = zq.log_softmax(-1), zp.log_softmax(-1)
    r = lq - lp
    small = r.square() * (0.5 + r * (-1 / 6 + r * (1 / 24 + r * (-1 / 120 + r / 720))))
    f = torch.where(r.abs() < 0.01, small, torch.expm1(-r) + r)
    return (lq.exp() * f).sum(-1)


def weighted_variance(x, p):
    return (p * x.square()).sum(-1) - (p * x).sum(-1).square()


def profiles(vocab, rows, device):
    diffuse = torch.randn(rows, vocab, dtype=torch.float64, device=device)
    peaked = torch.randn_like(diffuse) - 18
    peaked[:, :3] = torch.tensor([0.0, -1.5, -5.0], dtype=torch.float64, device=device)
    return {"diffuse": diffuse, "peaked": peaked}


def noise_experiment(device, trials):
    results = []
    bases = {
        "uniform": torch.zeros(64, dtype=torch.float64, device=device),
        "peaked": torch.tensor([0.0, -1.5, -5.0] + [-18.0] * 61, dtype=torch.float64, device=device),
    }
    for name, z in bases.items():
        p = z.softmax(-1)
        concentration = (1 - p.square().sum()).item()
        for delta in [2**-7, 2**-5, 2**-4, 2**-3]:
            for rho in [0.0, 0.5, 0.9, 1.0]:
                et = (torch.rand(trials, 64, dtype=torch.float64, device=device) - 0.5) * delta
                independent = (torch.rand_like(et) - 0.5) * delta
                es = torch.where(torch.rand_like(et) < rho, et, independent)
                measured = stable_kl(z + es, z + et)
                pred = (1 - rho) * delta**2 / 12 * concentration
                quadratic = 0.5 * weighted_variance(es - et, p)
                radius = (es - et).amax(-1) - (es - et).amin(-1)
                assert bool(torch.all(measured <= radius.square() / 8 + 1e-12))
                if rho == 1:
                    assert measured.abs().max().item() == 0
                results.append(
                    {
                        "profile": name,
                        "step": delta,
                        "rho": rho,
                        "concentration": concentration,
                        "predicted_mean": pred,
                        "measured_mean": measured.mean().item(),
                        "quadratic_mean": quadratic.mean().item(),
                        "mc_standard_error": measured.std().item() / math.sqrt(trials),
                        "measured_over_predicted": measured.mean().item() / pred if pred else None,
                    }
                )
    return results


def arithmetic_experiment(device, vocab, rows):
    results = []
    for name, z in profiles(vocab, rows, device).items():
        noise = torch.randn_like(z)
        variance = weighted_variance(noise, z.softmax(-1)).mean().item()
        for target in [1.5e-3, 4e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-10, 0.0]:
            scale = math.sqrt(2 * target / variance)
            for _ in range(5 if target else 0):
                scale *= math.sqrt(target / stable_kl(z + scale * noise, z).mean().item())
            for shift in [0.0, 16.0]:
                q, p = z + scale * noise + shift, z + shift
                ref = stable_kl(q, p)
                variants = {
                    "naive_fp32": kl(q.float(), p.float()),
                    "stable_fp32": stable_kl(q.float(), p.float()),
                    "bf16_logits_fp32": kl(q.bfloat16().float(), p.bfloat16().float()),
                }
                result = {
                    "profile": name,
                    "target": target,
                    "shift": shift,
                    "reference_mean": ref.mean().item(),
                    "reference_min": ref.min().item(),
                    "reference_max": ref.max().item(),
                }
                for method, value in variants.items():
                    error = value.double() - ref
                    result[method] = {
                        "mean": value.double().mean().item(),
                        "max_abs_error": error.abs().max().item(),
                        "rms_error": error.square().mean().sqrt().item(),
                        "negative_rows": int((value < 0).sum()),
                        "zero_rows": int((value == 0).sum()),
                    }
                if not target:
                    assert all(bool(torch.all(v == 0)) for v in variants.values())
                results.append(result)
    return results


def rounding_variance(x):
    step = torch.pow(2.0, torch.floor(torch.log2(x.abs().clamp_min(1e-30))) - 7)
    return step.square() / 12


def projection_experiment(device):
    h = torch.randn(128, 128, dtype=torch.float64, device=device)
    w = torch.randn(1024, 128, dtype=torch.float64, device=device) / math.sqrt(128)
    z = h @ w.T
    quantized = h.bfloat16().double() @ w.bfloat16().double().T
    p = z.softmax(-1)
    variance_h, variance_w = rounding_variance(h), rounding_variance(w)
    hidden_term = (variance_h * (p @ w.square() - (p @ w).square())).sum(-1)
    weight_variance = h.square() @ variance_w.T
    weight_term = (p * (1 - p) * weight_variance).sum(-1)
    measured = stable_kl(quantized, z)
    return {
        "rows": len(h),
        "hidden": h.shape[1],
        "vocab": len(w),
        "independent_rounding_prediction": (0.5 * (hidden_term + weight_term)).mean().item(),
        "actual_quantization_kl_mean": measured.mean().item(),
        "actual_error_quadratic_mean": (0.5 * weighted_variance(quantized - z, p)).mean().item(),
        "identical_quantized_models_kl": stable_kl(quantized, quantized).max().item(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--vocab", type=int, default=151936)
    p.add_argument("--rows", type=int, default=8)
    p.add_argument("--trials", type=int, default=4096)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    results = {
        "scope": __doc__,
        "device": str(device),
        "torch": torch.__version__,
        "vocab": args.vocab,
        "rows": args.rows,
        "trials": args.trials,
        "seed": args.seed,
        "noise_model": noise_experiment(device, args.trials),
        "arithmetic": arithmetic_experiment(device, args.vocab, args.rows),
        "projection": projection_experiment(device),
    }
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    ratios = [r["measured_over_predicted"] for r in results["noise_model"] if r["measured_over_predicted"] is not None]
    print(
        json.dumps(
            {"noise_prediction_ratio_range": [min(ratios), max(ratios)], "projection": results["projection"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
