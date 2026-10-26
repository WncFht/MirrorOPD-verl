#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Near-target arithmetic audit on identical saved FP32 logits and target weights.

Logit interpolation isolates loss arithmetic; it is not a late checkpoint replay.
At alpha=0 the reference is nearly zero, so inspect absolute norms, not relative
errors alone. CPU uses PyTorch rows; CUDA uses the actual Triton implementation.
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

import torch

from verl.trainer.distillation.objectives import full_vocab_rows
from verl.trainer.distillation.targets import build_log_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logits", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--alphas", type=float, nargs="+", default=[1.0, 0.1, 0.03, 0.003, 0.0])
    parser.add_argument("--states", type=int, default=128)
    args = parser.parse_args()
    torch.set_num_threads(4)
    source = torch.load(args.logits, weights_only=True, map_location="cpu")
    student = source["student"][: args.states].to(args.device)
    teacher = source["teacher"][: args.states].to(args.device)
    if args.device == "cuda":
        from verl.trainer.distillation.full_vocab_triton import fused_mirror_rows

        current = lambda z, b, q, t, dtype: fused_mirror_rows(z, b, q, t, dtype, objective="mdpo", diagnostics=True)
    else:
        current = lambda z, b, q, t, dtype: full_vocab_rows(z, b, q, objective="mdpo", diagnostics=True, log_teacher=t)
    methods = {"current": current}
    if args.baseline_dir:
        sys.path.insert(0, str(args.baseline_dir.resolve()))
        if args.device == "cuda":
            fn = importlib.import_module("mirror_perf_baseline.full_vocab_triton").fused_mirror_rows
            methods["baseline"] = lambda z, b, q, t, dtype: fn(z, b, q, t, dtype, objective="mdpo", diagnostics=True)
        else:
            fn = importlib.import_module("mirror_perf_baseline.objectives").full_vocab_rows
            methods["baseline"] = lambda z, b, q, t, dtype: fn(
                z, b, q, objective="mdpo", diagnostics=True, log_teacher=t
            )
    results = []
    for alpha in args.alphas:
        z = teacher + alpha * (student - teacher)
        lq = z.log_softmax(-1)
        lt = teacher.log_softmax(-1)
        for kappa in [0.0, 0.5]:
            b = build_log_weights(lt, lq, kappa=kappa, eta=0.3)
            lp = z.double().log_softmax(-1)
            p = lp.exp()
            gap = lp - b.double()
            ref = p * (gap - (p * gap).sum(-1, keepdim=True))
            norm = ref.norm().item()
            measurements = {}
            for dtype in [torch.float32, torch.bfloat16]:
                for name, method in methods.items():
                    value, g, _ = method(z, b, lq, lt, dtype)
                    g = g.to(dtype).double()
                    error = g - ref
                    row_error = error.norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-30)
                    measurements[f"{name}/{dtype}"] = {
                        "relative_l2": error.norm().item() / max(norm, 1e-30),
                        "row_relative_p50": row_error.median().item(),
                        "row_relative_p90": row_error.quantile(0.9).item(),
                        "rows_relative_error_gt1": (row_error > 1).sum().item(),
                        "gradient_norm": g.norm().item(),
                        "sum_residual_l1": g.sum(-1).abs().sum().item(),
                        "loss_abs_error": abs(value.double().mean().item() - (p * gap).sum(-1).mean().item()),
                    }
            row = {
                "alpha": alpha,
                "kappa": kappa,
                "eta": 0.3,
                "reference_norm": norm,
                "teacher_kl": (p * (lp - teacher.double().log_softmax(-1))).sum(-1).mean().item(),
                "target_kl": (p * (lp - b.double().log_softmax(-1))).sum(-1).mean().item(),
                "measurements": measurements,
            }
            results.append(row)
            print(json.dumps(row), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "scope": __doc__,
                "torch": torch.__version__,
                "device": torch.cuda.get_device_name() if args.device == "cuda" else "CPU",
                "states": len(student),
                "vocab": student.shape[1],
                "source": str(args.logits),
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
