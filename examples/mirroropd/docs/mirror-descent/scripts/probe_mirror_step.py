#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Controlled real-model one-step probe with a nearby synthetic teacher.

Uses initial model weights, four saved math positions and a captured teacher-
student logit direction. Calibrates a nearby frozen teacher, then compares cold
AdamW steps. This does not replay E45/E47 late model/optimizer state or FSDP2.
Parameter casts emulate FP32 masters with BF16-valued forward weights; CPU or
single-device SDPA kernels are not asserted bitwise equal to packed training.
"""

import argparse
import gc
import importlib
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.func import functional_call
from transformers import AutoModelForCausalLM


class SelectedHead(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, ids, positions):
        hidden = self.m.model(input_ids=ids, use_cache=False).last_hidden_state[0].index_select(0, positions)
        # BF16-valued inputs with FP32 output/accumulation; isolate row-gradient
        # arithmetic here rather than substitute a new full engine backward.
        return hidden.float() @ self.m.lm_head.weight.float().T


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--logits", type=Path, required=True)
    parser.add_argument("--prefixes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--precisions", nargs="+", choices=["fp32", "rounded_weights", "bf16"], default=["fp32", "bf16"]
    )
    parser.add_argument("--gradients", nargs="+", choices=["fp64", "current", "baseline"], default=["fp64", "current"])
    parser.add_argument("--lrs", type=float, nargs="+", default=[1e-6, 3e-7, 1e-7, 3e-8, 1e-8, 0.0])
    parser.add_argument("--states", type=int, default=4)
    parser.add_argument("--teacher-kl", type=float, default=8e-4)
    parser.add_argument("--dither-masters", action="store_true")
    parser.add_argument("--same-step-precision", action="store_true")
    args = parser.parse_args()
    if "baseline" in args.gradients and not args.baseline_dir:
        parser.error("baseline requires --baseline-dir")
    if args.states < 1 or args.states > 32:
        parser.error("--states must be in 1..32")
    os.environ["HF_HUB_OFFLINE"] = "1"
    torch.set_num_threads(8 if args.device == "cpu" else 4)
    torch.manual_seed(43)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rec = json.loads(args.prefixes.read_text())["records"][0]
    positions = torch.tensor(rec["positions"][: args.states], device=args.device)
    ids = torch.tensor([rec["ids"][: int(positions[-1]) + 1]], device=args.device)
    saved = torch.load(args.logits, map_location="cpu", weights_only=True)
    signal = (saved["teacher"][: args.states] - saved["student"][: args.states]).double().to(args.device)
    del saved
    started = time.monotonic()
    model = SelectedHead(
        AutoModelForCausalLM.from_pretrained(
            str(args.model), dtype=torch.float32, attn_implementation="sdpa", local_files_only=True
        ).eval()
    )
    if args.dither_masters:
        # Synthetic sub-ULP states: preserve BF16 weights while placing FP32
        # masters inside rounding cells. Do not confuse this with trained states.
        with torch.no_grad():
            for parameter in model.parameters():
                rounded = parameter.to(torch.bfloat16)
                up = torch.nextafter(rounded, torch.full_like(rounded, float("inf"))).float() - parameter
                down = parameter - torch.nextafter(rounded, torch.full_like(rounded, -float("inf"))).float()
                noise = torch.rand_like(parameter) * 2 - 1
                parameter.add_(torch.where(noise > 0, noise * up, noise * down) * 0.49)
                assert torch.equal(parameter.to(torch.bfloat16), rounded)
    model = model.to(args.device)
    params = dict(model.named_parameters())
    base = {k: p.detach().clone() for k, p in params.items()}
    results = []
    if args.baseline_dir:
        sys.path.insert(0, str(args.baseline_dir.resolve()))

    def forward(mode):
        cast = (
            {k: p.to(torch.bfloat16) for k, p in params.items()}
            if mode == "bf16"
            else {k: p.to(torch.bfloat16).float() for k, p in params.items()}
            if mode == "rounded_weights"
            else params
        )
        return functional_call(model, cast, (ids, positions))

    for precision in args.precisions:
        with torch.no_grad():
            zq = forward(precision).detach()
            lq = zq.double().log_softmax(-1)
            anchors = (
                {mode: forward(mode).double().log_softmax(-1) for mode in ["fp32", "rounded_weights", "bf16"]}
                if args.same_step_precision
                else {}
            )
        lo, hi = 0.0, 1.0
        if (lq.exp() * (lq - (zq.double() + signal).log_softmax(-1))).sum(-1).mean() < args.teacher_kl:
            raise ValueError("Captured direction cannot reach requested teacher KL within alpha<=1")
        for _ in range(30):
            alpha = (lo + hi) / 2
            lt = (zq.double() + alpha * signal).log_softmax(-1)
            if (lq.exp() * (lq - lt)).sum(-1).mean().item() > args.teacher_kl:
                hi = alpha
            else:
                lo = alpha
        lt = (zq.double() + alpha * signal).log_softmax(-1)
        b = (0.7 * lq + 0.3 * lt).float().double()
        lm = b.log_softmax(-1)

        def measure(z, lm=lm, lt=lt, lq=lq):
            lp = z.double().log_softmax(-1)
            p = lp.exp()
            return {
                "target_kl": (p * (lp - lm)).sum(-1).mean().item(),
                "teacher_kl": (p * (lp - lt)).sum(-1).mean().item(),
                "anchor_kl": (p * (lp - lq)).sum(-1).mean().item(),
            }

        initial = measure(zq)
        for method in args.gradients:
            for p in params.values():
                p.grad = None
            z = forward(precision)
            initial_train = measure(z.detach())
            if method == "fp64":
                lp = z.double().log_softmax(-1)
                ((lp.exp() * (lp - b)).sum(-1).mean() / 0.3).backward()
            else:
                namespace = "mirror_perf_baseline" if method == "baseline" else "verl.trainer.distillation"
                if args.device == "cuda":
                    kernel = importlib.import_module(namespace + ".full_vocab_triton").fused_mirror_rows
                    _, dz, _ = kernel(
                        z.detach(),
                        b.float(),
                        lq.float(),
                        lt.float(),
                        torch.bfloat16 if precision == "bf16" else torch.float32,
                        objective="mdpo",
                        diagnostics=False,
                    )
                else:
                    kernel = importlib.import_module(namespace + ".objectives").full_vocab_rows
                    _, dz, _ = kernel(z.detach(), b.float(), objective="mdpo")
                z.backward(dz.float() / len(z) / 0.3)
            norm = torch.nn.utils.clip_grad_norm_(params.values(), 1.0).item()
            gradients = {k: p.grad.detach().clone() if p.grad is not None else None for k, p in params.items()}
            for lr in args.lrs:
                for p in params.values():
                    p.grad = None
                with torch.no_grad():
                    for k, p in params.items():
                        p.copy_(base[k])
                optimizer = torch.optim.AdamW(
                    params.values(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, foreach=False
                )
                for k, p in params.items():
                    p.grad = gradients[k]
                optimizer.step()
                with torch.no_grad():
                    after = measure(forward(precision))
                    displacement = {}
                    for mode, anchor in anchors.items():
                        lp = forward(mode).double().log_softmax(-1)
                        displacement[mode] = (lp.exp() * (lp - anchor)).sum(-1).mean().item()
                dot = sum(
                    (gradients[k].double() * (p.detach() - base[k]).double()).sum().item()
                    for k, p in params.items()
                    if gradients[k] is not None
                )
                row = {
                    "precision": precision,
                    "gradient": method,
                    "lr": lr,
                    "initial": initial,
                    "initial_train": initial_train,
                    "after": after,
                    "grad_norm": norm,
                    "grad_dot_delta": dot,
                    "alpha": alpha,
                    "same_step_anchor_kl": displacement,
                    "elapsed_seconds": time.monotonic() - started,
                }
                results.append(row)
                print(json.dumps(row), flush=True)
                del optimizer
                with torch.no_grad():
                    for k, p in params.items():
                        p.copy_(base[k])
                for p in params.values():
                    p.grad = None
                gc.collect()
            del gradients, z
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "scope": __doc__,
                "torch": torch.__version__,
                "device": args.device,
                "model": str(args.model),
                "selected_states": len(positions),
                "input_tokens": ids.shape[1],
                "dither_masters": args.dither_masters,
                "kappa": 0.0,
                "eta": 0.3,
                "records": results,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
