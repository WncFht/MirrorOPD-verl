#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Apply the SAME real-model FP32 update under three realization precisions.

FP32 vs rounded-weights/FP32 isolates weight rounding while keeping arithmetic
FP32. BF16 additionally changes forward arithmetic. Frozen real teacher and
student-generated prefix bank; full-parameter gradients; no synthetic target.
"""

import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from examples.mirroropd.research.studies.rkl_optimizers.study import CPUOptim, Study, norm, save


def cast(study, dtype):
    study.student.to(dtype=dtype)
    # Module.to(dtype) also casts rotary inv_freq and other floating buffers.
    # Keep them fixed so this intervention changes weights/arithmetic only.
    for name, value in study.reference_buffers.items():
        owner, _, leaf = name.rpartition(".")
        setattr(study.student.get_submodule(owner), leaf, value.clone())
    study.gpuparams = list(study.student.parameters())
    study.reset()


def main(args):
    common = SimpleNamespace(
        student=args.student,
        teacher=args.teacher,
        data=args.data,
        out=args.out,
        device="cuda:0",
        memory_gib=18.0,
        forward_dtype="float32",
        cap=64,
        prompt_cap=512,
        adam_lr=1e-6,
        arms=["adamw"],
        modes=["frozen"],
        outers=1,
        reuse=1,
        probe_every=1,
        seeds=[42],
    )
    study = Study(common)
    study.reference_buffers = {n: b.detach().clone() for n, b in study.student.named_buffers()}
    records = json.loads(args.prefixes.read_text())[:2]
    ht = study.teacher_cache(records)
    initial32 = study.measure(records, ht)["teacher_kl"]
    _, g32, _ = study.loss_and_grad(records, ht)
    cast(study, torch.bfloat16)
    initialbf = study.measure(records, ht)["teacher_kl"]
    _, gbf, _ = study.loss_and_grad(records, ht)
    dot = sum(float((a * b).sum()) for a, b in zip(g32, gbf, strict=False) if a is not None and b is not None)
    cos = dot / (norm(g32) * norm(gbf))
    diff = norm([a - b for a, b in zip(g32, gbf, strict=False) if a is not None and b is not None]) / norm(g32)
    del gbf
    gc.collect()
    results = []
    for multiplier in [0.25, 1.0, 4.0]:
        masters = [torch.nn.Parameter(p.clone()) for p in study.base]
        opt = CPUOptim(study.names, masters, "adamw", multiplier * 1e-6, study.device)
        opt.step(g32)
        changed32 = changedbf = 0
        delta2 = rounding_error2 = 0.0
        for p, b in zip(masters, study.base, strict=False):
            d = p.detach() - b
            rounded = p.detach().bfloat16().float() - b
            changed32 += int(torch.count_nonzero(d))
            changedbf += int(torch.count_nonzero(rounded))
            delta2 += float(d.square().sum())
            rounding_error2 += float((rounded - d).square().sum())
        row = {
            "lr": multiplier * 1e-6,
            "step_l2": delta2**0.5,
            "changed_fraction_fp32": changed32 / study.numel,
            "changed_fraction_bf16": changedbf / study.numel,
            "rounding_error_over_fp32_step": (rounding_error2 / delta2) ** 0.5,
            "initial_fp32_kl": initial32,
            "initial_bf16_kl": initialbf,
        }
        cast(study, torch.float32)
        study.reset(masters)
        row["after_fp32_kl"] = study.measure(records, ht)["teacher_kl"]
        with torch.no_grad():
            for gp, cp in zip(study.gpuparams, masters, strict=False):
                gp.copy_(cp.detach().bfloat16().float().to(study.device))
        row["after_rounded_weights_fp32_kl"] = study.measure(records, ht)["teacher_kl"]
        cast(study, torch.bfloat16)
        study.reset(masters)
        row["after_bf16_kl"] = study.measure(records, ht)["teacher_kl"]
        results.append(row)
        print(json.dumps(row), flush=True)
        save(
            args.out / "precision.json",
            {
                "fixed_prefixes": str(args.prefixes),
                "fp32_bf16_gradient_cosine": cos,
                "relative_gradient_error": diff,
                "records": results,
                "scope": "Same FP32 full-parameter AdamW step, no decay; separate writeback and arithmetic.",
            },
        )
        del opt, masters
        gc.collect()


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__)
    for name in ["student", "teacher", "data", "out", "prefixes"]:
        a.add_argument("--" + name, type=Path, required=True)
    main(a.parse_args())
