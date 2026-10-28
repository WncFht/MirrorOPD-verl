#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Teacher-only BF16/FP32 discrepancy on identical frozen held-out states."""

import argparse
from pathlib import Path

import torch
from accelerate import cpu_offload
from transformers import AutoModelForCausalLM

from examples.mirroropd.research.precision.polaris.run_study import dump, read
from verl.trainer.distillation.numerics import kl_rows

p = argparse.ArgumentParser()
for key in ["models", "data", "cache"]:
    p.add_argument("--" + key, type=Path, required=True)
p.add_argument("--student-weighted", action="store_true")
a = p.parse_args()
torch.set_num_threads(8)
torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.cuda.set_per_process_memory_fraction(8 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
model = (
    AutoModelForCausalLM.from_pretrained(
        a.models / "Polaris-7B-Preview", dtype=torch.float32, attn_implementation="sdpa", local_files_only=True
    )
    .eval()
    .requires_grad_(False)
)
cpu_offload(model, execution_device=torch.device("cuda:0"), offload_buffers=False)
output = []
with torch.inference_mode():
    for r in read(a.data / "records.json"):
        if r["role"] != "test":
            continue
        if a.student_weighted and not (
            r["source"] == "student" and (128 <= r["id"] < 144 or 192 <= r["id"] < 196 or 208 <= r["id"] < 212)
        ):
            continue
        tokens = r["prompt_ids"] + r["response_ids"][:-1]
        indices = torch.tensor([len(r["prompt_ids"]) - 1 + p for p in r["positions"]], device="cuda")
        z = model(input_ids=torch.tensor([tokens], device="cuda"), use_cache=False, logits_to_keep=indices).logits[0]
        b = torch.load(a.cache / "teacher" / f"{r['uid']}.pt", weights_only=True).cuda()
        rkl = kl_rows(b, z)[0]
        fkl = kl_rows(z, b)[0]
        pb = b.softmax(-1)
        p32 = z.softmax(-1)
        output.append(
            dict(
                uid=r["uid"],
                source=r["source"],
                domain=r["domain"],
                positions=len(r["positions"]),
                bf16_to_fp32_kl=float(rkl.mean()),
                fp32_to_bf16_kl=float(fkl.mean()),
                tv=float(0.5 * (pb - p32).abs().sum(-1).mean()),
                top1_agreement=float((b.argmax(-1) == z.argmax(-1)).float().mean()),
            )
        )
        if a.student_weighted:
            student = torch.load(a.cache / "baseline/initial" / f"{r['uid']}.pt", weights_only=True).cuda()
            output[-1].update(
                student_to_teacher_bf16=float(kl_rows(student, b)[0].mean()),
                student_to_teacher_fp32=float(kl_rows(student, z)[0].mean()),
            )
            del student
        print("teacher precision", r["uid"], output[-1]["bf16_to_fp32_kl"], flush=True)
dump(a.data / ("teacher-precision-student-weighted.json" if a.student_weighted else "teacher-precision.json"), output)
