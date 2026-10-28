#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Native BF16 head-stage benchmark, with interleaved methods on shared GPUs.

Includes projections, row diagnostics, dH/dW and backward application. Excludes
backbones, rollout, IO and distributed collectives; not an end-to-end speed claim.
"""

import argparse
import json
import statistics as st
import time
from pathlib import Path

import torch

from verl.trainer.distillation.forward_vocab import full_forward_kl_result
from verl.trainer.distillation.full_vocab import full_reverse_kl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tokens", type=int, default=512)
    p.add_argument("--repeats", type=int, default=7)
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.cuda.set_per_process_memory_fraction(0.25)
    h = (torch.randn(a.tokens, 1536, device="cuda") * 0.1).bfloat16().requires_grad_()
    w = (torch.randn(151936, 1536, device="cuda") / 1536**0.5).bfloat16().requires_grad_()
    t = (torch.randn(a.tokens, 3584, device="cuda") * 0.1).bfloat16()
    v = (torch.randn(152064, 3584, device="cuda") / 3584**0.5).bfloat16()
    methods = [
        ("forward_torch_512", "precompute", 512),
        ("forward_fused_512", "precompute_fused", 512),
        ("forward_fused_128", "precompute_fused", 128),
        ("reverse_fused_512", "precompute_fused", 512),
    ]

    def run(name, backend, chunk):
        h.grad = w.grad = None
        if name.startswith("reverse"):
            loss = full_reverse_kl(h, w, t, v, backend=backend, token_chunk=chunk)
        else:
            loss = full_forward_kl_result(
                h, w, t, v, backend=backend, token_chunk=chunk, teacher_support="student"
            ).loss_sum
        (loss / a.tokens).backward()
        return float(loss.detach())

    for row in methods:
        for _ in range(2):
            run(*row)
    samples = {name: [] for name, _, _ in methods}
    memory = {name: [] for name, _, _ in methods}
    values = {}
    for i in range(a.repeats):
        for row in methods if i % 2 == 0 else methods[::-1]:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            values[row[0]] = run(*row)
            torch.cuda.synchronize()
            samples[row[0]].append((time.perf_counter() - start) * 1000)
            memory[row[0]].append(torch.cuda.max_memory_allocated() / 2**30)
    out = dict(
        device=torch.cuda.get_device_name(),
        torch=torch.__version__,
        tokens=a.tokens,
        student_shape=[151936, 1536],
        teacher_shape=[152064, 3584],
        dtype="bfloat16 projections / FP32 accumulation",
        shared_gpu=True,
        scope=__doc__,
        results={
            name: dict(
                median_ms=st.median(xs),
                min_ms=min(xs),
                max_ms=max(xs),
                samples_ms=xs,
                peak_allocated_gib=max(memory[name]),
                loss_sum=values[name],
            )
            for name, xs in samples.items()
        },
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
