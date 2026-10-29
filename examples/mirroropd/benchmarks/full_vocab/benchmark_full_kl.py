# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Benchmark the production head-loss backends; synthetic inputs, no backbone."""

import argparse
import gc
import json
import math
from functools import partial
from pathlib import Path

import torch

from examples.mirroropd.benchmarks.full_vocab.probe_full_kl import measure, relative_error
from verl.trainer.distillation.full_vocab import full_reverse_kl


def mean_kl(hs, ws, ht, wt, backend, chunk):
    return full_reverse_kl(hs, ws, ht, wt, backend=backend, token_chunk=chunk, vocab_chunk=8192) / len(hs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", nargs="+", type=int, default=[512, 4096])
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--teacher-hidden", type=int, default=1536)
    parser.add_argument("--teacher-vocab", type=int, default=151936)
    parser.add_argument("--profile", action="store_true", help="Export a separate untimed CPU/CUDA head-loss trace")
    choices = ["dense", "recompute", "cached", "streaming", "precompute", "precompute_fused", "liger"]
    parser.add_argument("--backends", nargs="+", choices=choices, default=choices)
    parser.add_argument("--reference-backend", choices=choices, default="dense")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite benchmark evidence: {args.output}")
    if args.teacher_hidden < 1 or args.teacher_vocab < 151936 or args.chunk < 1 or min(args.tokens) < 1:
        parser.error("Positive shapes and a teacher vocabulary covering the student are required")
    if args.teacher_vocab != 151936 and "liger" in [*args.backends, args.reference_backend]:
        parser.error("The Liger comparison requires equal vocabulary widths")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = []
    for n in args.tokens:
        hs = torch.randn(n, 1536, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        ht = torch.randn(n, args.teacher_hidden, device="cuda", dtype=torch.bfloat16)
        ws = (torch.randn(151936, 1536, device="cuda", dtype=torch.bfloat16) / 1536**0.5).requires_grad_()
        wt = torch.randn(args.teacher_vocab, args.teacher_hidden, device="cuda", dtype=torch.bfloat16)
        wt /= args.teacher_hidden**0.5
        exact = mean_kl(hs, ws, ht, wt, args.reference_backend, args.chunk)
        gradients = torch.autograd.grad(exact, (hs, ws))
        expected, expected_grad = exact.item(), [g.cpu() for g in gradients]
        del exact, gradients
        for backend in args.backends:
            fn = partial(mean_kl, hs, ws, ht, wt, backend, args.chunk)
            actual = fn()
            gradients = torch.autograd.grad(actual, (hs, ws))
            errors = {
                "value_abs": abs(actual.item() - expected),
                "hidden_relative_l2": relative_error(gradients[0].cpu(), expected_grad[0]),
                "head_relative_l2": relative_error(gradients[1].cpu(), expected_grad[1]),
            }
            assert all(math.isfinite(value) for value in errors.values()), errors
            assert errors["value_abs"] < 1e-4 and max(errors.values()) < 0.02, errors
            del actual, gradients
            row = {"tokens": n, "backend": backend, "errors": errors, **measure(fn, hs, ws, True, warmup=2, repeats=5)}
            if args.profile:
                trace = args.output.with_name(f"{args.output.stem}-{backend}-n{n}.trace.json.gz")
                if trace.exists():
                    raise FileExistsError(trace)
                trace.parent.mkdir(parents=True, exist_ok=True)
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=True,
                ) as profile:
                    loss = fn()
                    grads = torch.autograd.grad(loss, (hs, ws))
                    torch.cuda.synchronize()
                del loss, grads
                profile.export_chrome_trace(str(trace))
                row["trace"] = str(trace)
            rows.append(row)
            print(json.dumps(row), flush=True)
        del hs, ht, ws, wt, expected_grad, fn
        gc.collect()
        torch.cuda.empty_cache()
    result = {
        "scope": "production operators only; BF16 input, FP32 logits/accumulation, no backbone/FSDP/TQ",
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "chunk": args.chunk,
        "vocab": 151936,
        "hidden": 1536,
        "teacher_hidden": args.teacher_hidden,
        "teacher_vocab": args.teacher_vocab,
        "reference_backend": args.reference_backend,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
