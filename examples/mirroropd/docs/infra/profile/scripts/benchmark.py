# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Paired synthetic-head timings, gradient checks and short CUDA traces."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from fused_candidate import FusedPrecomputedKL, fused_rows

from verl.trainer.distillation.full_vocab import full_reverse_kl


def call(hs, ws, ht, wt, backend, chunk):
    if backend == "fused_candidate":
        return FusedPrecomputedKL.apply(hs, ws, ht, wt, chunk) / len(hs)
    return full_reverse_kl(hs, ws, ht, wt, backend=backend, token_chunk=chunk) / len(hs)


def check_reduction():
    records = []
    for v in (31, 4096, 8193, 151936):
        for scale in (1.0, 20.0):
            a, b = [torch.randn(3, v, device="cuda") * scale for _ in range(2)]
            lp, lt = a.log_softmax(-1), b.log_softmax(-1)
            p = lp.exp()
            k = (p * (lp - lt)).sum(-1)
            dz = p * (lp - lt - k[:, None])
            actual, grad = fused_rows(a, b, torch.float32)
            torch.testing.assert_close(actual, k, atol=2e-4, rtol=2e-5)
            torch.testing.assert_close(grad, dz, atol=2e-4, rtol=2e-4)
            same, same_grad = fused_rows(a, a, torch.float32)
            torch.testing.assert_close(same, torch.zeros_like(same), atol=2e-5, rtol=0)
            torch.testing.assert_close(same_grad, torch.zeros_like(same_grad), atol=2e-5, rtol=0)
            records.append(
                {
                    "vocab": v,
                    "scale": scale,
                    "value_max_abs": (actual - k).abs().max().item(),
                    "grad_max_abs": (grad - dz).abs().max().item(),
                }
            )
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tokens", type=int, nargs="+", default=[4096, 32768])
    p.add_argument("--chunks", type=int, nargs="+", default=[128, 256, 512, 1024])
    p.add_argument("--backends", nargs="+", default=["precompute", "fused_candidate", "streaming", "liger"])
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--trace", action="store_true")
    p.add_argument("--capture-nsys", action="store_true")
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    result = {
        "scope": "synthetic BF16 head/loss forward+backward; no backbone/FSDP/TQ",
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "rows": [],
    }
    result["reduction_checks"] = check_reduction() if "fused_candidate" in args.backends else []
    for n in args.tokens:
        hs = torch.randn(n, 1536, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        ht = torch.randn_like(hs)
        ws = (torch.randn(151936, 1536, device="cuda", dtype=torch.bfloat16) / 1536**0.5).requires_grad_()
        wt = torch.randn_like(ws) / 1536**0.5
        expected = call(hs, ws, ht, wt, "precompute", 512)
        eg = [x.cpu() for x in torch.autograd.grad(expected, (hs, ws))]
        expected = expected.item()
        # Alternate order across chunks to expose systematic warm/cache drift.
        for ci, chunk in enumerate(args.chunks):
            backends = args.backends if ci % 2 == 0 else list(reversed(args.backends))
            for backend in backends:
                if backend == "liger" and chunk != 512:
                    continue  # Liger's own chunk policy ignores this wrapper parameter.

                def once(hs=hs, ws=ws, ht=ht, wt=wt, backend=backend, chunk=chunk):
                    hs.grad = ws.grad = None
                    loss = call(hs, ws, ht, wt, backend, chunk)
                    loss.backward()
                    return loss

                try:
                    actual = once()
                    errors = {"loss_abs": abs(actual.item() - expected)}
                    for name, x, y in zip(
                        ("hidden_relative_l2", "head_relative_l2"), (hs.grad, ws.grad), eg, strict=True
                    ):
                        errors[name] = (
                            (x.cpu().double() - y.double()).norm() / y.double().norm().clamp_min(1e-20)
                        ).item()
                    assert errors["loss_abs"] < 2e-4 and max(errors.values()) < 0.02, errors
                    del actual
                    once()
                    torch.cuda.synchronize()
                    hs.grad = ws.grad = None
                    torch.cuda.empty_cache()
                    resident = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    times = []
                    for _ in range(args.repeats):
                        torch.cuda.synchronize()
                        begin = time.perf_counter()
                        once()
                        torch.cuda.synchronize()
                        times.append((time.perf_counter() - begin) * 1000)
                    row = {
                        "tokens": n,
                        "chunk": chunk,
                        "backend": backend,
                        "errors": errors,
                        "wall_ms_samples": times,
                        "wall_ms_median": statistics.median(times),
                        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                        "resident_gib": resident / 2**30,
                    }
                    if args.capture_nsys:
                        torch.cuda.profiler.start()
                        once()
                        torch.cuda.synchronize()
                        torch.cuda.profiler.stop()
                    if args.trace and chunk == 512 and backend in ("precompute", "fused_candidate"):
                        trace = args.output.parent / f"head-{backend}-n{n}.json.gz"
                        with torch.profiler.profile(
                            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
                        ) as prof:
                            once()
                            torch.cuda.synchronize()
                        prof.export_chrome_trace(str(trace))
                        row["trace"] = str(trace)
                except torch.OutOfMemoryError:
                    hs.grad = ws.grad = None
                    torch.cuda.empty_cache()
                    row = {"tokens": n, "chunk": chunk, "backend": backend, "status": "oom"}
                result["rows"].append(row)
                args.output.write_text(json.dumps(result, indent=2) + "\n")
                print(json.dumps(row), flush=True)
        del hs, ht, ws, wt, eg, once
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
