# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Paired BF16 full-vocab VJP benchmark on fixed inputs; no backbone/IO timing.

Run from the repository with PYTHONPATH=. and one visible CUDA device. A baseline
is an isolated copy of the pre-edit numerical modules (including their imports),
not git HEAD when the working tree is dirty. See the implementation document for
baseline layout and invocation commands. Never overwrite a baseline during comparison.
"""

import argparse
import importlib
import json
import statistics
import sys
import time
from pathlib import Path

import torch

from verl.trainer.distillation.full_vocab import full_vocab_objective_result
from verl.workers.config.distillation_spec import resolve_distillation_spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--objective", choices=["mdpo", "bpo"], required=True)
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--width", type=int, default=1536)
    parser.add_argument("--vocab", type=int, default=151936)
    parser.add_argument("--chunks", type=int, nargs="+", default=[32, 128, 512])
    parser.add_argument("--rounds", type=int, default=9)
    args = parser.parse_args()
    if args.rounds < 3 or min(args.tokens, args.width, args.vocab, *args.chunks) <= 0:
        parser.error("positive dimensions and at least three rounds required")
    sys.path.insert(0, str(args.baseline_dir.resolve()))
    baseline = importlib.import_module("mirror_perf_baseline.full_vocab").full_vocab_objective_result
    torch.set_num_threads(1)
    torch.manual_seed(71)
    torch.cuda.manual_seed_all(71)
    spec = resolve_distillation_spec(
        {
            "loss_mode": "mirror",
            "mirror": {
                "objective": args.objective,
                "diagnostics": args.diagnostics,
                "eta": 0.3,
                "kappa": 0.5,
            },
        }
    )

    def random(shape):
        return torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.1

    hs, ws = [random(shape).requires_grad_() for shape in [(args.tokens, args.width), (args.vocab, args.width)]]
    ht, wt, hq, wq = [random(shape) for shape in [hs.shape, ws.shape, hs.shape, ws.shape]]
    cases = []
    for chunk in args.chunks:

        def run(fn, chunk=chunk):
            result = fn(hs, ws, ht, wt, spec=spec, hq=hq, wq=wq, backend="precompute_fused", token_chunk=chunk)
            gradients = torch.autograd.grad(result.loss_sum / spec.eta, (hs, ws))
            return result, gradients

        functions = {"before": baseline, "after": full_vocab_objective_result}
        # Check whole-projection VJPs before timed runs; the small independent
        # FP64 oracle lives in test_mirror_on_gpu.py.
        old, old_grad = run(baseline)
        new, new_grad = run(full_vocab_objective_result)
        torch.testing.assert_close(new.loss_sum, old.loss_sum, atol=0.002, rtol=2e-4)
        relative = []
        for got, wanted in zip(new_grad, old_grad, strict=True):
            error = ((got.float() - wanted.float()).norm() / wanted.float().norm().clamp_min(1e-12)).item()
            assert error < 0.004, error
            relative.append(error)
        for key, value in new.token_diagnostics.items():
            torch.testing.assert_close(value, old.token_diagnostics[key], atol=5e-6, rtol=3e-4)
        loss_diff = (new.loss_sum - old.loss_sum).abs().item()
        del old, new, old_grad, new_grad, got, wanted
        for _ in range(2):
            for fn in functions.values():
                run(fn)
        torch.cuda.synchronize()
        timings = {name: {"cuda_ms": [], "wall_ms": []} for name in functions}
        for iteration in range(args.rounds):
            for name in list(functions) if iteration % 2 == 0 else list(reversed(functions)):
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start = time.perf_counter()
                begin.record()
                result = run(functions[name])
                end.record()
                end.synchronize()
                timings[name]["wall_ms"].append((time.perf_counter() - start) * 1000)
                timings[name]["cuda_ms"].append(begin.elapsed_time(end))
                del result
        for name, fn in functions.items():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            allocated = torch.cuda.memory_allocated()
            result = run(fn)
            torch.cuda.synchronize()
            timings[name]["peak_extra_mib"] = (torch.cuda.max_memory_allocated() - allocated) / 2**20
            timings[name]["median_cuda_ms"] = statistics.median(timings[name]["cuda_ms"])
            timings[name]["median_wall_ms"] = statistics.median(timings[name]["wall_ms"])
            del result
        case = {"chunk": chunk, "loss_abs_diff": loss_diff, "vjp_relative_l2": relative, **timings}
        case["speedup"] = timings["before"]["median_wall_ms"] / timings["after"]["median_wall_ms"]
        cases.append(case)
        print(json.dumps(case), flush=True)
    output = {
        "scope": "three head projections + target + full objective/VJP; excludes backbone, IO, optimizer",
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "seed": 71,
        "dtype": "bfloat16",
        "eta": 0.3,
        "kappa": 0.5,
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
