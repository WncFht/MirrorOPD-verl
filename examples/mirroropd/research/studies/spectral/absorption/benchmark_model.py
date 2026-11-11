# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Verify projection backends on a real full-model RKL backward pass."""

import argparse
import time
from pathlib import Path

import torch

from examples.mirroropd.research.backends.direction_projection import DirectionProjector
from examples.mirroropd.research.core.io import dump, read

from ...teacher_difference.runtime import load_model, prefix_logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--job", type=int, required=True)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    job = next(j for j in read(args.run_root / "manifest.json")["jobs"] if j["job_id"] == args.job)
    banks = read(args.attempt / "banks.json")
    records = read(args.attempt / "matrix-index.json")
    reference, head = torch.load(args.attempt / "reference.pt", weights_only=True, mmap=True)
    model = load_model(job["endpoint"], trainable=True, checkpointing=True, device="cuda")
    model.requires_grad_(False)
    params = dict(model.named_parameters())
    for record in records:
        params[record["key"]].requires_grad_(True)
    origin = "initial-student"
    row = banks[origin][0]
    with torch.no_grad():
        target = torch.nn.functional.linear(reference[origin, row["row_id"]][0].cuda(), head.cuda()).log_softmax(-1)
    torch.cuda.synchronize()
    started = time.perf_counter()
    logits = prefix_logits(model, row, 0).log_softmax(-1)
    loss = (logits.exp() * (logits - target)).sum()
    loss.backward()
    torch.cuda.synchronize()
    backward_seconds = time.perf_counter() - started
    results, times = {}, {}
    for backend in ["cpu-reference", "cuda-stream"]:
        projector = DirectionProjector(args.attempt / "directions", backend=backend)
        started = time.perf_counter()
        values = []
        for record in records:
            gradient = params[record["key"]].grad
            norm, projected = projector.project(record["index"], gradient)
            # Retain the production path's full gradient D2H for domain sums.
            copied = gradient.detach().float().cpu()
            values.append(dict(key=record["key"], norm=norm, directions=projected))
            del copied
        torch.cuda.synchronize()
        times[backend] = time.perf_counter() - started
        results[backend] = values
        if backend == "cuda-stream":
            started = time.perf_counter()
            for record in records:
                gradient = params[record["key"]].grad
                projector.project(record["index"], gradient)
                copied = gradient.detach().float().cpu()
                del copied
            torch.cuda.synchronize()
            times["cuda-stream-warm"] = time.perf_counter() - started
        dump(args.output.with_suffix(".progress.json"), dict(times=times))
    errors = []
    for a, b in zip(results["cpu-reference"], results["cuda-stream"], strict=True):
        errors.append(abs(a["norm"] - b["norm"]) / max(a["norm"], 1e-30))
        for band, x in a["directions"].items():
            y = b["directions"][band]
            errors.append(abs(x["dot"] - y["dot"]) / max(a["norm"] * x["norm"], 1e-30))
            errors.append(abs(x["norm"] - y["norm"]) / max(x["norm"], 1e-30))
    passed = max(errors) <= 1e-10
    dump(
        args.output,
        dict(
            passed=passed,
            reverse_kl=float(loss.detach()),
            max_scaled_error=max(errors),
            matrices=len(records),
            backward_seconds=backward_seconds,
            times=times,
            results=results,
            peak_cuda_bytes=torch.cuda.max_memory_allocated(),
            row_id=row["row_id"],
            position=0,
            scope="same real model backward; full-layer projection and gradient D2H; cold/warm caches",
        ),
    )
    if not passed:
        raise ValueError("Actual model gradient projection identity failed")


if __name__ == "__main__":
    main()
