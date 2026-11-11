# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Benchmark real matrix projections with CPU FP64 as numerical reference."""

import argparse
import hashlib
import time
from pathlib import Path

import torch

from examples.mirroropd.research.backends.direction_projection import DirectionProjector, reduce_projection
from examples.mirroropd.research.core.io import dump, read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--matrices", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    records = read(args.attempt / "matrix-index.json")
    # Uniform coverage of matrix shapes/layer depths; no choice based on timing.
    indices = torch.linspace(0, len(records) - 1, min(args.matrices, len(records))).long().tolist()
    projector = DirectionProjector(args.attempt / "directions", backend="cuda-stream")
    rows = []
    for selected in indices:
        record = records[selected]
        path = args.attempt / "directions" / f"{record['index']:04d}.pt"
        data = projector.directions(record["index"])
        generator = torch.Generator().manual_seed(20261110 + record["index"])
        # Dense synthetic gradient; real checkpoint directions. End-to-end model gate is separate.
        gradient = torch.randn(record["shape"], generator=generator)
        gpu_gradient = gradient.cuda()
        reference = reduce_projection(gradient, data, device="cpu")
        projector.project(record["index"], gpu_gradient)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.repeats):
            actual = projector.project(record["index"], gpu_gradient)
        torch.cuda.synchronize()
        gpu_seconds = (time.perf_counter() - start) / args.repeats
        start = time.perf_counter()
        for _ in range(args.repeats):
            # Reproduce the previous worker, including repeated mmap/FP64 reductions.
            payload = torch.load(path, weights_only=True, mmap=True)
            reduce_projection(gradient, {name: payload[name] for name in data}, device="cpu")
        cpu_seconds = (time.perf_counter() - start) / args.repeats
        errors = [abs(actual[0] - reference[0]) / max(reference[0], 1e-30)]
        for band in data:
            a, b = actual[1][band], reference[1][band]
            # Normalize dot error by the Cauchy bound, robust to cancellation.
            errors.append(abs(a["dot"] - b["dot"]) / max(reference[0] * b["norm"], 1e-30))
            errors.append(abs(a["norm"] - b["norm"]) / max(b["norm"], 1e-30))
        rows.append(
            dict(
                key=record["key"],
                shape=record["shape"],
                cpu_seconds=cpu_seconds,
                cuda_seconds=gpu_seconds,
                speedup=cpu_seconds / gpu_seconds,
                max_scaled_error=max(errors),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
        dump(args.output.with_suffix(".progress.json"), rows)
    passed = all(row["max_scaled_error"] <= 1e-10 for row in rows)
    dump(
        args.output,
        dict(
            passed=passed,
            rows=rows,
            repeats=args.repeats,
            scope="projection microbenchmark; synthetic dense gradients, real checkpoint directions",
            gpu=torch.cuda.get_device_name(),
            torch_version=torch.__version__,
            source_sha256={
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [
                    Path(__file__),
                    Path(
                        __import__(
                            "examples.mirroropd.research.backends.direction_projection", fromlist=["__file__"]
                        ).__file__
                    ),
                ]
            },
            peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        ),
    )
    if not passed:
        raise ValueError("FP64 projection identity gate failed")


if __name__ == "__main__":
    main()
