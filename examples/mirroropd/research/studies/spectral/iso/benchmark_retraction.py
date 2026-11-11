# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Synchronized real-checkpoint polar benchmark, including numerical gates and fallbacks."""

import argparse
import hashlib
import statistics
import time
from pathlib import Path

import torch

from examples.mirroropd.research.core.io import dump, read

from ..scan import thin_svd
from ..weights import DEFAULT_SCOPE, TensorCheckpoint
from .retraction import retract


def benchmark(manifest, output, stop, job_ids=(0, 6, 7, 11, 19), repeats=2, smoke=False):
    output, stop = Path(output), Path(stop)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(211)
    rows = []
    jobs = read(manifest)["jobs"]
    seen = set()
    dump(
        output / "protocol.json",
        dict(
            manifest_sha256=hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
            device=torch.cuda.get_device_name(),
            repeats=repeats,
            arithmetic="FP64 Gram/Newton and gesvd; FP32 output rounding separately gated",
            input="Real layer-0 q/k/up checkpoint singular frames plus synthetic isotropic 1e-5 perturbation",
            unit="matrix/frame; elapsed time is synchronized GPU wall time, not end-to-end training",
        ),
    )

    def check():
        if stop.exists():
            raise InterruptedError("Benchmark STOP")

    def timed(function):
        timings = []
        result = None
        for _ in range(repeats):
            check()
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = function()
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - started)
        return result, timings

    for job_id in job_ids:
        job = next(j for j in jobs if j["job_id"] == job_id)
        checkpoint = TensorCheckpoint(job["base"])
        keys = [
            k
            for k in checkpoint.selected(DEFAULT_SCOPE)
            if ".layers.0." in k and any(f".{module}." in k for module in ("q_proj", "k_proj", "up_proj"))
        ]
        if smoke:
            keys = [k for k in keys if ".k_proj." in k]
        for key in keys:
            check()
            value = checkpoint.get(key).to(device="cuda", dtype=torch.float64)
            u, s, vh = thin_svd(value, driver="gesvd", transpose_wide=True)
            for label, frame in [("left", u), ("right", vh.T.contiguous())]:
                shape = tuple(frame.shape)
                if shape in seen:
                    continue
                seen.add(shape)
                perturbed = frame + torch.randn_like(frame) * 1e-5
                reference, old_times = timed(lambda x=perturbed: retract(x, method="svd"))
                (actual, info), new_times = timed(lambda x=perturbed: retract(x, return_info=True))
                relative = float((reference - actual).norm() / reference.norm())
                identity = torch.eye(frame.shape[1], device=frame.device, dtype=frame.dtype)
                orth = float((actual.T @ actual - identity).abs().max())
                expected_weight = (reference * s) @ vh if label == "left" else (u * s) @ reference.T
                actual_weight = (actual * s) @ vh if label == "left" else (u * s) @ actual.T
                reconstruction = float((expected_weight - actual_weight).norm() / expected_weight.norm())
                check_gate = max(relative, orth, reconstruction) <= 1e-9 and info["used"] == "newton"
                row = dict(
                    job_id=job_id,
                    tensor=key,
                    frame=label,
                    shape=list(shape),
                    reference_seconds=old_times,
                    newton_seconds=new_times,
                    speedup=statistics.median(old_times) / statistics.median(new_times),
                    polar_relative_error=relative,
                    orthogonality_max=orth,
                    reconstructed_weight_relative_error=reconstruction,
                    telemetry=info,
                    passed=check_gate,
                )
                rows.append(row)
                dump(output / "results.json", rows)
                dump(output / "progress.json", dict(completed=len(rows), last=row))
                if not check_gate:
                    raise ValueError(f"Numerical gate failed: {row}")
            del value, u, s, vh
            torch.cuda.empty_cache()
    # Exercise fallback on a deliberately nonorthogonal input, separately from near-frame speedups.
    x = torch.randn(97, 31, device="cuda", dtype=torch.float64)
    actual, info = retract(x, return_info=True)
    reference = retract(x, method="svd")
    torch.testing.assert_close(actual, reference, atol=1e-10, rtol=1e-10)
    assert info["used"] == "svd"
    dump(
        output / "complete.json",
        dict(
            matrices=len(rows),
            passed=True,
            fallback=info,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            completed_at=time.time(),
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "output", "stop"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--job-ids", nargs="+", type=int, default=[0, 6, 7, 11, 19])
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--smoke", action="store_true")
    benchmark(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
