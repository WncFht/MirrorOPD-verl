# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded real-matrix scan benchmark, executed only inside a caller-owned guard."""

import argparse
import json
from pathlib import Path

import torch

from examples.mirroropd.research.core.io import dump

from .decomposition import weight_summary
from .scan import scan
from .weights import TensorCheckpoint, analyze


def records(root):
    return [d for p in Path(root).glob("*.json") if "targets" in (d := json.loads(p.read_text()))]


def benchmark(manifest, job_id, output, stop_paths=()):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    pattern = r"model\.layers\.0\.(self_attn\.(q_proj|k_proj)|mlp\.(up_proj|down_proj))\.weight$"
    options = dict(manifest=manifest, job_ids=[job_id], device="cuda:0", pattern=pattern, stop_paths=stop_paths)
    for name, extra in (
        ("weights", dict()),
        ("weights-transpose", dict(transpose_wide=True)),
        ("updates", dict(transpose_wide=True, updates=True)),
        ("updates-warm", dict(transpose_wide=True, updates=True)),
    ):
        scan(output=output / name, cache=output / "cache", driver="gesvd", **options, **extra)
    # Compare the reduced path to the existing complete diagnostics on one real
    # k matrix on CPU FP64, including rank of the update in the reference path.
    job = next(j for j in json.loads(Path(manifest).read_text())["jobs"] if j["job_id"] == job_id)
    key = "model.layers.0.self_attn.k_proj.weight"
    raw = [TensorCheckpoint(job[k]).get(key).double() for k in ("base", "teacher_path", "endpoint")]
    errors = []
    actual = next(r for r in records(output / "weights") if r["tensor"] == key)
    for path, target in zip((job["teacher_path"], job["endpoint"]), raw[1:], strict=True):
        reference = weight_summary(raw[0], target)
        for metric in ("delta_sigma", "rho_sigma", "kappa_spec"):
            value, expected = actual["targets"][path][metric], reference[metric]
            error = abs(value - expected) / max(abs(expected), 1e-12)
            if error > 1e-6:
                raise ValueError(f"Real CPU FP64 gate failed: {metric}: {error}")
            errors.append(dict(path=path, metric=metric, relative_error=error))
    original = {r["tensor"]: r for r in records(output / "weights")}
    for row in records(output / "weights-transpose"):
        for path, facts in row["targets"].items():
            for metric in ("delta_sigma", "rho_sigma", "kappa_spec"):
                expected = original[row["tensor"]]["targets"][path][metric]
                error = abs(facts[metric] - expected) / max(abs(expected), 1e-12)
                if error > 1e-6:
                    raise ValueError(f"Transpose gate failed: {row['tensor']}, {metric}: {error}")
    checkpoints = [TensorCheckpoint(job[k]) for k in ("base", "teacher_path", "endpoint")]
    compatible = True
    try:
        for checkpoint in checkpoints[1:]:
            checkpoints[0].compatible(checkpoint)
    except ValueError as error:
        compatible = False
        dump(output / "legacy-scope.json", dict(executed=False, reason=str(error)))
    if compatible:
        analyze(
            job["base"],
            job["teacher_path"],
            job["endpoint"],
            output / "legacy-k",
            pattern=r"model\.layers\.0\.self_attn\.k_proj\.weight$",
            device="cuda:0",
            driver="gesvd",
        )
        dump(output / "legacy-scope.json", dict(executed=True))
    dump(output / "numerical-gate.json", dict(passed=True, cpu_fp64_errors=errors, transpose_gate=True))
    dump(output / "complete.json", dict(job_id=job_id, teacher=job["teacher"], passed=True))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--job-id", type=int, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--stop-paths", nargs="*", default=[])
    args = vars(p.parse_args())
    torch.set_num_threads(2)
    benchmark(**args)


if __name__ == "__main__":
    main()
