# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Synthetic-gradient GPU regression for the complete frame optimizer step."""

import argparse
import json
import time
from pathlib import Path

import torch

from examples.mirroropd.research.studies.spectral.iso.retraction import spectrum_gradient
from examples.mirroropd.research.studies.spectral.iso.training import FactorAdam


def main():
    parser = argparse.ArgumentParser(description="Matched-gradient, four-step FactorAdam GPU validation")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    if out.exists():
        raise FileExistsError(f"Benchmark evidence already exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(37)
    rows = []
    for shape in [(256, 1536), (1024, 2048), (2048, 2048)]:
        initial = torch.randn(*shape, device="cuda") * 0.02
        a = torch.nn.Parameter(initial.clone())
        b = torch.nn.Parameter(initial.clone())
        old = FactorAdam([("weight", a)], 1e-5, "frames", retraction="svd")
        new = FactorAdam([("weight", b)], 1e-5, "frames", retraction="newton")
        times = {"svd": [], "newton": []}
        errors = []
        for step in range(4):
            g = torch.randn_like(a) * 0.01
            a.grad = g
            b.grad = g
            for name, opt in [("svd", old), ("newton", new)]:
                torch.cuda.synchronize()
                t = time.perf_counter()
                opt.step()
                torch.cuda.synchronize()
                times[name].append(time.perf_counter() - t)
            error = float((a - b).norm() / a.norm())
            errors.append(error)
            assert error < 1e-5, (shape, step, error)
        u, s, v = old.entries[0][2:5]
        g = torch.randn_like(a)
        reference = (u.T @ g @ v).diagonal()
        actual = spectrum_gradient(g, u, v)
        torch.testing.assert_close(actual, reference, atol=5e-5, rtol=1e-3)
        row = dict(
            shape=shape,
            seconds=times,
            weight_relative_errors=errors,
            speedup=sum(times["svd"]) / sum(times["newton"]),
            spectrum_gradient_max_error=float((actual - reference).abs().max()),
        )
        rows.append(row)
        out.write_text(json.dumps(rows, indent=2))
        del old, new, a, b, u, s, v, g, initial
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
