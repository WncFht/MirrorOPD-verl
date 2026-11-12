# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Compare exact trajectory engines on real tasks under an externally leased GPU."""

import argparse
import gzip
import importlib.util
import json
import math
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .trajectory import measure
from .update_dynamics import atomic_json


def compare(reference, actual, path=""):
    """Validate every scientific field, including full spectra and angle distributions."""
    differences = []
    if isinstance(reference, dict):
        for key, value in reference.items():
            if key not in {"execution", "device", "timing_seconds", "wall_seconds"}:
                if key not in actual:
                    differences.append([path + "/" + key, "missing"])
                else:
                    differences.extend(compare(value, actual[key], path + "/" + key))
    elif isinstance(reference, list):
        if len(reference) != len(actual):
            return [[path, "length mismatch"]]
        for i, (a, b) in enumerate(zip(reference, actual, strict=False)):
            differences.extend(compare(a, b, f"{path}/{i}"))
    elif isinstance(reference, int | float) and not isinstance(reference, bool):
        tolerance = 2e-4 if "angle" in path else (2e-6 if "chordal_distance" in path else 1e-8)
        if not isinstance(actual, int | float) or not math.isclose(reference, actual, rel_tol=1e-6, abs_tol=tolerance):
            differences.append([path, reference, actual])
    elif reference != actual:
        differences.append([path, reference, actual])
    return differences


def benchmark(task, device, reference_file, output):
    spec = importlib.util.spec_from_file_location(
        "examples.mirroropd.research.studies.spectral._trajectory_reference", reference_file
    )
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    rows = []
    reference = None
    for label, function, options in [
        ("reference-default", old.measure, {}),
        ("shared-default", measure, {}),
        ("shared-gesvd", measure, {"driver": "gesvd"}),
    ]:
        record = function(task, device, **options)
        differences = [] if reference is None else compare(reference, record)
        reference = record if reference is None else reference
        target = Path(output) / f"{device.replace(':', '-')}-{label}.json.gz"
        target.write_bytes(gzip.compress(json.dumps(record, allow_nan=False).encode()))
        row = dict(
            label=label,
            seconds=record["wall_seconds"],
            timings=record["timing_seconds"],
            passed=not differences,
            differences=differences,
            result=str(target),
        )
        rows.append(row)
        atomic_json(Path(output) / f"{device.replace(':', '-')}.json", dict(task=task, variants=rows))
    return dict(task=task, device=device, variants=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tasks = json.loads(args.tasks.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=len(tasks), mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(benchmark, t, f"cuda:{i}", args.reference, args.output) for i, t in enumerate(tasks)]
        for future in as_completed(futures):
            results.append(future.result())
            atomic_json(args.output / "summary.json", dict(started_at=started, results=results))
    passed = all(v["passed"] for r in results for v in r["variants"])
    atomic_json(args.output / "exit.json", dict(passed=passed, seconds=time.time() - started))
    if not passed:
        raise RuntimeError("Numerical acceptance failed; inspect per-field differences")


if __name__ == "__main__":
    main()
