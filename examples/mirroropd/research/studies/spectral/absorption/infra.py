# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Lease one naturally released lane for numerical/performance benchmarks."""

import argparse
import sys
from pathlib import Path

from examples.mirroropd.research.runtime.broker import run_leased


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--resource", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-path", type=Path, action="append", default=[])
    parser.add_argument("--model-run-root", type=Path)
    parser.add_argument("--model-job", type=int)
    args = parser.parse_args()
    command = [
        sys.executable,
        "-m",
        "examples.mirroropd.research.studies.spectral.absorption."
        + ("benchmark_model" if args.model_run_root else "benchmark"),
        "--attempt",
        str(args.attempt),
        "--output",
        str(args.output / "projection.json"),
    ]
    if args.model_run_root:
        if args.model_job is None:
            parser.error("--model-run-root requires --model-job")
        command += ["--run-root", str(args.model_run_root), "--job", str(args.model_job)]
    code = run_leased(
        resource=args.resource,
        repo=args.repo,
        gpu=args.gpu,
        owner=args.output.name,
        output=args.output,
        command=command,
        stop_paths=args.stop_path,
    )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
