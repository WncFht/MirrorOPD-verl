# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Execute an OPD absorption job using an explicit manifest and run directory."""

import argparse
from pathlib import Path

from examples.mirroropd.research.core.io import read

from .protocol import RunContext


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--job", type=int, required=True)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--phase", choices=("all", "interventions", "gradients"), default="all")
    parser.add_argument("--projection", choices=("cpu-reference", "cuda-stream"), default="cpu-reference")
    parser.add_argument("--stop-path", type=Path, action="append", default=[])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    context = RunContext(args.run_root, args.projection, tuple(args.stop_path))
    context.check_stop()
    jobs = read(context.root / "manifest.json")["jobs"]
    matches = [job for job in jobs if job["job_id"] == args.job]
    if len(matches) != 1:
        raise ValueError(f"Expected one manifest job with ID {args.job}")
    from .experiment import run

    run(context, matches[0], args.attempt, smoke=args.smoke, phase=args.phase)


if __name__ == "__main__":
    main()
