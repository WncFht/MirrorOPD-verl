# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Wait for broker-owned idle capacity, replay saved tokens, then publish CPU analysis."""

import argparse
import fcntl
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from examples.mirroropd.research.core.io import dump, read
from examples.mirroropd.research.runtime.broker import occupied, run_leased


def run(args):
    repo, plan_path = Path.cwd(), args.plan.resolve()
    plan = read(plan_path)
    root = plan_path.parent
    stop = root / "STOP"
    deadline = time.monotonic() + args.wait_seconds
    env = dict(
        os.environ, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false"
    )
    module = "examples.mirroropd.research.studies.latent_state"
    pairs = list(plan["pairs"])
    while True:
        if stop.exists():
            raise InterruptedError(f"Stop requested via {stop}")
        available = [gpu for gpu in args.gpus if occupied(args.broker, gpu)]
        if len(available) >= min(args.workers, len(pairs)):
            break
        dump(root / "queue-status.json", dict(state="waiting_for_broker", updated_at=time.time()))
        if time.monotonic() >= deadline:
            raise TimeoutError("No broker-owned idle capacity before the queue deadline")
        time.sleep(15)

    def lane(gpu, assigned):
        for pair in assigned:
            allocation = root / "allocations" / f"{pair}-{time.time_ns()}"
            code = run_leased(
                resource=args.broker,
                repo=repo,
                gpu=gpu,
                owner=f"latent-reference-{pair}",
                output=allocation,
                command=[sys.executable, "-m", module, "extract", "--plan", str(plan_path), "--pair", pair],
                stop_paths=[stop],
                env=env,
            )
            if code:
                raise RuntimeError(f"State extraction failed: {pair}; inspect {allocation}")

    workers = min(args.workers, len(pairs))
    dump(root / "queue-status.json", dict(state="extracting", gpus=available[:workers], updated_at=time.time()))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(lane, available[i], pairs[i::workers]) for i in range(workers)]
        for future in futures:
            future.result()
    if not args.extract_only:
        for pair in pairs:
            subprocess.run(
                [str(args.analysis_python), "-m", module, "analyze", "--plan", str(plan_path), "--pair", pair],
                cwd=repo,
                env=env,
                check=True,
            )
        subprocess.run(
            [str(args.analysis_python), "-m", module, "report", "--plan", str(plan_path), "--output", str(args.output)],
            cwd=repo,
            env=env,
            check=True,
        )
    dump(
        root / "queue-status.json", dict(state="extracted" if args.extract_only else "complete", updated_at=time.time())
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--broker", type=Path, required=True)
    parser.add_argument("--analysis-python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--wait-seconds", type=int, default=86400)
    args = parser.parse_args()
    if args.workers < 1 or len(args.gpus) != len(set(args.gpus)) or args.workers > len(args.gpus):
        parser.error("Require positive workers and sufficient unique GPUs")
    with (args.plan.parent / "queue.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(args)
        except BaseException as error:
            dump(
                args.plan.parent / "queue-status.json", dict(state="failed", error=repr(error), updated_at=time.time())
            )
            raise


if __name__ == "__main__":
    main()
