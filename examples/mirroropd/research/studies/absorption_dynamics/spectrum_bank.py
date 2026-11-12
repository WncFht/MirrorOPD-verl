# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Prepare and execute paired sigma0 intervention scores on existing full fixed banks."""

import argparse
import multiprocessing
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

from examples.mirroropd.research.runtime.processes import identity
from verl.trainer.evaluation.artifacts import file_sha256

from ..spectral.update_dynamics import atomic_json
from .spectrum_bank_inputs import prepare
from .spectrum_bank_worker import read_job, run_pair


def execute(
    plan,
    output,
    pairs,
    devices,
    *,
    token_chunk=256,
    smoke=False,
    probe_only=False,
    edit_method="svd",
    reference_cache_mib=0,
):
    import json
    import os

    plan = json.loads(Path(plan).read_text())
    if len(set(devices)) != len(devices) or len(set(pairs)) != len(pairs) or not pairs or not devices:
        raise ValueError("Select distinct nonempty pairs/devices")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(
        output / "protocol.json",
        dict(
            plan=plan,
            pairs=pairs,
            devices=devices,
            token_chunk=token_chunk,
            smoke=smoke,
            probe_only=probe_only,
            edit_method=edit_method,
            reference_cache_mib=reference_cache_mib,
        ),
    )
    atomic_json(output / "process.json", dict(pid=os.getpid(), identity=identity(os.getpid()), started_at=time.time()))
    pools = [ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) for _ in devices]
    queue = iter(pairs)
    pending = {}
    done, failed = 0, 0
    started = time.monotonic()

    def submit(slot):
        name = next(queue, None)
        if name is None or (output / "STOP").exists():
            return
        item = plan["pairs"][name]
        if file_sha256(item["input"]) != item["sha256"]:
            raise ValueError("Frozen bank intervention inputs changed")
        future = pools[slot].submit(
            run_pair,
            read_job(item["input"]),
            output / name,
            devices[slot],
            token_chunk=token_chunk,
            smoke=smoke,
            probe_only=probe_only,
            edit_method=edit_method,
            reference_cache_mib=reference_cache_mib,
        )
        pending[future] = (slot, name)

    try:
        for slot in range(len(devices)):
            submit(slot)
        while pending:
            ready, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in ready:
                slot, name = pending.pop(future)
                try:
                    result = future.result()
                    if not result["complete"]:
                        raise ValueError("Incomplete pair results")
                    done += 1
                except Exception as error:
                    failed += 1
                    atomic_json(output / f"{name}-error.json", dict(error=repr(error), at=time.time()))
                atomic_json(
                    output / "progress.json",
                    dict(complete=done, failed=failed, total=len(pairs), seconds=time.monotonic() - started),
                )
                submit(slot)
    finally:
        for pool in pools:
            pool.shutdown()
    atomic_json(
        output / "exit.json", dict(complete=done, failed=failed, total=len(pairs), seconds=time.monotonic() - started)
    )
    if done != len(pairs) or failed:
        raise RuntimeError("Spectrum bank scoring incomplete; preserve errors and row journals")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    build = sub.add_parser("prepare")
    build.add_argument("--geometry-manifest", type=Path, action="append", required=True)
    build.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--pairs", nargs="+", required=True)
    run.add_argument("--devices", nargs="+", required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--token-chunk", type=int, default=256)
    run.add_argument("--smoke", action="store_true")
    run.add_argument("--probe-only", action="store_true")
    run.add_argument("--edit-method", choices=("svd", "gram"), default="svd")
    run.add_argument("--reference-cache-mib", type=int, default=0)
    args = vars(parser.parse_args())
    action = args.pop("action")
    if action == "prepare":
        prepare(args["geometry_manifest"], args["output"])
    else:
        execute(**args)


if __name__ == "__main__":
    main()
