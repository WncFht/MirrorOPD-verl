# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""CLI and deferred, exclusively guarded GPU pipeline."""

import argparse
import concurrent.futures
import fcntl
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

from .data import contract, freeze, prepare, read, write
from .objectives import ARMS

MODULE = "examples.mirroropd.research.studies.fixed_transfer.sft_distribution"


def invoke(root, stage, *, visible=None, **kwargs):
    command = [sys.executable, "-u", "-m", MODULE, stage, "--root", str(root)]
    for key, value in kwargs.items():
        command.extend(["--" + key.replace("_", "-"), str(value)])
    label = "-".join([stage, *map(str, kwargs.values())])
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    env = dict(
        os.environ,
        VLLM_PLUGINS="",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        TOKENIZERS_PARALLELISM="false",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
    )
    if visible is not None:
        env["CUDA_VISIBLE_DEVICES"] = visible
    with (logs / f"{label}.log").open("a") as stream:
        subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def parallel(calls):
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(calls)) as pool:
        jobs = [pool.submit(function, *args, **kwargs) for function, args, kwargs in calls]
        errors = []
        for job in concurrent.futures.as_completed(jobs):
            try:
                job.result()
            except Exception as error:
                errors.append(error)
        if errors:
            raise RuntimeError(f"{len(errors)} pipeline workers failed; inspect per-stage logs") from errors[0]


def pipeline(root):
    cfg = contract(root)
    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if len(devices) != 4 or cfg["expert_tp"] != 2:
        raise ValueError("This pipeline reserves exactly four GPUs, with two TP2 expert lanes")
    # Exclusive allocation receipt is checked before any model import/initialization.
    guard = resource_guard()
    guard.verify_allocation(Path(__file__).resolve().parents[7], ",".join(devices))
    if not (root / "gpu-acceptance.json").exists():
        write(root / "queue.json", dict(state="gpu-acceptance", updated=time.time()))
        invoke(root, "acceptance", visible=devices[0])
    for split in ("calibration", "train"):
        if (root / f"quality-{split}.json").exists() and read(root / f"quality-{split}.json")["passed"]:
            continue
        if split == "train" and not read(root / "quality-calibration.json")["passed"]:
            raise ValueError("Calibration must pass before bulk generation")
        write(root / "queue.json", dict(state="generating", split=split, updated=time.time()))
        pairs = [",".join(devices[:2]), ",".join(devices[2:])]
        parallel(
            [
                (invoke, (root, "generate"), dict(visible=pair, split=split, shard=i, shards=2))
                for i, pair in enumerate(pairs)
            ]
        )
        write(root / "queue.json", dict(state="scoring", split=split, updated=time.time()))
        for kind in ("base", "expert"):
            parallel(
                [
                    (invoke, (root, "score"), dict(visible=pair, split=split, kind=kind, shard=i, shards=2))
                    for i, pair in enumerate(pairs)
                ]
            )
        invoke(root, "collect", split=split)
    if not (root / "frozen.json").exists():
        freeze(root)

    # Each arm owns one GPU; the two seeds run in sequence on that lane.
    def lane(arm, gpu):
        for seed in cfg["seeds"]:
            invoke(root, "train", visible=gpu, arm=arm, seed=seed)

    write(root / "queue.json", dict(state="sft-training", updated=time.time()))
    parallel([(lane, (arm, gpu), {}) for arm, gpu in zip(ARMS, devices, strict=True)])
    write(
        root / "queue.json",
        dict(
            state="teachers-complete",
            updated=time.time(),
            scope="four SFT arms, two seeds; capability/OOD evaluation not completed",
        ),
    )


def resource_guard():
    path = Path(__file__).resolve().parents[7] / "runs/gpu-expansion-20261030/idle-occupancy/runtime/resource_guard.py"
    spec = importlib.util.spec_from_file_location("sft_distribution_resource_guard", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wait_run(root, gpus, poll):
    cfg = contract(root)
    guard = resource_guard()
    requested = guard.parse_gpus(gpus)
    if set(requested) != {0, 5, 6, 7}:
        raise ValueError("This run is restricted to physical GPUs 0,5,6,7; existing E93 allocation is excluded")
    if not 10 <= poll <= 300:
        raise ValueError("Resource poll interval must be 10–300 seconds")
    with (root / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(root / "queue-owner.json", dict(pid=os.getpid(), started=time.time(), gpus=requested))
        while True:
            pending = [str(p) for p in cfg.get("wait_for", []) if not Path(p).is_file()]
            if pending:
                write(
                    root / "queue.json",
                    dict(
                        state="waiting-for-prior-evaluation",
                        pending=pending,
                        updated=time.time(),
                        owner_pid=os.getpid(),
                    ),
                )
                time.sleep(poll)
                continue
            write(
                root / "queue.json",
                dict(state="waiting-for-exclusive-gpus", gpus=requested, updated=time.time(), owner_pid=os.getpid()),
            )
            try:
                code = guard.run(
                    Path(__file__).resolve().parents[6],
                    gpus,
                    [sys.executable, "-u", "-m", MODULE, "pipeline", "--root", str(root)],
                )
            except ValueError as error:
                if "already locked" not in str(error) and "existing compute process" not in str(error):
                    raise
                time.sleep(poll)
                continue
            if code:
                write(root / "queue.json", dict(state="needs-attention", returncode=code, updated=time.time()))
                raise RuntimeError("Pipeline stopped; inspect logs, do not overwrite scientific trajectories")
            return


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stage",
        choices=("prepare", "acceptance", "generate", "score", "collect", "freeze", "train", "pipeline", "wait-run"),
    )
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--plan", type=Path)
    p.add_argument("--split", choices=("calibration", "train"), default="calibration")
    p.add_argument("--kind", choices=("base", "expert"), default="base")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--arm", choices=ARMS, default="RAW-C")
    p.add_argument("--seed", type=int, default=20261109)
    p.add_argument("--resume", type=Path)
    p.add_argument("--gpus", default="0,5,6,7")
    p.add_argument("--poll", type=int, default=60)
    a = p.parse_args()
    a.root = a.root.resolve()
    if not 0 <= a.shard < a.shards:
        p.error("Require 0 <= shard < shards")
    if a.stage == "prepare":
        if not a.plan:
            p.error("prepare requires --plan")
        prepare(a.root, a.plan)
    elif a.stage == "freeze":
        freeze(a.root)
    elif a.stage == "pipeline":
        pipeline(a.root)
    elif a.stage == "wait-run":
        wait_run(a.root, a.gpus, a.poll)
    else:
        from . import runtime

        if a.stage == "acceptance":
            runtime.acceptance(a.root)
        elif a.stage == "generate":
            runtime.generate(a.root, a.split, a.shard, a.shards)
        elif a.stage == "score":
            runtime.score(a.root, a.split, a.kind, a.shard, a.shards)
        elif a.stage == "collect":
            runtime.collect_scores(a.root, a.split)
        elif a.stage == "train":
            runtime.train(a.root, a.arm, a.seed, a.resume)


if __name__ == "__main__":
    main()
