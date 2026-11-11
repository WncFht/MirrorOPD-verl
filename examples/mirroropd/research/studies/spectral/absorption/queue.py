# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Durable absorption queue using existing broker leases and frozen package code.

Tasks explicitly name job ID, phase, input attempt and output attempt. A queue
owns its own lock/STOP and never discovers or takes over other supervisors.
"""

import argparse
import concurrent.futures
import fcntl
import hashlib
import os
import sys
import time
import traceback
from pathlib import Path

from examples.mirroropd.research.core.io import dump, read
from examples.mirroropd.research.runtime.broker import occupied, run_leased
from examples.mirroropd.research.runtime.processes import identity
from examples.mirroropd.research.runtime.source import freeze

MODULE = "examples.mirroropd.research.studies.spectral.absorption"


def initialize(root, manifest, prompts, tasks, repo, *, stop_paths=()):
    """Create immutable inputs/source and a fresh mutable task ledger.

    Imported evidence stays in its original attempt. New workers can reuse bank,
    reference and direction files via read-only-by-convention links; gradients
    always have a fresh output directory. Completed old stages are linked whole.
    """
    root.mkdir(parents=True, exist_ok=False)
    dump(root / "manifest.json", manifest)
    dump(root / "prompts.json", prompts)
    dump(root / "plan.json", tasks)
    dump(root / "settings.json", dict(stop_paths=[str(p) for p in stop_paths]))
    freeze(repo, root / "source")
    hashes = {
        str(path.relative_to(root / "source")): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (root / "source").rglob("*.py")
    }
    dump(root / "source-hashes.json", hashes)
    dump(root / "queue.json", {"jobs": [dict(task, task_id=i, status="pending") for i, task in enumerate(tasks)]})


def update(root, mutate):
    with (root / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = read(root / "queue.json")
        result = mutate(state)
        dump(root / "queue.json", state)
        return result


def claim(root, gpu):
    def change(state):
        settings = read(root / "settings.json") if (root / "settings.json").exists() else {}
        if settings.get("allowed_gpus") is not None and gpu not in settings["allowed_gpus"]:
            raise PermissionError(f"Physical GPU {gpu} is outside this queue allocation")
        if any(path.exists() for path in [root / "STOP", *map(Path, settings.get("stop_paths", []))]):
            return None
        for task in state["jobs"]:
            if task["status"] != "pending":
                continue
            if task.get("ready_file") and not Path(task["ready_file"]).exists():
                continue
            task.update(
                status="running",
                gpu=gpu,
                started_at=time.time(),
                worker_identity=identity(os.getpid()),
                heartbeat_at=time.time(),
            )
            return dict(task)
        return None

    return update(root, change)


def reconcile_running(root):
    """Classify abandoned tasks without retrying, signalling or deleting attempts.

    A matching live worker or child blocks supervisor takeover. Legacy tasks
    without process identities require review, rather than assuming they died.
    """

    def change(state):
        live, interrupted, unknown = [], [], []
        for task in state["jobs"]:
            if task["status"] != "running":
                continue
            records = [task.get("worker_identity"), task.get("child_identity")]
            if any(record and identity(record[0]) == record for record in records):
                live.append(task["task_id"])
                continue
            if not task.get("worker_identity"):
                unknown.append(task["task_id"])
                continue
            task.update(
                status="interrupted",
                interrupted_at=time.time(),
                reason="Recorded worker and child identities exited; attempt preserved; explicit retry required",
            )
            interrupted.append(task["task_id"])
        return dict(live=live, interrupted=interrupted, unknown=unknown)

    result = update(root, change)
    dump(root / "recovery" / f"inspection-{time.time_ns()}.json", result)
    return result


def prepare_attempt(task):
    destination = Path(task["output_attempt"])
    if destination.exists():
        raise FileExistsError(f"Preserve previous attempt; supply a new output_attempt: {destination}")
    destination.mkdir(parents=True)
    if task.get("input_attempt"):
        source = Path(task["input_attempt"])
        for name in (
            "banks.json",
            "reference.pt",
            "matrix-index.json",
            "directions",
            "scores",
            "no-edit-gate.json",
            "interventions-complete.json",
        ):
            path = source / name
            if path.exists():
                (destination / name).symlink_to(path.absolute(), target_is_directory=path.is_dir())
        for complete in (source / "gradients").glob("*/complete.json"):
            target = destination / "gradients" / complete.parent.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(complete.parent.absolute(), target_is_directory=True)
    dump(destination / "inputs.json", task)
    return destination


def worker(root, gpu):
    # Runs inside one guard. The source snapshot precedes the workspace on imports.
    for relative, expected in read(root / "source-hashes.json").items():
        if hashlib.sha256((root / "source" / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Changed frozen source: {relative}")
    while True:
        task = claim(root, gpu)
        if task is None:
            state = read(root / "queue.json")["jobs"]
            stops = [root / "STOP", *map(Path, read(root / "settings.json").get("stop_paths", []))]
            if any(path.exists() for path in stops):
                return
            if any(t["status"] == "pending" for t in state) and any(t["status"] == "running" for t in state):
                time.sleep(2)
                continue
            return
        code = None
        try:
            out = prepare_attempt(task)
            command = [
                sys.executable,
                "-c",
                "import runpy,sys;sys.path.insert(0,sys.argv.pop(1));"
                "runpy.run_module(sys.argv.pop(1),run_name='__main__')",
                str(root / "source"),
                MODULE,
                "--run-root",
                str(root),
                "--job",
                str(task["job_id"]),
                "--attempt",
                str(out),
                "--phase",
                task["phase"],
                "--projection",
                task.get("projection", "cpu-reference"),
            ]
            if task.get("module"):
                command = [
                    sys.executable,
                    "-c",
                    "import runpy,sys;sys.path.insert(0,sys.argv.pop(1));"
                    "runpy.run_module(sys.argv.pop(1),run_name='__main__')",
                    str(root / "source"),
                    task["module"],
                    *[value.replace("{attempt}", str(out)) for value in task.get("arguments", [])],
                ]
            else:
                for stop in read(root / "settings.json").get("stop_paths", []):
                    command += ["--stop-path", stop]
            import subprocess

            dump(out / "command.json", dict(argv=command, started_at=time.time()))
            with (out / "command.log").open("x") as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)

                def started(state, task=task, child=child):
                    state["jobs"][task["task_id"]].update(child_identity=identity(child.pid), heartbeat_at=time.time())

                try:
                    update(root, started)
                    last_heartbeat = time.monotonic()
                    while child.poll() is None:
                        if time.monotonic() - last_heartbeat >= 10:

                            def heartbeat(state, task=task):
                                state["jobs"][task["task_id"]]["heartbeat_at"] = time.time()

                            update(root, heartbeat)
                            last_heartbeat = time.monotonic()
                        stops = [root / "STOP", *map(Path, read(root / "settings.json").get("stop_paths", []))]
                        if any(path.exists() for path in stops):
                            child.terminate()
                            break
                        time.sleep(1)
                    code = child.wait(timeout=90)
                finally:
                    if child.poll() is None:
                        child.terminate()
                        try:
                            child.wait(timeout=90)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait(timeout=30)
            marker = "interventions-complete.json" if task["phase"] == "interventions" else "complete.json"
            complete = code == 0 and (out / task.get("complete_marker", marker)).exists()
            dump(out / "exit.json", dict(returncode=code, complete=complete, finished_at=time.time()))
            if not complete:
                raise RuntimeError(f"Incomplete task {task['task_id']}: returncode={code}")
        except Exception:
            complete = False
            dump(root / "errors" / f"task-{task['task_id']}.json", dict(error=traceback.format_exc()))

        def finish(state, task=task, complete=complete, code=code):
            state["jobs"][task["task_id"]].update(
                status="complete" if complete else "failed", returncode=code, finished_at=time.time()
            )

        update(root, finish)


def run(root, repo, resource, gpus):
    with (root / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        recovery = reconcile_running(root)
        if recovery["live"] or recovery["unknown"]:
            raise RuntimeError(f"Existing tasks prevent takeover; inspect process identities: {recovery}")
        settings = read(root / "settings.json")
        if settings.get("allowed_gpus") is not None and not set(gpus).issubset(settings["allowed_gpus"]):
            raise PermissionError("Requested physical GPUs exceed this queue allocation")
        stops = [root / "STOP", *map(Path, settings.get("stop_paths", []))]

        def lane(gpu):
            while not occupied(resource, gpu):
                if any(path.exists() for path in stops):
                    return
                time.sleep(5)
            # Keep waiting if prerequisite evidence is still being produced by an old queue.
            while True:
                tasks = read(root / "queue.json")["jobs"]
                pending = [t for t in tasks if t["status"] == "pending"]
                if not pending or any(path.exists() for path in stops):
                    return
                if any(not t.get("ready_file") or Path(t["ready_file"]).exists() for t in pending):
                    break
                time.sleep(5)
            code = run_leased(
                resource=resource,
                repo=repo,
                gpu=gpu,
                owner=f"{root.name}-gpu{gpu}",
                output=root / "lanes" / str(gpu),
                stop_paths=stops,
                command=[
                    sys.executable,
                    "-c",
                    "import runpy,sys;sys.path.insert(0,sys.argv.pop(1));"
                    "runpy.run_module(sys.argv.pop(1),run_name='__main__')",
                    str(root / "source"),
                    MODULE + ".queue",
                    "worker",
                    "--run-root",
                    str(root),
                    "--gpu",
                    str(gpu),
                ],
            )
            if code:
                raise RuntimeError(f"GPU {gpu} worker exited {code}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures = {pool.submit(lane, gpu): gpu for gpu in gpus}
            for future in concurrent.futures.as_completed(futures):
                gpu = futures[future]
                try:
                    future.result()
                except Exception:
                    dump(root / "errors" / f"lane-{gpu}.json", dict(error=traceback.format_exc()))
        state = read(root / "queue.json")
        restored = all(occupied(resource, gpu) for gpu in gpus)
        dump(
            root / "exit.json",
            dict(
                queue=state,
                occupancy_restored=restored,
                finished_at=time.time(),
                complete=all(t["status"] == "complete" for t in state["jobs"]),
            ),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "worker", "recover"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--resource", type=Path)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--gpus", type=int, nargs="+")
    args = parser.parse_args()
    if args.mode == "recover":
        with (args.run_root / "supervisor.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            print(reconcile_running(args.run_root))
    elif args.mode == "worker":
        worker(args.run_root, args.gpu)
    else:
        if not args.repo or not args.resource or not args.gpus or len(set(args.gpus)) != len(args.gpus):
            parser.error("run requires --repo, --resource and distinct --gpus")
        run(args.run_root, args.repo, args.resource, args.gpus)


if __name__ == "__main__":
    main()
