# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Dependency-aware round-two queue; preserve GPU ownership and restore occupation."""

import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from examples.mirroropd.research.core.io import dump, read
from examples.mirroropd.research.core.paths import REPO
from examples.mirroropd.research.runtime import occupation as resource
from examples.mirroropd.research.runtime.processes import identity
from examples.mirroropd.research.studies.boundary.round2_common import ARMS, OLD_OUT, OUT

MODULE = "examples.mirroropd.research.studies.boundary"


def prepare_resources():
    if not (OUT / "occupy-management.json").exists():
        dump(OUT / "occupy-management.json", read(OLD_OUT / "occupy-management.json"))


def job_list():
    jobs = []
    for index, (name, spec) in enumerate(read(OUT / "eval-specs.json").items()):
        dependencies = []
        priority = 5 if spec["kind"] == "teacher" else 60
        if spec["kind"] == "new-endpoint":
            dependencies = ["train-" + spec["arm"]]
            priority = 40
        jobs.append(
            dict(
                id=name,
                command=["round2_eval", name, "--port", str(19040 + index)],
                priority=priority,
                dependencies=dependencies,
                complete=str(OUT / "evaluations" / name / "complete.json"),
            )
        )
    jobs.append(
        dict(
            id="optimize",
            command=["fast_profile"],
            priority=10,
            dependencies=[],
            ready_file=str(OUT / "optimization-code-ready.json"),
            complete=str(OUT / "optimization/complete.json"),
        )
    )
    for arm, config in ARMS.items():
        dependencies = ["optimize"]
        if not config["old_arm"]:
            dependencies.append("teacher-" + config["teacher"])
        jobs.append(
            dict(
                id="train-" + arm,
                command=["fast_train", arm],
                priority=20,
                dependencies=dependencies,
                complete=str(OUT / "arms" / arm / "complete.json"),
            )
        )
    scientific_jobs = [job["id"] for job in jobs if job["id"].startswith("train-") or job["id"] == "optimize"]
    for name, module, marker in [
        ("optimize-batched", "fast_batch_profile", "optimization-batched/complete.json"),
        ("teacher-copy", "copy_reference", "teacher-copy/complete.json"),
    ]:
        jobs.append(
            dict(id=name, command=[module], priority=80, dependencies=scientific_jobs, complete=str(OUT / marker))
        )
    return sorted(jobs, key=lambda j: j["priority"])


def run_queue(directory, tag):
    jobs = job_list()
    dump(directory / "jobs.json", jobs)
    done = {j["id"] for j in jobs if Path(j["complete"]).exists()}
    failed = set()
    pending = [j for j in jobs if j["id"] not in done]
    running = []
    last_sample = 0.0
    while pending or running:
        busy = {x[2] for x in running}
        for job in list(pending):
            available = [g for g in [4, 5, 6, 7] if g not in busy]
            if not available:
                break
            if not set(job["dependencies"]) <= done:
                continue
            if job.get("ready_file") and not Path(job["ready_file"]).exists():
                continue
            gpu = available[0]
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=str(gpu),
                OMP_NUM_THREADS="4",
                OPENBLAS_NUM_THREADS="4",
                TOKENIZERS_PARALLELISM="false",
                HF_HUB_OFFLINE="1",
                VLLM_WORKER_MULTIPROC_METHOD="spawn",
            )
            env[resource.TAG] = tag
            stream = (directory / (job["id"] + ".log")).open("a")
            command = [sys.executable, "-m", MODULE + "." + job["command"][0], *job["command"][1:]]
            child = subprocess.Popen(
                command,
                cwd=REPO,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            dump(
                directory / (job["id"] + "-launch.json"),
                dict(identity=identity(child.pid), physical_gpu=gpu, command=command, time=time.time()),
            )
            running.append((job, child, gpu, stream))
            pending.remove(job)
            busy.add(gpu)
        for item in list(running):
            job, child, gpu, stream = item
            code = child.poll()
            if code is None:
                continue
            stream.close()
            running.remove(item)
            success = code == 0 and Path(job["complete"]).exists()
            (done if success else failed).add(job["id"])
            dump(directory / (job["id"] + "-exit.json"), dict(exit_code=code, success=success, time=time.time()))
        dump(
            directory / "queue.json",
            dict(
                running=[x[0]["id"] for x in running],
                pending=[j["id"] for j in pending],
                done=sorted(done),
                failed=sorted(failed),
                time=time.time(),
            ),
        )
        if time.monotonic() - last_sample > 15:
            sample = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu,memory.used,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            with (directory / "gpu-samples.jsonl").open("a") as stream:
                import json

                stream.write(
                    json.dumps(
                        dict(
                            time=time.time(),
                            running=[x[0]["id"] for x in running],
                            readings=[
                                line
                                for line in sample.stdout.splitlines()
                                if line.split(",")[0].strip() in ["4", "5", "6", "7"]
                            ],
                        )
                    )
                    + "\n"
                )
            last_sample = time.monotonic()
        if not running and failed:
            raise RuntimeError(f"Failed jobs: {sorted(failed)}; dependent jobs not started")
        if pending or running:
            time.sleep(3)
    dump(
        OUT / "complete.json",
        dict(completed_jobs=sorted(done), training_updates=384, evaluation_responses=19 * 2520, time=time.time()),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["run", "guard"])
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--owner", nargs=2)
    parser.add_argument("--tag")
    args = parser.parse_args()
    prepare_resources()
    if args.action == "guard":
        owner = [int(args.owner[0]), args.owner[1]]
        dump(args.directory / "guard-ready.json", dict(owner=owner, guard=identity(os.getpid())))
        while identity(owner[0]) == owner:
            if (args.directory / "recovered.json").exists():
                return
            time.sleep(3)
        resource.recover(args.directory, args.tag, OUT)
        return
    control = OUT / "control"
    control.mkdir(parents=True, exist_ok=True)
    if (control / "latest.json").exists():
        old = read(control / "latest.json")["owner"]
        if identity(old[0]) == old:
            raise RuntimeError("Round-two supervisor is already active")
    lease = (OLD_OUT / "control/gpu-lease.lock").open("a")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    directory = control / time.strftime("%Y%m%d-%H%M%S")
    directory.mkdir()
    owner = identity(os.getpid())
    tag = "round2-" + directory.name
    dump(control / "latest.json", dict(owner=owner, directory=str(directory), tag=tag))
    with (directory / "guard.log").open("a") as stream:
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                MODULE + ".round2_launch",
                "guard",
                "--directory",
                str(directory),
                "--owner",
                str(owner[0]),
                owner[1],
                "--tag",
                tag,
            ],
            cwd=REPO,
            stdout=stream,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    deadline = time.monotonic() + 30
    while not (directory / "guard-ready.json").exists():
        if time.monotonic() > deadline:
            raise TimeoutError("Recovery guard did not start")
        time.sleep(0.1)

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGHUP]:
        signal.signal(sig, interrupted)
    try:
        status = resource.occupy_status(OUT)
        if status.get("process_state") == "alive":
            assert status["status"]["cuda_visible_devices"] == "4,5,6,7"
            subprocess.run(
                ["bash", str(resource.OCCUPY), "stop", "--timeout", "30"], env=resource.occupy_env(OUT), check=True
            )
        memory = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True
        )
        assert all(
            int(line.split(",")[1]) < 1024 for line in memory.splitlines() if int(line.split(",")[0]) in [4, 5, 6, 7]
        ), memory
        run_queue(directory, tag)
        dump(directory / "complete.json", dict(time=time.time()))
    except BaseException as error:
        dump(directory / "error.json", dict(error=repr(error), time=time.time()))
        raise
    finally:
        for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGHUP]:
            signal.signal(sig, signal.SIG_IGN)
        resource.recover(directory, tag, OUT)


if __name__ == "__main__":
    main()
