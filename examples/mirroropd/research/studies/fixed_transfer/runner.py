# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""One dependency queue for every study; completion identities and bounded retries."""

import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from examples.mirroropd.research.core.io import dump as write
from examples.mirroropd.research.core.io import read
from examples.mirroropd.research.core.paths import REPO, workspace
from examples.mirroropd.research.runtime.processes import identity
from examples.mirroropd.research.runtime.resources import TAG, Resource
from examples.mirroropd.research.runtime.source import environment, freeze
from examples.mirroropd.research.studies.fixed_transfer.data import bank_directory
from examples.mirroropd.research.studies.fixed_transfer.evaluation import evaluation_inputs
from examples.mirroropd.research.studies.fixed_transfer.spec import Study, fingerprint

MODULE = "examples.mirroropd.research.studies.fixed_transfer"


def jobs(study, phase):
    prepared = read(study.directory / "prepared/complete.json")
    result = []
    if phase not in ["baselines", "full"]:
        raise ValueError(phase)
    for shard in range(4):
        inputs = dict(bank_key=prepared["bank_key"], shard=shard, shards=4)
        result.append(
            dict(
                name=f"bank-{shard}",
                args=["bank", "--shard", str(shard)],
                marker=str(bank_directory(study) / f"shard-{shard}/complete.json"),
                fingerprint=fingerprint(inputs),
                dependencies=[],
                retries=0,
            )
        )
    for name in study.models:
        inputs = evaluation_inputs(study, name)
        result.append(
            dict(
                name="baseline-" + name,
                args=["baseline", "--model", name],
                marker=str(study.cache_directory / "evaluations" / fingerprint(inputs) / "complete.json"),
                fingerprint=fingerprint(inputs),
                dependencies=[],
                retries=2,
            )
        )
    if phase == "full":
        from examples.mirroropd.research.studies.fixed_transfer.training import training_jobs

        result.extend(training_jobs(study, result))
    for job in result:
        job.setdefault("priority", 0 if job["name"].startswith("bank-") else 10)
    return sorted(result, key=lambda job: job["priority"])


def check_inputs(path, expected):
    actual = read(path).get("inputs", {})
    if any(key not in actual or fingerprint(actual[key]) != fingerprint(value) for key, value in expected.items()):
        raise ValueError(f"Artifact inputs changed: {path}")


def ready(job):
    if not job.get("ready_file"):
        return True
    path = Path(job["ready_file"])
    if not path.exists():
        return False
    check_inputs(path, job.get("ready_inputs", {}))
    return True


def finished(job):
    path = Path(job["marker"])
    if not path.exists():
        return False
    if job.get("fingerprint") and read(path)["fingerprint"] != job["fingerprint"]:
        raise ValueError(f"Completion identity mismatch: {job['name']}")
    check_inputs(path, job.get("expected_inputs", {}))
    if job.get("completion_weight_from_ready"):
        if not ready(job):
            raise ValueError(f"Missing producer receipt: {job['name']}")
        if read(path)["inputs"]["weights"] != read(job["ready_file"])["weight_identity"]:
            raise ValueError(f"Evaluation belongs to a different checkpoint: {job['name']}")
    return True


def validate_graph(all_jobs):
    index = {job["name"]: job for job in all_jobs}
    if len(index) != len(all_jobs):
        raise ValueError("Duplicate job identity")
    visited, active = set(), set()

    def visit(name):
        if name not in index:
            raise ValueError(f"Unknown dependency: {name}")
        if name in active:
            raise ValueError(f"Dependency cycle: {name}")
        if name in visited:
            return
        active.add(name)
        job = index[name]
        for parent in job["dependencies"] + ([job["producer"]] if job.get("producer") else []):
            visit(parent)
        active.remove(name)
        visited.add(name)

    for name in index:
        visit(name)


def queue(study, plan, phase, directory, tag):
    all_jobs = jobs(study, phase)
    validate_graph(all_jobs)
    write(directory / "jobs.json", all_jobs)
    done = {j["name"] for j in all_jobs if finished(j)}
    pending = [j for j in all_jobs if j["name"] not in done]
    attempts = {j["name"]: 0 for j in all_jobs}
    running = []
    failed = []
    while pending or running:
        busy = {x[2] for x in running}
        for job in list(pending):
            free = [g for g in study.gpus if g not in busy]
            if not free:
                break
            if not set(job["dependencies"]) <= done:
                continue
            if not ready(job):
                if job.get("producer") in done:
                    raise RuntimeError(f"Completed producer did not publish {job['ready_file']}")
                continue
            gpu = free[0]
            attempts[job["name"]] += 1
            attempt = directory / job["name"] / str(attempts[job["name"]])
            attempt.mkdir(parents=True)
            env = dict(
                environment(directory / "source", workspace()),
                CUDA_VISIBLE_DEVICES=str(gpu),
                MIRROROPD_ALLOWED_GPUS=",".join(map(str, study.gpus)),
                OMP_NUM_THREADS="4",
                OPENBLAS_NUM_THREADS="4",
                HF_HUB_OFFLINE="1",
                TOKENIZERS_PARALLELISM="false",
                VLLM_WORKER_MULTIPROC_METHOD="spawn",
            )
            env[TAG] = tag
            command = [sys.executable, "-m", MODULE, *job["args"], "--plan", str(plan)]
            stream = (attempt / "worker.log").open("w")
            child = subprocess.Popen(
                command,
                cwd=directory / "source",
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            write(attempt / "launch.json", dict(identity=identity(child.pid), gpu=gpu, command=command))
            running.append((job, child, gpu, stream, attempt))
            pending.remove(job)
            busy.add(gpu)
        for item in list(running):
            job, child, gpu, stream, attempt = item
            code = child.poll()
            if code is None:
                continue
            stream.close()
            running.remove(item)
            success = code == 0 and finished(job)
            write(attempt / "exit.json", dict(code=code, success=success, time=time.time()))
            if success:
                done.add(job["name"])
            elif attempts[job["name"]] <= job["retries"]:
                pending.append(job)
            else:
                failed.append(job["name"])
        write(
            directory / "queue.json",
            dict(
                running=[x[0]["name"] for x in running],
                pending=[j["name"] for j in pending],
                done=sorted(done),
                failed=failed,
                time=time.time(),
            ),
        )
        if not running and failed:
            raise RuntimeError(f"Failed jobs: {failed}")
        if not running and pending:
            raise RuntimeError(f"No runnable jobs: {[j['name'] for j in pending]}")
        if pending or running:
            time.sleep(3)
    write(directory / "jobs-complete.json", dict(done=sorted(done), phase=phase))


def run(plan, phase):
    plan = Path(plan).resolve()
    study = Study.load(plan)
    from examples.mirroropd.research.studies.fixed_transfer.data import prepare

    prepare(study)
    resource = Resource()
    if set(study.gpus) != set(resource.config["gpus"]):
        raise ValueError("Study devices do not match the runtime resource allocation")
    directory = study.directory / "control" / f"{time.time_ns()}-{phase}"
    directory.mkdir(parents=True)
    frozen_plan = directory / "plan.json"
    shutil.copy2(plan, frozen_plan)
    plan = frozen_plan
    freeze(REPO, directory / "source")
    lease = resource.acquire()
    write(directory / "study.json", dict(study=asdict(study), identity=study.identity))
    tag = study.name + "-" + directory.name
    write(directory / "owner.json", dict(identity=identity(os.getpid()), tag=tag, plan=str(plan), phase=phase))
    write(study.directory / "control/latest.json", dict(directory=str(directory), owner=identity(os.getpid()), tag=tag))
    try:
        with (directory / "guard.log").open("w") as log:
            subprocess.Popen(
                [sys.executable, "-m", MODULE, "guard", "--directory", str(directory)],
                cwd=directory / "source",
                env=environment(directory / "source", workspace()),
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        deadline = time.monotonic() + 30
        while not (directory / "guard-ready.json").exists():
            if time.monotonic() > deadline:
                raise TimeoutError("Recovery guard startup")
            time.sleep(0.1)

        def interrupted(signum, frame):
            raise SystemExit(128 + signum)

        for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGHUP]:
            signal.signal(sig, interrupted)
        resource.prepare()
        if phase == "full":
            bootstrap = jobs(study, "baselines")
            if not all(finished(job) for job in bootstrap if job["name"].startswith("bank-")):
                queue(study, plan, "baselines", directory, tag)
        queue(study, plan, phase, directory, tag)
    except BaseException as error:
        write(directory / "error.json", dict(error=repr(error), time=time.time()))
        raise
    finally:
        for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGHUP]:
            signal.signal(sig, signal.SIG_IGN)
        resource.recover(directory, tag)
        lease.close()
    write(study.directory / f"{phase}-complete.json", dict(control=str(directory), time=time.time()))
