# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""One dependency queue for every study; completion identities and bounded retries."""

import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
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


@dataclass(frozen=True)
class ExecutionContext:
    name: str
    output: str
    gpus: tuple
    identity: str

    @property
    def directory(self):
        return Path(self.output)


def jobs(study, phase):
    if phase == "stage-attribution":
        from examples.mirroropd.research.studies.fixed_transfer.stage_preferences import inputs

        identity, _ = inputs(study)
        return [
            dict(
                name="stage-preferences",
                args=[],
                module="examples.mirroropd.research.studies.fixed_transfer.stage_preferences",
                marker=str(study.directory / "analysis/stage-preferences/complete.json"),
                fingerprint=fingerprint(identity),
                dependencies=[],
                retries=0,
            )
        ]
    if phase == "long-panel":
        from examples.mirroropd.research.studies.fixed_transfer.long_panel import jobs as panel_jobs

        return panel_jobs(study)
    if phase == "endpoints":
        from examples.mirroropd.research.studies.fixed_transfer.training import endpoint_jobs

        result = endpoint_jobs(study)
        for job in result:
            job.pop("producer", None)
            if not ready(job):
                raise ValueError(f"Endpoint-only scheduling requires saved weights: {job['name']}")
        return result
    if phase == "probes":
        from examples.mirroropd.research.studies.fixed_transfer.mode_probe import probe_inputs

        return [
            dict(
                name="probe-" + name,
                args=["--model", name],
                module="examples.mirroropd.research.studies.fixed_transfer.mode_probe",
                marker=str(study.directory / "mode-probes" / name / "complete.json"),
                fingerprint=fingerprint(probe_inputs(study, name)),
                dependencies=[],
                retries=0,
            )
            for name in study.models
            if name != study.student or (getattr(study, "prompt_policy", None) or {}).get("template_model")
        ]
    if phase == "resume-check":
        result = []
        for arm in study.arms:
            root = study.directory / "arms" / arm.name / "resume"
            saved = read(root / "complete.json")
            expected = dict(step=saved["step"], signature=saved["signature"], weight_identity=saved["weight_identity"])
            result.append(
                dict(
                    name="resume-" + arm.name,
                    args=["--arm", arm.name],
                    module="examples.mirroropd.research.studies.fixed_transfer.resume_probe",
                    marker=str(root / "gpu-verified.json"),
                    fingerprint=fingerprint(expected),
                    dependencies=[],
                    retries=0,
                    priority=0,
                )
            )
        return result
    prepared = read(study.directory / "prepared/complete.json")
    result = []
    if phase not in ["banks", "baselines", "full", "training", "fit", "targets"]:
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
    if phase == "banks":
        return result
    for name in [] if phase == "fit" else study.models:
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
    if phase in ["full", "training", "fit", "targets"]:
        if phase == "fit" and (study.continuation or {}).get("accuracy_gate", True):
            raise ValueError("Fit-only scheduling requires explicit accuracy_gate=false")
        if phase == "training":
            for job in result:
                job["external"] = True
        from examples.mirroropd.research.studies.fixed_transfer.training import training_jobs

        result.extend(training_jobs(study, result))
    if phase == "targets":
        result = [job for job in result if job["args"][0] in ["bank", "cache"]]
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
    if any(not Path(value).exists() for value in job.get("required_files", [])):
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


def queue(study, plan, phase, directory, tag, *, custom_jobs=None):
    all_jobs = jobs(study, phase) if custom_jobs is None else custom_jobs
    validate_graph(all_jobs)
    write(directory / "jobs.json", all_jobs)
    done = {j["name"] for j in all_jobs if finished(j)}
    external = [j for j in all_jobs if j.get("external") and j["name"] not in done]
    pending = [j for j in all_jobs if not j.get("external") and j["name"] not in done]
    attempts = {j["name"]: 0 for j in all_jobs}
    running = []
    failed = []
    while pending or running or external:
        for job in list(external):
            if finished(job):
                done.add(job["name"])
                external.remove(job)
        busy = {gpu for x in running for gpu in x[2]}
        for job in list(pending):
            free = [g for g in study.gpus if g not in busy]
            preferred = job.get("preferred_gpus", study.gpus)
            if not set(preferred) <= set(study.gpus):
                raise ValueError(f"Job requests GPUs outside the lease: {job['name']}")
            free = [g for g in preferred if g in free]
            width = job.get("width", 1)
            if not 1 <= width <= len(study.gpus):
                raise ValueError(f"Invalid GPU width: {job['name']}")
            if len(free) < width:
                continue
            if not set(job["dependencies"]) <= done:
                continue
            if not ready(job):
                if job.get("producer") in done:
                    raise RuntimeError(f"Completed producer did not publish {job['ready_file']}")
                continue
            assigned = tuple(free[:width])
            attempts[job["name"]] += 1
            attempt = directory / job["name"] / str(attempts[job["name"]])
            attempt.mkdir(parents=True)
            source = Path(job.get("source", directory / "source"))
            env = dict(
                environment(source, workspace()),
                CUDA_VISIBLE_DEVICES=",".join(map(str, assigned)),
                MIRROROPD_ALLOWED_GPUS=",".join(map(str, study.gpus)),
                OMP_NUM_THREADS="4",
                OPENBLAS_NUM_THREADS="4",
                HF_HUB_OFFLINE="1",
                TOKENIZERS_PARALLELISM="false",
                VLLM_WORKER_MULTIPROC_METHOD="spawn",
            )
            env[TAG] = tag
            command = [
                sys.executable,
                "-m",
                job.get("module", MODULE),
                *job["args"],
                "--plan",
                str(job.get("plan", plan)),
            ]
            stream = (attempt / "worker.log").open("w")
            child = subprocess.Popen(
                command,
                cwd=source,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            write(
                attempt / "launch.json",
                dict(identity=identity(child.pid), gpu=assigned[0] if width == 1 else list(assigned), command=command),
            )
            running.append((job, child, assigned, stream, attempt))
            pending.remove(job)
            busy.update(assigned)
        completed_any = False
        for item in list(running):
            job, child, gpu, stream, attempt = item
            code = child.poll()
            if code is None:
                continue
            completed_any = True
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
                external=[j["name"] for j in external],
                done=sorted(done),
                failed=failed,
                time=time.time(),
            ),
        )
        if not running and failed:
            raise RuntimeError(f"Failed jobs: {failed}")
        if not running and pending and not external and not completed_any:
            raise RuntimeError(f"No runnable jobs: {[j['name'] for j in pending]}")
        if pending or running or external:
            time.sleep(3)
    write(directory / "jobs-complete.json", dict(done=sorted(done), phase=phase))


def queue_with_banks(study, plan, phase, directory, tag):
    """Materialize banks before constructing content-bound teacher/training jobs."""
    if phase in {"full", "fit", "targets"}:
        bootstrap_phase = "baselines" if phase == "full" else "banks"
        bank_jobs = jobs(study, "banks")
        if not all(finished(job) for job in bank_jobs):
            queue(study, plan, bootstrap_phase, directory, tag)
    queue(study, plan, phase, directory, tag)


def run(plan, phase, *, wait_resource=False, prepared_study=None, custom_job_builder=None):
    plan = Path(plan).resolve()
    if (prepared_study is None) != (custom_job_builder is None):
        raise ValueError("External adapters must supply both a prepared study and a job builder")
    study = prepared_study or Study.load(plan)
    if prepared_study is None:
        from examples.mirroropd.research.studies.fixed_transfer.data import prepare

        prepare(study)
        if phase in {"full", "training", "fit"}:
            from examples.mirroropd.research.studies.fixed_transfer.models import audit
            from examples.mirroropd.research.studies.fixed_transfer.training import validation_teacher
            from examples.mirroropd.research.studies.fixed_transfer.validation import inherit_resume_profile

            # Reject incompatible teacher-copy references before waiting for GPUs or generating banks.
            audit(study)
            near = validation_teacher(study)
            inherit_resume_profile(study, near)
    resource = Resource()
    if set(study.gpus) != set(resource.config["gpus"]):
        raise ValueError("Study devices do not match the runtime resource allocation")
    directory = study.directory / "control" / f"{time.time_ns()}-{phase}"
    directory.mkdir(parents=True)
    frozen_plan = directory / "plan.json"
    shutil.copy2(plan, frozen_plan)
    plan = frozen_plan
    freeze(REPO, directory / "source")
    write(
        directory / "resource-wait.json",
        dict(identity=identity(os.getpid()), resource=str(resource.path), waiting=wait_resource),
    )
    lease = resource.acquire(wait=wait_resource)
    write(directory / "resource-acquired.json", dict(identity=identity(os.getpid()), time=time.time()))
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
        if custom_job_builder is None:
            queue_with_banks(study, plan, phase, directory, tag)
        else:
            queue(study, plan, phase, directory, tag, custom_jobs=custom_job_builder(plan))
    except BaseException as error:
        write(directory / "error.json", dict(error=repr(error), time=time.time()))
        raise
    finally:
        for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGHUP]:
            signal.signal(sig, signal.SIG_IGN)
        resource.recover(directory, tag)
        lease.close()
    write(study.directory / f"{phase}-complete.json", dict(control=str(directory), time=time.time()))


def status(study):
    """Read-only snapshot; queue files alone never establish process liveness."""
    controllers = []
    for directory in sorted((study.directory / "control").glob("*")):
        if not directory.is_dir():
            continue
        owner_file = directory / "owner.json"
        waiting_file = directory / "resource-wait.json"
        record = read(owner_file) if owner_file.exists() else read(waiting_file) if waiting_file.exists() else {}
        expected = record.get("identity")
        alive = bool(expected and identity(expected[0]) == expected)
        error = directory / "error.json"
        finished_jobs = (directory / "jobs-complete.json").exists()
        if (directory / "superseded.json").exists():
            state = "retiring" if alive else "superseded"
        elif error.exists():
            state = "recovering_after_error" if alive else "failed"
        elif finished_jobs:
            state = (
                "recovering"
                if alive
                else "completed"
                if (directory / "recovered.json").exists()
                else "jobs_complete_recovery_unverified"
            )
        elif not alive:
            state = "stopped" if expected else "unowned"
        elif owner_file.exists():
            state = "running"
        else:
            state = "waiting_resource"
        queue_file = directory / "queue.json"
        controllers.append(
            dict(
                directory=str(directory),
                identity=expected,
                alive=alive,
                state=state,
                queue=read(queue_file) if queue_file.exists() else None,
                error=read(error) if error.exists() else None,
            )
        )
    locations = {path.name: path for path in (study.directory / "evaluations").glob("*")}
    expected = {}
    if (study.directory / "prepared/complete.json").exists():
        for name in study.models:
            expected[name] = fingerprint(evaluation_inputs(study, name))
            locations[name] = study.cache_directory / "evaluations" / expected[name]
    evaluations = []
    for label, directory in sorted(locations.items()):
        progress = directory / "progress.json"
        if directory.is_dir() and progress.exists():
            job = dict(name=label, marker=str(directory / "complete.json"))
            if label in expected:
                job["fingerprint"] = expected[label]
            evaluations.append(
                dict(label=label, complete=finished(job), progress=read(progress), source=str(directory))
            )
    return dict(study=study.name, observed_at=time.time(), controllers=controllers, evaluations=evaluations)
