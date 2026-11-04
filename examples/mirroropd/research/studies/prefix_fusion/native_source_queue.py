# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Prepare and serially run E78-E80 on available explicitly released GPUs.

Uses the existing broker for verified occupier handoff and resource_guard for
UUID/lock admission. A model owner or an unknown CUDA context is never stopped.
Failures stop the queue; there are no automatic experiment retries.
"""

import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import yaml

EXPERIMENTS = ("E78", "E79", "E80")
GPU_POOL = (5, 6, 0, 7, 1, 2, 3, 4)
WORLDS = (1, 2, 4, 8)
OWNER = "qwen-source-b128-r16-o100"


def package_versions():
    return {name: version(name) for name in ("torch", "vllm", "transformers", "ray", "pyarrow")}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment(repo, source, study, python, devices=(0, 1)):
    env = dict(os.environ)
    # Never inherit unrelated experiment overrides from the launching session.
    for key in list(env):
        if key.startswith(("PREFIX_", "FUSION_", "QWEN_")) or key in {
            "CONFIG_ONLY",
            "DRY_RUN",
            "TRAIN_BATCH_SIZE",
            "PPO_EPOCHS",
            "TOTAL_TRAINING_STEPS",
            "STUDENT_MODEL",
            "TEACHER_MODEL",
            "TRAIN_FILE",
            "RUN_DIR",
            "CHECKPOINT_DIR",
            "EXPERIMENT_NAME",
        }:
            env.pop(key)
    env.update(
        PYTHON_BIN=str(python),
        PYTHONPATH=str(source),
        MIRROROPD_WORKSPACE=str(repo.parent),
        MODEL_ROOT=str(repo.parent / "model"),
        DATA_ROOT=str(repo.parent / "data"),
        QWEN_EVAL_ROOT=str(study),
        STUDENT_GPUS=str(len(devices)),
        CUDA_VISIBLE_DEVICES=",".join(map(str, devices)),
        TRAIN_BATCH_SIZE="128",
        PPO_EPOCHS="16",
        TOTAL_TRAINING_STEPS="100",
        LEARNING_RATE="1e-6",
        SEED="42",
        TOTAL_EPOCHS="1",
        MAX_PROMPT_LENGTH="4096",
        MAX_RESPONSE_LENGTH="7168",
        MAX_VAL_RESPONSE_LENGTH="31744",
        FULL_KL_BACKEND="precompute_fused",
        FULL_KL_TERMINATION_TOKEN_IDS="[]",
        VERL_USE_UV="0",
        WANDB_MODE="offline",
        OMP_NUM_THREADS="1",
    )
    return env


def config_only(source, study, python, env, experiment, destination):
    local = dict(env, CONFIG_ONLY="1", RUN_DIR=str(destination), EXPERIMENT_NAME=destination.name)
    completed = subprocess.run(
        ["bash", str(source / "examples/mirroropd/run_math_opd_ablation_fsdp2.sh"), experiment],
        cwd=source,
        env=local,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"Configuration {experiment} failed: {completed.stderr[-4000:]}")
    config = yaml.safe_load(completed.stdout)
    loggers = config["trainer"]["logger"]
    if "file" not in loggers:
        loggers.append("file")
    destination.mkdir(parents=True, exist_ok=True)
    ray = config["ray_kwargs"]["ray_init"]
    ray["address"] = "local"
    ray["_temp_dir"] = "${oc.env:SMOKE_RAY_TMPDIR}"
    variables = ray.setdefault("runtime_env", {}).setdefault("env_vars", {})
    variables["VERL_ROLLOUT_IPC_NAMESPACE"] = "${oc.env:VERL_ROLLOUT_IPC_NAMESPACE}"
    variables["VERL_FILE_LOGGER_PATH"] = str(destination / "metrics.jsonl")
    variables["WANDB_DIR"] = str(destination)
    path = study / "configs" / f"{experiment}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    return config, path


def audit_config(config, *, gate=False):
    from verl.trainer.evaluation.config import EvaluationConfig
    from verl.trainer.evaluation.plan import compile_plan

    assert "file" in config["trainer"]["logger"]
    batch, reuse, outer = (8, 1, 2) if gate else (128, 16, 100)
    actor = config["actor_rollout_ref"]["actor"]
    assert config["data"]["train_batch_size"] == actor["ppo_mini_batch_size"] == batch
    assert actor["ppo_epochs"] == reuse and config["trainer"]["total_training_steps"] == outer
    assert config["trainer"]["n_gpus_per_node"] in WORLDS and config["trainer"]["nnodes"] == 1
    assert config["actor_rollout_ref"]["rollout"]["n"] == 1
    assert actor["optim"]["lr"] == 1e-6 and actor["loss_agg_mode"] == "token-mean"
    assert config["distillation"]["execution_mode"] == "shared_sync"
    assert config["distillation"]["distillation_loss"]["full_vocab"]["backend"] == "precompute_fused"
    assert config["distillation"]["distillation_loss"]["full_vocab"]["termination_token_ids"] == []
    assert config["actor_rollout_ref"]["rollout"]["temperature"] == 1.0
    assert config["actor_rollout_ref"]["rollout"]["top_p"] == 1.0
    assert config["data"]["shuffle"] is False and config["data"]["filter_overlong_prompts"] is False
    assert config["data"]["apply_chat_template_kwargs"]["enable_thinking"] is False
    assert config["rollout_policy"]["model"]["kind"] in {"current", "initial", "teacher"}
    values = dict(config["trainer"]["evaluation"])
    values.pop("_target_")
    spec = EvaluationConfig.parse(values)
    assert spec.source_enabled and not spec.history_enabled
    counts = {d.name: d.n for d in spec.datasets}
    expected = {"MATH-500": 4, "AMC23": 16, "AIME24": 32, "AIME25": 32, "AIME26": 32}
    assert counts == ({k: 1 for k in expected} if gate else expected)
    assert spec.source_count == (8 if gate else 128)
    assert sum(len(d.panel_ids) for d in spec.datasets) == (10 if gate else 128)
    boundaries = [
        n
        for n in range(outer + 1)
        if "accuracy" in compile_plan(spec, outer=n, total=outer, optimizer_steps=n * reuse).tasks
    ]
    assert boundaries == ([0, 1, 2] if gate else [0, 50, 100])
    assert all(
        "history" not in compile_plan(spec, outer=n, total=outer, optimizer_steps=n * reuse).tasks
        for n in range(outer + 1)
    )
    return dict(batch=batch, reuse=reuse, outer=outer, updates=outer * reuse, benchmarks=counts, bench_at=boundaries)


def prepare_gate(source, study, python, env, question_root):
    import pyarrow.parquet as pq

    gate = study / "gate"
    questions = gate / "questions"
    questions.mkdir(parents=True)
    for path in question_root.glob("*.parquet"):
        count = 16 if path.stem == "train" else 8 if path.stem == "source" else 2
        pq.write_table(pq.read_table(path).slice(0, count), questions / path.name)
    small = dict(
        env,
        QWEN_EVAL_ROOT=str(gate),
        TRAIN_BATCH_SIZE="8",
        PPO_EPOCHS="1",
        TOTAL_TRAINING_STEPS="2",
        MAX_PROMPT_LENGTH="4096",
        MAX_RESPONSE_LENGTH="64",
        MAX_VAL_RESPONSE_LENGTH="64",
        ROLLOUT_MAX_NUM_SEQS="2",
        PREFIX_SINGLE_KV_BYTES=str(2 * 2**30),
        FUSION_MAX_NUM_SEQS="2",
        FUSION_KV_BYTES=str(2 * 2**30),
    )
    audits = {}
    for experiment in EXPERIMENTS:
        config, path = config_only(source, gate, python, small, experiment, gate / "training" / experiment)
        spec = config["trainer"]["evaluation"]
        spec.update(
            source_count=8,
            history_size=8,
            generation_concurrency=2 * int(env["STUDENT_GPUS"]),
            generation_chunk_size=16,
        )
        for protocol in spec["protocols"].values():
            protocol.update(max_new_tokens=64, prompt_limit=4096)
        for dataset in spec["datasets"]:
            dataset.update(n=1, panel_ids=[f"{dataset['name']}:{i}" for i in range(2)])
        spec["schedules"] = {
            "source": {"every": 1},
            "history": {"final": False},
            "banks": {"every": 1},
            "bench": {"at": [1, 2]},
        }
        config["actor_rollout_ref"]["rollout"]["cudagraph_capture_sizes"] = [1, 2]
        path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
        audits[experiment] = audit_config(config, gate=True)
    for producer in ["initial", "teacher"]:
        path = gate / "configs" / f"{producer}.yaml"
        path.write_bytes((gate / "configs/E78.yaml").read_bytes())
    write(gate / "config-audit.json", audits)


def prepare(repo, source, study, python, *, worlds=(2, 1), prepared_questions=None):
    from examples.mirroropd.assets.paths import registered_model

    from ...core.artifacts import file_hash
    from .prepare_native import configs
    from .prepare_native import prepare as questions

    if study.exists():
        raise FileExistsError("Use a new study directory; do not overwrite a prepared experiment")
    study.mkdir(parents=True)
    env = environment(repo, source, study, python)
    if not worlds or len(set(worlds)) != len(worlds) or any(world not in WORLDS for world in worlds):
        raise ValueError("Declare distinct supported world sizes")
    if prepared_questions is None:
        questions(
            repo / "runs/qwen-prefix-six-20261031/prepared-03",
            study / "questions",
            protocol="source_b128_v1",
            tokenizer_path=registered_model("Qwen3-1.7B", root=repo.parent / "model"),
        )
    else:
        prepared_questions = Path(prepared_questions)
        receipt = read(prepared_questions / "complete.json")
        if (
            receipt.get("protocol") != "source_b128_v1"
            or receipt.get("training_rows") != 12800
            or receipt.get("source_count") != 128
        ):
            raise ValueError("Prepared questions do not match the declared source study")
        if not 0 < receipt.get("max_prompt_tokens", 0) <= 4096:
            raise ValueError("Prepared questions lack a valid tokenizer length audit")
        for name, expected in receipt["files"].items():
            if file_hash(prepared_questions / name) != expected:
                raise ValueError(f"Prepared question file changed: {name}")
        shutil.copytree(prepared_questions, study / "questions")
    for world in worlds:
        profile = study / "profiles" / f"world-{world}"
        profile.mkdir(parents=True)
        local = dict(env, STUDENT_GPUS=str(world), CUDA_VISIBLE_DEVICES=",".join(map(str, range(world))))
        audits = {}
        for experiment in EXPERIMENTS:
            config, _ = config_only(source, profile, python, local, experiment, study / "training" / experiment)
            audits[experiment] = audit_config(config)
        configs(profile / "configs/E78.yaml", profile / "configs/bootstrap")
        prepare_gate(source, profile, python, local, study / "questions")
        write(profile / "config-audit.json", audits)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    inputs = {str(p.relative_to(study)): file_hash(p) for p in study.rglob("*") if p.is_file()}
    write(
        study / "prepared.json",
        dict(
            source=str(source),
            revision=revision,
            packages=package_versions(),
            experiments=EXPERIMENTS,
            world_sizes=worlds,
            inputs=inputs,
            environment={
                k: v
                for k, v in env.items()
                if k in {"MODEL_ROOT", "DATA_ROOT", "PYTHON_BIN", "PYTHONPATH", "QWEN_EVAL_ROOT", "STUDENT_GPUS"}
            },
            scope="CPU preparation only; GPU gates must pass before scientific bootstrap and training",
        ),
    )


def eligible(broker, devices):
    registry = broker.registry()
    for gpu in devices:
        entry = registry["gpus"].get(str(gpu), {})
        if entry.get("desired") != "released" or entry.get("owner") is not None:
            return False, f"GPU {gpu} remains assigned to {entry.get('owner')}"
        run = entry.get("active_run")
        if run:
            # Prove that any process we may stop belongs to the occupancy service.
            broker.owned(run, gpu)
        elif not broker.gpu_lock_free(gpu) or broker.gpu_busy(gpu):
            return False, f"GPU {gpu} has an existing allocation"
    return True, None


def choose_devices(broker, worlds=(2, 1), pool=GPU_POOL):
    available = [gpu for gpu in pool if eligible(broker, (gpu,))[0]]
    for world in worlds:
        if world not in WORLDS:
            raise ValueError("Supported world sizes are 1/2/4/8")
        if len(available) >= world:
            return tuple(available[:world])
    return ()


class ReservationBusy(RuntimeError):
    pass


def reserve_all(broker, devices):
    ready, reason = eligible(broker, devices)
    if not ready:
        return False, reason
    reserved = []
    original = broker.registry

    def registry():
        value = original()
        # broker.reserve reads this while holding its GPU-control and registry locks.
        if reserving[0] is not None:
            entry = value["gpus"].get(str(reserving[0]), {})
            if entry.get("owner") not in (None, OWNER) or entry.get("desired") not in ("released", "reserved"):
                raise ReservationBusy("Reservation contention; preserve the existing owner")
        return value

    reserving = [None]
    broker.registry = registry
    try:
        for gpu in devices:
            reserving[0] = gpu
            try:
                broker.reserve(gpu, OWNER)
            except RuntimeError as exc:
                if "existing allocation" in str(exc) or "active model lease" in str(exc):
                    raise ReservationBusy(str(exc)) from exc
                raise
            reserved.append(gpu)
        return True, None
    except BaseException as exc:
        reserving[0] = None
        for gpu in reserved:
            broker.release(gpu, OWNER, "Incomplete GPU reservation; no model was started")
        if isinstance(exc, ReservationBusy):
            return False, str(exc)
        raise
    finally:
        broker.registry = original


def frozen(source, manifest):
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip() != manifest["revision"]:
        raise ValueError("Frozen source revision changed")
    if subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=source).returncode:
        raise ValueError("Frozen tracked source has local modifications")


def audit_training(study, experiment, *, gate=False):
    from verl.trainer.evaluation.artifacts import ArtifactStore

    base = study / "gate" if gate else study
    output = base / "training" / experiment
    store = ArtifactStore(output / "evaluation")
    manifest = store.manifest(complete=True)
    expected_outer = 2 if gate else 100
    shards = sorted(name for name in manifest["files"] if name.startswith("updates/actor-outer-"))
    assert shards == [f"updates/actor-outer-{n:03d}.parquet" for n in range(1, expected_outer + 1)]
    # The trainer publishes these exact optimizer events with content hashes.
    # Logging backends are optional observability, not the training authority.
    updates = [dict(row["metrics"]) for name in shards for row in store.read_rows(name)]
    expected = 2 if gate else 1600
    assert [r["step/optimizer"] for r in updates] == list(range(1, expected + 1))
    assert all(
        r["optim/actor/update_applied"] == 1
        and r["optim/actor/update_skipped"] == 0
        and r["optim/actor/loss_nonfinite"] == 0
        for r in updates
    )
    assert all(
        math.isfinite(r["optim/actor/loss"]) and math.isfinite(r["optim/actor/grad_norm_preclip"]) for r in updates
    )
    assert manifest["status"] == "complete"
    assert manifest["metadata"]["optimizer_steps"] == expected
    assert manifest["metadata"]["completed_outer"] == expected_outer
    export = read(output / "evaluation/final-model/export-audit/manifest.json")
    assert export["status"] == "complete"
    write(base / f"{experiment}-audit.json", dict(passed=True, optimizer_updates=expected, evaluation_complete=True))


def run_queue(repo, source, study, python, broker_path, worlds=(2, 1)):
    # A short procfs spelling keeps Unix socket names below Linux's limit while
    # all Ray files remain under the repository's runs directory.
    os.chdir(repo)
    spec = importlib.util.spec_from_file_location("existing_gpu_broker", broker_path)
    broker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(broker)
    manifest = read(study / "prepared.json")
    frozen(source, manifest)
    if package_versions() != manifest["packages"] or str(python) != manifest["environment"]["PYTHON_BIN"]:
        raise ValueError("Prepared runtime environment changed")
    if not set(worlds) <= set(manifest["world_sizes"]):
        raise ValueError("Requested GPU counts have no prepared configuration")
    for name, expected in manifest["inputs"].items():
        if digest(study / name) != expected:
            raise ValueError(f"Prepared input changed: {name}")
    env = environment(repo, source, study, python)
    stages = []
    state = dict(
        state="waiting_for_gpus",
        pid=os.getpid(),
        revision=manifest["revision"],
        preferred_world_sizes=worlds,
        gpus=[],
        stages=stages,
    )

    def publish(**values):
        state.update(values, updated_at=time.time())
        write(study / "queue-status.json", state)

    with (study / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (study / "queue-started.json").exists():
            raise ValueError("Queue already started; recovery must be explicit")
        write(study / "queue-started.json", dict(pid=os.getpid(), started_at=time.time()))
        while True:
            frozen(source, manifest)
            devices = choose_devices(broker, worlds)
            publish(wait_reason=None if devices else "No released GPU is available; model reservations are preserved")
            if devices:
                acquired, _ = reserve_all(broker, devices)
                if acquired:
                    break
            time.sleep(30)
        env = environment(repo, source, study, python, devices)
        profile = study / "profiles" / f"world-{len(devices)}"
        publish(state="running", wait_reason=None, gpus=devices, world_size=len(devices))
        guard = source / "examples/mirroropd/benchmarks/fusion/resource_guard.py"

        def execute(name, arguments):
            frozen(source, manifest)
            if package_versions() != manifest["packages"]:
                raise ValueError("Runtime environment changed between stages")
            if any(broker.registry()["gpus"][str(g)].get("owner") != OWNER for g in devices):
                raise RuntimeError("GPU ownership changed; refusing the next stage")
            ray_id = hashlib.sha256((str(study) + name).encode()).hexdigest()[:8]
            command = [
                str(python),
                str(guard),
                "run",
                "--repo",
                str(repo),
                "--gpus",
                ",".join(map(str, devices)),
                "--",
                "env",
                f"PYTHONPATH={source}",
                f"SMOKE_RAY_TMPDIR=/proc/{os.getpid()}/cwd/runs/r{ray_id}",
                str(python),
                *map(str, arguments),
            ]
            receipt = dict(name=name, command=command, state="starting", started_at=time.time())
            stages.append(receipt)
            logs = study / "logs"
            logs.mkdir(exist_ok=True)
            with (logs / f"{name}.log").open("x") as log:
                child = subprocess.Popen(
                    command, cwd=source, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT
                )
                monitor = subprocess.Popen(
                    [
                        str(python),
                        str(source / "examples/mirroropd/benchmarks/fusion/monitor.py"),
                        "--parent",
                        str(child.pid),
                        "--out",
                        str(logs / f"{name}-resources.jsonl"),
                    ],
                    cwd=source,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                receipt.update(state="running", pid=child.pid, monitor_pid=monitor.pid)
                publish(current=name)
                while child.poll() is None:
                    publish()
                    time.sleep(10)
                try:
                    monitor.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    monitor.terminate()
                    monitor.wait(timeout=10)
                receipt.update(
                    exit_code=child.returncode,
                    monitor_exit_code=monitor.returncode,
                    finished_at=time.time(),
                    state="complete" if child.returncode == 0 else "failed",
                )
                publish()
                if child.returncode:
                    raise RuntimeError(f"{name} exited {child.returncode}; see {log.name}")
                if monitor.returncode and not name.endswith("bootstrap-finalize"):
                    raise RuntimeError(f"{name} resource monitor failed; see {log.name}")

        try:
            for label, base in [("gate", profile / "gate"), ("formal", profile)]:
                bootstrap_configs = base / "configs" if label == "gate" else base / "configs/bootstrap"
                for producer in ["initial", "teacher"]:
                    execute(
                        f"{label}-bootstrap-{producer}",
                        [
                            "-m",
                            "verl.trainer.evaluation.bootstrap",
                            "--config",
                            bootstrap_configs / f"{producer}.yaml",
                            "--producer",
                            producer,
                        ],
                    )
                execute(
                    f"{label}-bootstrap-finalize",
                    [
                        "-m",
                        "verl.trainer.evaluation.bootstrap",
                        "--config",
                        base / "configs/E78.yaml",
                        "--producer",
                        "finalize",
                    ],
                )
                for experiment in EXPERIMENTS:
                    execute(
                        f"{label}-{experiment}",
                        ["-m", "verl.trainer.main_ppo", "--config-path", base / "configs", "--config-name", experiment],
                    )
                    audit_training(profile if label == "gate" else study, experiment, gate=label == "gate")
            publish(state="complete")
        except BaseException as exc:
            publish(state="failed", error=repr(exc))
            raise
        finally:
            # The broker refuses to start occupancy if CUDA cleanup or a resource lock remains unresolved.
            for gpu in devices:
                broker.release(gpu, OWNER, "Qwen source study queue finished")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run"])
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--broker", type=Path)
    parser.add_argument("--questions-from", type=Path, help="Reuse a verified CPU-prepared question set")
    parser.add_argument("--worlds", default="2,1", help="Preferred available GPU counts, e.g. 2,1 or 4,2,1")
    args = parser.parse_args()
    # Resolving a venv's python symlink selects the bare interpreter and loses its packages.
    values = [p.resolve() for p in [args.repo, args.source, args.study]] + [args.python.absolute()]
    if args.action == "prepare":
        prepare(*values, worlds=tuple(int(v) for v in args.worlds.split(",")), prepared_questions=args.questions_from)
    else:
        if args.broker is None:
            parser.error("--broker is required for resource ownership checks")
        try:
            run_queue(*values, args.broker.resolve(), tuple(int(v) for v in args.worlds.split(",")))
        except BaseException as exc:
            path = values[2] / "queue-status.json"
            state = read(path) if path.exists() else {"pid": os.getpid()}
            if state.get("pid") == os.getpid():
                state.update(state="failed", error=repr(exc), updated_at=time.time())
                write(path, state)
            raise


if __name__ == "__main__":
    main()
