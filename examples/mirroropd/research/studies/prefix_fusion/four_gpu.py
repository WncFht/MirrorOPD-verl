# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Frozen four-GPU Qwen source comparison with native RKL and retained snapshots."""

import argparse
import collections
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
import yaml

from verl.trainer.evaluation.artifacts import GENERATION, ArtifactStore, file_sha256, identity
from verl.trainer.evaluation.identity import model_identity

from .native_source_queue import config_only, environment, package_versions

BOUNDARIES = (0, 1, 2, 3, 4, 5, 8, 10, 15, 20, 30, 40)
EXPERIMENTS = ("E64", "E63", "E65")
DEVICES = (4, 5, 6, 7)
OWNER = "qwen-three-source-four-gpu-20261111"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def banks(repo, study, student, teacher):
    """Validate old Qwen tokens/assets, then publish explicitly imported short views."""
    from transformers import AutoTokenizer

    from .assets import verify_model_assets

    legacy = repo / "runs/qwen-prefix-six-20261031"
    assets = read(legacy / "assets/complete.json")
    tokenizer = AutoTokenizer.from_pretrained(student, local_files_only=True)
    primary = {r["id"]: r for r in read(legacy / "prepared-03/heldout-questions.json") if r["primary"]}
    target_identity = model_identity(teacher)
    for source, model in (("S0", student), ("Q", teacher)):
        verify_model_assets(assets["models"][source])
        if Path(assets["models"][source]["source"]).resolve() != Path(model).resolve():
            raise ValueError("Bank assets and active model paths differ")
        directory = legacy / "banks" / source
        receipt = read(directory / "complete.json")
        if file_sha256(directory / "bank.json") != receipt["bank_sha256"]:
            raise ValueError("Legacy fixed bank changed")
        original = read(directory / "bank.json")
        if {r["id"] for r in original} != set(primary) or len(original) != 128:
            raise ValueError("Three-domain bank question coverage differs")
        rows = []
        for row in original:
            question = primary[row["id"]]
            expected = tokenizer.apply_chat_template(
                question["messages"],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=False,
            )
            if expected != row["prompt_ids"] or question["messages"] != row["messages"]:
                raise ValueError("Frozen bank Qwen prompt/template identity differs")
            response = row["response_ids"]
            if not response or len(response) > (7168 if row["domain"] == "math" else 1024):
                raise ValueError("Invalid legacy response length")
            rows.append(
                dict(
                    id=f"{source}:{row['id']}",
                    source_id=str(row["id"]),
                    outer=0,
                    prompt_ids=row["prompt_ids"],
                    response_ids=response,
                    response_mask=[True] * len(response),
                    bank="heldout",
                    dataset=row["domain"],
                    domain=row["domain"],
                    sample_index=0,
                    protocol_id=identity(receipt["inputs"]),
                    seed=row["seed"],
                    producer=source,
                    prompt_tokens=len(expected),
                    response_tokens=len(response),
                    valid_tokens=len(response),
                    finish_reason=row["finish_reason"],
                )
            )
        metadata = dict(
            model_identity=model_identity(model),
            teacher_identity=target_identity,
            provenance="Imported actual frozen Qwen trajectories; not regenerated native samples",
            original_receipt=str(directory / "complete.json"),
            original_receipt_sha256=file_sha256(directory / "complete.json"),
            original_bank_sha256=receipt["bank_sha256"],
            sampling=receipt["inputs"]["parameters"],
        )
        for mode in ("formal", "gate"):
            chosen = rows
            if mode == "gate":
                chosen = [
                    r for domain in ("math", "code", "if") for r in [x for x in rows if x["domain"] == domain][:2]
                ]
            store = ArtifactStore.create(study / mode / "banks" / source, kind="imported_fixed_bank", metadata=metadata)
            store.write_rows("archive/full.parquet", chosen, schema=GENERATION)
            short = [
                dict(
                    r,
                    response_ids=r["response_ids"][:1024],
                    response_mask=r["response_mask"][:1024],
                    response_tokens=min(r["response_tokens"], 1024),
                    valid_tokens=min(r["valid_tokens"], 1024),
                )
                for r in chosen
            ]
            store.write_rows("generations/short.parquet", short, schema=GENERATION)
            store.finish(metadata={"counts": dict(collections.Counter(r["domain"] for r in chosen))})


def audit_config(config, *, gate):
    actor = config["actor_rollout_ref"]["actor"]
    loss = config["distillation"]["distillation_loss"]["full_vocab"]
    assert loss["backend"] == "precompute_fused" and loss["termination_token_ids"] == []
    assert config["data"]["train_batch_size"] == actor["ppo_mini_batch_size"] == (8 if gate else 64)
    assert actor["ppo_epochs"] == (1 if gate else 16)
    assert config["trainer"]["total_training_steps"] == (2 if gate else 40)
    assert config["trainer"]["n_gpus_per_node"] == 4
    assert config["actor_rollout_ref"]["rollout"]["response_length"] == (32 if gate else 7168)
    assert actor["optim"]["lr"] == 1e-6 and actor["loss_agg_mode"] == "token-mean"
    assert actor["fsdp_config"]["model_dtype"] == "fp32"
    evaluation = config["trainer"]["evaluation"]
    assert not evaluation["source_enabled"] and not evaluation["history_enabled"]
    assert set(evaluation["fixed_banks"]) == {"public_initial", "public_teacher"}
    assert config["trainer"]["save_freq"] == -1
    assert evaluation["export_outers"] == ([1, 2] if gate else list(BOUNDARIES[1:]))
    from verl.trainer.evaluation.config import EvaluationConfig
    from verl.trainer.evaluation.plan import compile_plan

    options = dict(evaluation)
    options.pop("_target_")
    spec = EvaluationConfig.parse(options)
    for outer in range(config["trainer"]["total_training_steps"] + 1):
        plan = compile_plan(
            spec,
            outer=outer,
            total=config["trainer"]["total_training_steps"],
            optimizer_steps=outer * actor["ppo_epochs"],
        )
        if outer:
            assert ("public_initial" in plan.tasks) == (outer in ([1, 2] if gate else BOUNDARIES))
            assert not ({"history", "source_heldout"} & set(plan.tasks))


def prepare(repo, source, study, python):
    from .prepare_native import prepare as prepare_questions

    study.mkdir(parents=True, exist_ok=False)
    base = environment(repo, source, study, python, DEVICES)
    base.update(TRAIN_BATCH_SIZE="64", TOTAL_TRAINING_STEPS="40", PPO_EPOCHS="16", PREFIX_RAY_CPUS="48")
    # Canonical registered storage gives active model/asset identities a stable binding.
    base["MODEL_ROOT"] = str(repo.parent / "model")
    prepare_questions(repo / "runs/qwen-prefix-six-20261031/prepared-03", study / "questions")
    paths = []
    formal_configs = []
    for experiment in EXPERIMENTS:
        config, _ = config_only(
            source, study / "formal", python, base, experiment, study / "formal/training" / experiment
        )
        formal_configs.append((experiment, config))
    first = formal_configs[0][1]
    student = first["actor_rollout_ref"]["model"]["path"]
    teacher = first["distillation"]["teacher_models"]["teacher_model"]["model_path"]
    banks(repo, study, student, teacher)
    for mode in ("gate", "formal"):
        gate = mode == "gate"
        root = study / mode
        qroot = root / "questions"
        qroot.mkdir(exist_ok=True)
        for name in ("MATH-500", "AMC23", "AIME24", "AIME25", "AIME26"):
            table = pq.read_table(study / "questions" / f"{name}.parquet")
            if gate:
                table = table.slice(0, 1)
            pq.write_table(table, qroot / f"{name}.parquet")
        for experiment, original in formal_configs:
            config = copy.deepcopy(original)
            output = root / "training" / experiment
            output.mkdir(parents=True, exist_ok=True)
            evaluation = config["trainer"]["evaluation"]
            config["actor_rollout_ref"]["rollout"]["max_model_len"] = 32768
            # Legacy trainer validation is disabled (test_freq=-1). Accuracy
            # uses the independent evaluation protocols and their own engines.
            config["actor_rollout_ref"]["rollout"]["val_kwargs"]["max_tokens"] = None
            evaluation.update(
                output=str(output / "evaluation"),
                bootstrap=str(root / "bootstrap"),
                source_enabled=False,
                history_enabled=False,
                fixed_banks={"public_initial": str(root / "banks/S0"), "public_teacher": str(root / "banks/Q")},
                export_outers=[1, 2] if gate else list(BOUNDARIES[1:]),
            )
            nodes = [1, 2] if gate else list(BOUNDARIES[1:])
            evaluation["schedules"] = {
                "source": {"at": [], "final": False},
                "history": {"at": [], "final": False},
                "banks": {"at": nodes},
                "bench": {"at": [2] if gate else [20, 40]},
            }
            for d in evaluation["datasets"]:
                d.update(path=str(qroot / f"{d['name']}.parquet"), n=1 if gate else 4, panel_ids=[])
            if gate:
                for protocol in evaluation["protocols"].values():
                    protocol.update(max_new_tokens=32, prompt_limit=4096)
                config["data"].update(train_batch_size=8, max_response_length=32)
                config["actor_rollout_ref"]["actor"].update(ppo_mini_batch_size=8, ppo_epochs=1)
                config["actor_rollout_ref"]["rollout"].update(response_length=32)
                config["trainer"].update(total_training_steps=2)
            # The resolved sampler and training split are identical across all arms.
            config["data"]["train_files"] = str(study / "questions/train.parquet")
            config["trainer"].update(
                default_local_dir=str(output / "runtime"), experiment_name=f"qwen-three-{mode}-{experiment}"
            )
            variables = config["ray_kwargs"]["ray_init"]["runtime_env"]["env_vars"]
            variables["VERL_FILE_LOGGER_PATH"] = str(output / "metrics.jsonl")
            variables["WANDB_DIR"] = str(output)
            audit_config(config, gate=gate)
            destination = root / "configs" / f"{experiment}.yaml"
            destination.parent.mkdir(exist_ok=True)
            destination.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
            paths.append(destination)
        for producer in ("initial", "teacher"):
            config = yaml.safe_load((root / "configs/E64.yaml").read_text())
            # Four TP1 bootstrap engines, bounded independently of training actor.
            config["actor_rollout_ref"]["rollout"]["max_num_seqs"] = 8 if producer == "teacher" else 16
            config["actor_rollout_ref"]["rollout"]["engine_kwargs"]["vllm"]["kv_cache_memory_bytes"] = (
                40 if producer == "teacher" else 64
            ) * 2**30
            destination = root / "configs" / f"{producer}.yaml"
            destination.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
            paths.append(destination)
    # Immutable source snapshot is checked between every GPU stage.
    code = {
        str(p.relative_to(source)): file_sha256(p)
        for top in ("verl", "examples/mirroropd")
        for p in (source / top).rglob("*.py")
    }
    write(
        study / "prepared.json",
        dict(
            source=str(source),
            packages=package_versions(),
            code=code,
            configs={str(p): file_sha256(p) for p in paths},
            devices=DEVICES,
            protocol="qwen-three-source-math64-code32-if32-v1",
            environment=base,
        ),
    )


def audit_run(root, experiment, *, gate):
    store = ArtifactStore(root / "training" / experiment / "evaluation")
    manifest = store.manifest(complete=True)
    total, batch, reuse = (2, 8, 1) if gate else (40, 64, 16)
    for outer in range(1, total + 1):
        rows = store.read_rows(f"trajectories/outer-{outer:03d}.parquet")
        if len(rows) != batch:
            raise ValueError("Training trajectory count differs")
    for outer in [0, 1, 2] if gate else BOUNDARIES:
        event = ArtifactStore(store.root / f"events/outer-{outer:03d}").manifest(complete=True)
        if not event["metadata"]["coverage"]["scientific_complete"]:
            raise ValueError("Held-out RKL coverage incomplete")
        if event["metadata"]["optimizer_steps"] != outer * reuse:
            raise ValueError("Actual optimizer update count differs")
    for outer in [1, 2] if gate else BOUNDARIES[1:]:
        checkpoint = store.root / ("final-model" if outer == total else f"snapshots/outer-{outer:03d}")
        receipt = ArtifactStore(checkpoint / "export-audit").manifest(complete=True)
        if receipt["metadata"]["native_reload_rkl"] != "passed":
            raise ValueError("Snapshot reload parity failed")
        if set(r["dtype"] for r in ArtifactStore(checkpoint / "export-audit").read_rows("tensors.parquet")) != {"F32"}:
            raise ValueError("Snapshot is not FP32")
    accuracy = ArtifactStore(store.root / f"events/outer-{total:03d}/accuracy").manifest(complete=True)
    if not accuracy["metadata"]["metrics"]:
        raise ValueError("Final accuracy missing")
    write(
        root / f"{experiment}-audit.json",
        dict(
            passed=True,
            outers=total,
            updates=total * reuse,
            trajectories=total * batch,
            evaluation_artifact=manifest["artifact_id"],
        ),
    )


def run(repo, source, study, python, broker_path):
    spec = importlib.util.spec_from_file_location("four_gpu_broker", broker_path)
    broker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(broker)
    from . import native_source_queue as queue

    queue.OWNER = OWNER
    manifest = read(study / "prepared.json")
    if package_versions() != manifest["packages"]:
        raise ValueError("Runtime packages changed")
    for path, digest in manifest["configs"].items():
        if file_sha256(path) != digest:
            raise ValueError("Prepared config changed")
    state = dict(pid=os.getpid(), state="waiting_for_gpus", devices=DEVICES, stages=[])
    acquired = False

    def publish(**values):
        state.update(values, updated_at=time.time())
        write(study / "status.json", state)

    def frozen():
        for name, digest in manifest["code"].items():
            if file_sha256(source / name) != digest:
                raise ValueError(f"Frozen source changed: {name}")

    def execute(label, args):
        frozen()
        if any(broker.registry()["gpus"][str(g)].get("owner") != OWNER for g in DEVICES):
            raise RuntimeError("GPU ownership changed")
        env = dict(manifest["environment"], PYTHONPATH=str(source))
        stage_id = hashlib.sha256(label.encode()).hexdigest()[:8]
        env.update(
            SMOKE_RAY_TMPDIR=f"/proc/{os.getpid()}/cwd/runs/q{stage_id}",
            VERL_ROLLOUT_IPC_NAMESPACE=f"qwen-three-{stage_id}",
        )
        command = [
            str(python),
            str(source / "examples/mirroropd/benchmarks/fusion/resource_guard.py"),
            "run",
            "--repo",
            str(repo),
            "--source",
            str(source),
            "--gpus",
            "4,5,6,7",
            "--",
            str(python),
            *map(str, args),
        ]
        stage = dict(name=label, command=command, started_at=time.time(), state="running")
        state["stages"].append(stage)
        logs = study / "logs"
        logs.mkdir(exist_ok=True)
        with (logs / f"{label}.log").open("x") as log:
            child = subprocess.Popen(
                command, cwd=source, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT
            )
            stage["pid"] = child.pid
            publish(state="running", current=label)
            code = child.wait()
        stage.update(exit_code=code, finished_at=time.time(), state="complete" if code == 0 else "failed")
        publish()
        if code:
            raise RuntimeError(f"{label} failed with exit {code}; no automatic retries")

    try:
        while True:
            frozen()
            acquired, reason = queue.reserve_all(broker, DEVICES)
            if acquired:
                break
            publish(wait_reason=reason)
            time.sleep(30)
        # Attach the supervisor itself: broker restores all four occupiers on any exit.
        for gpu in DEVICES:
            broker.attach(gpu, OWNER, os.getpid(), exit_receipt=study / "exit.json")
        for mode in ("gate", "formal"):
            root = study / mode
            for producer in ("initial", "teacher", "finalize"):
                name = producer if producer != "finalize" else "E64"
                execute(
                    f"{mode}-bootstrap-{producer}",
                    [
                        "-m",
                        "verl.trainer.evaluation.bootstrap",
                        "--config",
                        root / "configs" / f"{name}.yaml",
                        "--producer",
                        producer,
                    ],
                )
            for experiment in EXPERIMENTS:
                execute(
                    f"{mode}-{experiment}",
                    ["-m", "verl.trainer.main_ppo", "--config-path", root / "configs", "--config-name", experiment],
                )
                audit_run(root, experiment, gate=mode == "gate")
        publish(state="complete", current=None)
        write(study / "exit.json", dict(returncode=0, finished_at=time.time()))
    except BaseException as exc:
        publish(state="failed", error=repr(exc))
        write(study / "exit.json", dict(returncode=1, error=repr(exc), finished_at=time.time()))
        raise
    finally:
        # Attached supervisor must exit before release. Broker daemon verifies PID,
        # locks and CUDA cleanup, then restores gpu-occupy automatically.
        if acquired:
            publish(occupancy_restoration="broker_auto_release_on_supervisor_exit")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--broker", type=Path)
    args = parser.parse_args()
    values = args.repo.resolve(), args.source.resolve(), args.study.resolve(), Path(sys.executable)
    if args.action == "prepare":
        prepare(*values)
    else:
        run(*values, args.broker.resolve())


if __name__ == "__main__":
    main()
