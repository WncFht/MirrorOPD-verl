# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Prepare E94–E123, then wait for four released GPUs and execute through the broker.

No existing allocation is released. Each stage keeps immutable commands, logs,
exit receipts and source hashes. Failures stop the queue; retries use a new root.
"""

import argparse
import fcntl
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from examples.mirroropd.launchers.absorption import PAIRS, SNAPSHOTS
from examples.mirroropd.research.core.io import dump, read
from examples.mirroropd.research.runtime.processes import identity as process_identity
from examples.mirroropd.research.runtime.source import freeze


def selected_pairs(names=None):
    """Select explicit unique IDs without changing the append-only pair catalog."""
    if names is None:
        return PAIRS
    catalog = {f"E{row[0]}": row for row in PAIRS}
    if not names or len(names) != len(set(names)) or set(names) - catalog.keys():
        raise ValueError("Prepare requires unique registered experiment IDs")
    return tuple(catalog[name] for name in names)


def prepare(repo, study, names=None):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import yaml

    from examples.mirroropd.research.studies.prefix_fusion.native_source_queue import config_only, environment
    from verl.trainer.evaluation.artifacts import file_sha256

    pairs = selected_pairs(names)
    study.mkdir(parents=True, exist_ok=False)
    qroot = study / "questions"
    qroot.mkdir()
    original = repo / "runs/qwen-prefix-six-20261031/prepared-03"
    candidates = read(original / "heldout-questions.json")
    datasets = []
    for domain, count, verifier in (("math", 128, "math"), ("code", 64, "lcb"), ("if", 64, "ifeval")):
        rows = [r for r in candidates if r["domain"] == domain][:count]
        if len(rows) != count or len({r["id"] for r in rows}) != count:
            raise ValueError("Insufficient unique heldout questions")
        path = qroot / f"{domain}.parquet"
        pq.write_table(
            pa.Table.from_pylist(
                [dict(id=str(r["id"]), prompt=json.dumps(r["messages"], ensure_ascii=False)) for r in rows]
            ),
            path,
        )
        datasets.append(
            dict(
                name=domain,
                domain=domain,
                path=str(path),
                verifier=verifier,
                n=1,
                protocol="public",
                panel_ids=[str(r["id"]) for r in rows],
            )
        )
    dump(qroot / "datasets.json", datasets)
    shutil.copy2(original / "train.parquet", qroot / "train.parquet")
    training = pq.read_table(qroot / "train.parquet").to_pylist()
    if len(training) < 2560:
        raise ValueError("Not enough fresh DAPO training prompts")

    # Check textual overlap as well as preserving the earlier prepared split receipt.
    def text(value):
        if isinstance(value, str):
            value = json.loads(value)
        return " ".join(" ".join(m["content"].split()).casefold() for m in value)

    heldout = {text(r["messages"]) for r in candidates}
    if any(text(row["prompt"]) in heldout for row in training[:2560]):
        raise ValueError("Training and heldout prompt overlap")
    dump(
        qroot / "provenance.json",
        dict(
            source=str(original),
            source_receipt_sha256=file_sha256(original / "complete.json"),
            train_rows=len(training),
            new_rollouts=True,
            train_heldout_exact_text_overlap=0,
            files={p.name: file_sha256(p) for p in qroot.iterdir() if p.is_file()},
        ),
    )
    source = freeze(repo, study / "source")
    shutil.copytree(
        repo / "examples/mirroropd/launchers",
        source / "examples/mirroropd/launchers",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    for path in (repo / "examples/mirroropd").glob("*.sh"):
        shutil.copy2(path, source / "examples/mirroropd" / path.name)
    env = environment(repo, source, study, Path(sys.executable), devices=(0, 1, 2, 3))
    env.update(
        ABSORPTION_ROOT=str(study),
        TRAIN_BATCH_SIZE="64",
        TOTAL_TRAINING_STEPS="40",
        PPO_EPOCHS="16",
        MAX_VAL_RESPONSE_LENGTH="7168",
        HF_HUB_OFFLINE="1",
        PYTHONPATH=str(source),
        LD_LIBRARY_PATH=os.environ.get("LD_LIBRARY_PATH", ""),
        MIRROROPD_RAY_PREWARM="1",
        RAY_raylet_start_wait_time_s="300",
        RAY_agent_register_timeout_ms="300000",
    )
    configs = {}
    teacher_configs = {}
    model_checks = []

    def prepare_pair(row):
        number = row[0]
        name = f"E{number}"
        config, path = config_only(source, study, Path(sys.executable), env, name, study / "training" / name)
        if config["trainer"]["evaluation"]["accuracy_enabled"]:
            raise ValueError("Accuracy must be disabled")
        # Full native bank references: no imported short bank and no hidden H1024 slicing.
        if config["trainer"]["evaluation"].get("fixed_banks"):
            raise ValueError("New study must generate its own full-response banks")
        for role, model in (
            ("student", config["actor_rollout_ref"]["model"]["path"]),
            ("teacher", config["distillation"]["teacher_models"]["teacher_model"]["model_path"]),
        ):
            directory = Path(model)
            index = directory / "model.safetensors.index.json"
            shards = set(read(index)["weight_map"].values()) if index.exists() else {"model.safetensors"}
            missing = [s for s in shards if not (directory / s).is_file()]
            if missing:
                raise FileNotFoundError(f"{name} {role}: missing {missing} under {directory}")
            model_checks.append(dict(experiment=name, role=role, path=model, shards=len(shards)))
        configs[name] = str(path)
        # Bootstrap throughput follows the model being loaded, not student size.
        teacher_alias = next(row[2] for row in PAIRS if name == f"E{row[0]}")
        sequences = (
            128
            if teacher_alias in {"deepscaler", "JustRL-DeepSeek-1.5B", "minimath", "nemotron", "brorl"}
            else 64
            if teacher_alias in {"Qwen3-1.7B-GRPO-KL-math-reasoning", "e3-1.7B", "Qwen3-1.7B-Math"}
            else 32
        )
        teacher_config = json.loads(json.dumps(config))
        teacher_config["actor_rollout_ref"]["rollout"]["max_num_seqs"] = sequences
        teacher_config["trainer"]["evaluation"]["generation_concurrency"] = 4 * sequences
        teacher_path = path.with_name(f"{name}-teacher.yaml")
        teacher_path.write_text(yaml.safe_dump(teacher_config, sort_keys=False, allow_unicode=True))
        teacher_configs[name] = str(teacher_path)
        # CPU pair/token/context/prompt audit runs before any broker reservation.
        audit = study / "preflight" / f"{name}.log"
        audit.parent.mkdir(exist_ok=True)
        with audit.open("x") as log:
            code = subprocess.call(
                [
                    sys.executable,
                    "-m",
                    "examples.mirroropd.research.studies.absorption_dynamics.queue",
                    "preflight",
                    "--study",
                    str(study),
                    "--config",
                    str(path),
                ],
                cwd=source,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if code:
            raise RuntimeError(f"CPU compatibility preflight failed for {name}: {audit}")

    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(prepare_pair, pairs))
    configs = {f"E{row[0]}": configs[f"E{row[0]}"] for row in pairs}
    dump(
        study / "prepared.json",
        dict(
            experiments=configs,
            teacher_configs=teacher_configs,
            repo=str(repo),
            devices=[0, 1, 2, 3],
            models=model_checks,
            source=str(source),
            environment=env,
            configs={p: file_sha256(p) for p in (*configs.values(), *teacher_configs.values())},
            code={str(p.relative_to(source)): file_sha256(p) for p in source.rglob("*.py")},
            settings=dict(outer=40, reuse=16, batch=64, cap=7168, accuracy=False, snapshots=SNAPSHOTS),
        ),
    )


def partition_numbers(experiments):
    """Keep historical assignments and split extensions from the accepted manifest."""
    front = {94, 96, 98, 99, 101, 103, 105, 107, 110}
    back = {95, 97, 100, 102, 104, 106, 108, 109, 111}
    numbers = sorted(int(name.removeprefix("E")) for name in experiments)
    if any(f"E{n}" not in {f"E{row[0]}" for row in PAIRS} for n in numbers):
        raise ValueError("Manifest contains an unregistered absorption experiment")
    extra = [n for n in numbers if n not in front | back]
    return (
        tuple(n for n in numbers if n in front or n in extra[::2]),
        tuple(n for n in numbers if n in back or n in extra[1::2]),
    )


def partition(origin, study):
    """Relocate accepted CPU configurations into two disjoint four-GPU lanes."""
    from omegaconf import OmegaConf

    from verl.trainer.evaluation.artifacts import file_sha256
    from verl.trainer.evaluation.generation_contract import require_current_generation

    original = read(origin / "prepared.json")
    for path, digest in original["configs"].items():
        if file_sha256(path) != digest:
            raise ValueError("Accepted configuration changed before partition")
    for relative, digest in original["code"].items():
        if file_sha256(Path(original["source"]) / relative) != digest:
            raise ValueError("Accepted source changed before partition")
    front, back = partition_numbers(original["experiments"])
    study.mkdir(parents=True, exist_ok=False)
    source = study / "source"
    shutil.copytree(Path(original["source"]), source)
    runner = Path(__file__).resolve()
    shutil.copy2(runner, source / "examples/mirroropd/research/studies/absorption_dynamics/queue.py")
    shutil.copy2(
        Path(original["repo"]) / "examples/mirroropd/launchers/absorption.py",
        source / "examples/mirroropd/launchers/absorption.py",
    )
    groups = []
    for label, numbers, devices in (("front4", front, [0, 1, 2, 3]), ("back4", back, [4, 5, 6, 7])):
        lane = study / f"absorption-{label}"
        lane.mkdir()
        shutil.copytree(origin / "questions", lane / "questions")
        manifest = json.loads(json.dumps(original).replace(str(origin), str(lane)))
        names = [f"E{number}" for number in numbers]
        manifest["experiments"] = {n: manifest["experiments"][n] for n in names}
        manifest["teacher_configs"] = {n: manifest["teacher_configs"][n] for n in names}
        # Scientific modules are copied unchanged; the new snapshot includes the queue fix.
        manifest["source"] = str(source)
        manifest["environment"]["PYTHONPATH"] = str(source)
        manifest["environment"]["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devices))
        manifest["devices"] = devices
        manifest["models"] = [m for m in manifest["models"] if m["experiment"] in names]
        (lane / "configs").mkdir()
        (lane / "preflight").mkdir()
        paths = []
        for name in names:
            for field in ("experiments", "teacher_configs"):
                old_path = Path(original[field][name])
                target = Path(manifest[field][name])
                target.write_text(
                    old_path.read_text().replace(str(origin), str(lane)).replace(str(lane / "source"), str(source))
                )
                assert (
                    target.read_text().replace(str(source), original["source"]).replace(str(lane), str(origin))
                    == old_path.read_text()
                )
                config = OmegaConf.load(target)
                model_type = read(Path(config.actor_rollout_ref.model.path) / "config.json")["model_type"]
                config.actor_rollout_ref.model.trust_remote_code = model_type == "mimo"
                require_current_generation(config, independent_engine=True)
                require_current_generation(config)
                OmegaConf.save(config, target)
                paths.append(target)
            shutil.copy2(origin / "preflight" / f"{name}.log", lane / "preflight" / f"{name}.log")
        dataset = lane / "questions/datasets.json"
        dataset.write_text(dataset.read_text().replace(str(origin), str(lane)))
        provenance = read(lane / "questions/provenance.json")
        provenance["files"] = {
            p.name: file_sha256(p)
            for p in (lane / "questions").iterdir()
            if p.is_file() and p.name != "provenance.json"
        }
        provenance["accepted_origin"] = str(origin)
        dump(lane / "questions/provenance.json", provenance)
        manifest["code"] = {str(p.relative_to(source)): file_sha256(p) for p in source.rglob("*.py")}
        manifest["configs"] = {str(path): file_sha256(path) for path in paths}
        manifest["partition"] = dict(
            origin=str(origin),
            accepted_manifest_sha256=file_sha256(origin / "prepared.json"),
            changes=[
                "artifact paths",
                "physical GPU mapping",
                "queue environment and legacy guard fixes",
                "standard architectures disable remote-code execution; native sampler gate passed",
            ],
            cpu_preflight="inherited",
        )
        dump(lane / "prepared.json", manifest)
        groups.append(dict(study=str(lane), devices=devices, experiments=names))
    dump(study / "partition.json", dict(groups=groups, experiments=len(original["experiments"]), gpus_per_experiment=4))


def preflight(path):
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer

    from verl.trainer.distillation.compatibility import preflight_pair
    from verl.trainer.evaluation.config import EvaluationConfig
    from verl.trainer.evaluation.coordinator import EvaluationCoordinator, read_questions
    from verl.trainer.ppo.readout_callback import readout_options

    config = OmegaConf.load(path)
    from verl.trainer.ppo.reward import get_custom_reward_fn

    get_custom_reward_fn(config)
    preflight_pair(config)
    options = EvaluationConfig.parse(readout_options(config, EvaluationCoordinator))
    from verl.utils.tokenizer.continuous_token_wiring import resolve_continuous_token_model_family

    resolve_continuous_token_model_family(
        OmegaConf.select(config, "data.continuous_token.model_family", default="auto")
    )
    for protocol in options.protocols.values():
        resolve_continuous_token_model_family(protocol.model_family)
    tokenizer = AutoTokenizer.from_pretrained(
        config.actor_rollout_ref.model.path, trust_remote_code=True, local_files_only=True
    )
    template = json.loads(options.protocols["public"].template_json)
    for dataset in options.datasets:
        for row in read_questions(dataset):
            ids = tokenizer.apply_chat_template(
                row["prompt"], tokenize=True, add_generation_prompt=True, return_dict=False, **template
            )
            if len(ids) > 4096:
                raise ValueError(f"Overlong heldout prompt: {dataset.name}/{row['id']}: {len(ids)}")
    print("CPU pair, token semantics, context and heldout prompt budget passed", flush=True)


def retry(origin, study, names, reuse_from=None):
    """Create a fresh immutable attempt for untrained pairs across failed lanes."""
    from omegaconf import OmegaConf

    from verl.trainer.evaluation.artifacts import file_sha256

    lanes = [p for p in origin.glob("absorption-*4") if (p / "prepared.json").exists()]
    originals = [(p, read(p / "prepared.json")) for p in lanes]
    originals = [(p, m) for p, m in originals if not m.get("retry_of")]
    if reuse_from is not None:
        originals = [(reuse_from, read(reuse_from / "prepared.json"))]
    if not names or len(names) != len(set(names)):
        raise ValueError("Retry requires unique explicit experiment IDs")
    selected = {}
    for name in names:
        matches = [(p, m) for p, m in originals if name in m["experiments"]]
        if len(matches) != 1:
            raise ValueError(f"Ambiguous original attempt: {name}")
        lane, manifest = matches[0]
        status = read(lane / "status.json")
        if status["state"] not in ("failed", "complete", "stopped"):
            raise ValueError("Original supervisor has not reached a terminal state")
        entered = any(s["label"] == f"{name}-train" for s in status["stages"])
        if (lane / f"{name}-complete.json").exists() or (entered and reuse_from is None):
            raise ValueError(f"Refuse restarting previously entered training: {name}")
        if entered:
            metrics = lane / "training" / name / "metrics.jsonl"
            rows = [json.loads(line) for line in metrics.read_text().splitlines()] if metrics.exists() else []
            if any(row["data"].get("step/optimizer", 0) > 0 for row in rows):
                raise ValueError("Recovery requires native trainer resume after any completed update")
            if list((lane / "training" / name).glob("**/outer-001/manifest.json")):
                raise ValueError("Refuse restarting a completed outer")
            failure = (lane / "logs" / f"{name}-train.log").read_text()
            if "Custom module file not found:" not in failure:
                raise ValueError("Zero-update restart requires the diagnosed missing reward-module failure")
        selected[name] = lane, manifest
    study.mkdir(parents=True, exist_ok=False)
    lane, original = selected[names[0]]
    source = study / "source"
    shutil.copytree(original["source"], source, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    repo = Path(original["repo"])
    patches = (
        "verl/utils/ray_utils.py",
        "verl/trainer/main_ppo.py",
        "verl/trainer/evaluation/bootstrap.py",
        "examples/mirroropd/launchers/absorption.py",
    )
    for relative in patches:
        shutil.copy2(repo / relative, source / relative)
    shutil.copytree(repo / "examples/mirroropd/rewards", source / "examples/mirroropd/rewards", dirs_exist_ok=True)
    shutil.copytree(lane / "questions", study / "questions")
    (study / "configs").mkdir()
    manifest = dict(original, source=str(source), devices=[0, 1, 2, 3], experiments={}, teacher_configs={}, configs={})
    reuse = {}
    for name, (old_lane, old) in selected.items():
        for field in ("experiments", "teacher_configs"):
            previous = Path(old[field][name])
            if file_sha256(previous) != old["configs"][str(previous)]:
                raise ValueError("Original configuration changed")
            content = previous.read_text().replace(str(old_lane), str(study)).replace(str(old["source"]), str(source))
            config = OmegaConf.create(content)
            if config.data.continuous_token.model_family == "llama":
                config.data.continuous_token.model_family = "default"
            for protocol in config.trainer.evaluation.protocols.values():
                if protocol.model_family == "llama":
                    protocol.model_family = "default"
            bootstrap = old_lane / "bootstrap" / name
            if reuse_from is not None and (bootstrap / "manifest.json").exists():
                original_config = OmegaConf.load(previous)
                config.trainer.evaluation.bootstrap = str(bootstrap)
                config.trainer.evaluation.datasets = original_config.trainer.evaluation.datasets
                if read(bootstrap / "manifest.json")["status"] != "complete":
                    raise ValueError("Only finalized banks may be reused")
                reuse[name] = dict(path=str(bootstrap), sha256=file_sha256(bootstrap / "manifest.json"))
            from verl.trainer.ppo.reward import get_custom_reward_fn

            get_custom_reward_fn(config)
            target = study / "configs" / previous.name
            OmegaConf.save(config, target)
            manifest[field][name] = str(target)
            manifest["configs"][str(target)] = file_sha256(target)
    env = dict(original["environment"])
    env = {k: v.replace(str(lane), str(study)).replace(str(original["source"]), str(source)) for k, v in env.items()}
    env.update(
        PYTHONPATH=str(source),
        CUDA_VISIBLE_DEVICES="0,1,2,3",
        MIRROROPD_RAY_PREWARM="1",
        RAY_raylet_start_wait_time_s="300",
        RAY_agent_register_timeout_ms="300000",
    )
    manifest.update(
        environment=env,
        models=[m for _, old in originals for m in old["models"] if m["experiment"] in names],
        code={str(p.relative_to(source)): file_sha256(p) for p in source.rglob("*.py")},
        retry_of={n: str(p) for n, (p, _) in selected.items()},
        source_patches=[*patches, "examples/mirroropd/rewards/reward.py"],
        reuse_bootstrap=reuse,
    )
    dump(study / "prepared.json", manifest)


def verify_reused_bootstrap(config, digest):
    """Read-only verification of a finalized bank before skipping generation."""
    from verl.trainer.evaluation.artifacts import ArtifactStore, file_sha256, identity
    from verl.trainer.evaluation.config import EvaluationConfig
    from verl.trainer.evaluation.coordinator import EvaluationCoordinator
    from verl.trainer.evaluation.identity import bootstrap_contract, model_identity
    from verl.trainer.evaluation.trajectory import select_panel, trajectory_index
    from verl.trainer.ppo.readout_callback import readout_options

    options = EvaluationConfig.parse(readout_options(config, EvaluationCoordinator))
    root = ArtifactStore(options.bootstrap)
    if file_sha256(root.manifest_path) != digest:
        raise ValueError("Reusable bootstrap manifest changed")
    manifest = root.manifest(complete=True)
    expected = bootstrap_contract(
        options,
        student=model_identity(config.actor_rollout_ref.model.path),
        teacher=model_identity(config.distillation.teacher_models.teacher_model.model_path),
    )
    if identity(expected) != manifest["metadata"]["contract_sha256"]:
        raise ValueError("Reusable bank models, questions or protocols differ")
    for name in options.bootstrap_children:
        child = ArtifactStore(root.root / name)
        child.manifest(complete=True)
        if file_sha256(child.manifest_path) != manifest["metadata"]["children_sha256"][name]:
            raise ValueError("Reusable bank child changed")
        select_panel(trajectory_index(child, bank=name), options.datasets)


def run(study, broker_path):
    from verl.trainer.evaluation.artifacts import ArtifactStore, file_sha256

    manifest = read(study / "prepared.json")
    repo = Path(manifest["repo"])
    source = Path(manifest["source"])
    owner = study.name
    spec = importlib.util.spec_from_file_location("absorption_broker", broker_path)
    broker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(broker)
    devices = manifest["devices"]
    acquired = []
    state = dict(state="waiting_for_gpus", pid=os.getpid(), identity=process_identity(os.getpid()), stages=[])
    dump(study / "process.json", state)

    def publish(**updates):
        state.update(updates, updated_at=time.time())
        dump(study / "status.json", state)

    def stopped():
        return (study / "STOP").exists()

    def execute(label, command):
        if stopped():
            raise InterruptedError("Study STOP exists")
        for relative, digest in manifest["code"].items():
            if file_sha256(source / relative) != digest:
                raise ValueError("Frozen source changed")
        env = dict(
            manifest["environment"],
            PYTHONPATH=str(source),
            SMOKE_RAY_TMPDIR=f"/tmp/ad-{os.getpid()}-{len(state['stages'])}",
            VERL_ROLLOUT_IPC_NAMESPACE=f"ad-{os.getpid()}-{len(state['stages'])}",
        )
        args = [
            sys.executable,
            str(broker_path.parent / "runtime/resource_guard.py"),
            "run",
            "--repo",
            str(repo),
            "--gpus",
            ",".join(map(str, devices)),
            "--",
            sys.executable,
            "-c",
            'import runpy,sys;sys.path.insert(0,sys.argv.pop(1));runpy.run_module(sys.argv.pop(1),run_name="__main__")',
            str(source),
            *map(str, command[1:]),
        ]
        stage = dict(label=label, command=args, started_at=time.time(), state="running")
        state["stages"].append(stage)
        publish(state="running", current=label)
        logs = study / "logs"
        logs.mkdir(exist_ok=True)
        with (logs / f"{label}.log").open("x") as stream:
            child = subprocess.Popen(args, cwd=source, env=env, stdout=stream, stderr=subprocess.STDOUT)
            stage["pid"] = child.pid
            publish()
            while child.poll() is None:
                if stopped():
                    child.terminate()
                    break
                time.sleep(2)
            code = child.wait(timeout=120)
        stage.update(returncode=code, state="complete" if code == 0 else "failed", finished_at=time.time())
        dump(logs / f"{label}-exit.json", stage)
        publish()
        if code:
            raise RuntimeError(f"Stage failed: {label}; original log and exit retained")

    try:
        for path, digest in manifest["configs"].items():
            if file_sha256(path) != digest:
                raise ValueError("Prepared config changed")
        while True:
            if stopped():
                raise InterruptedError("STOP before allocation")
            registry = broker.registry()["gpus"]
            blockers = {
                str(g): registry.get(str(g), {}).get("owner")
                for g in devices
                if registry.get(str(g), {}).get("desired") != "released"
                or registry.get(str(g), {}).get("owner") is not None
            }
            if not blockers:
                break
            publish(state="waiting_for_gpus", blockers=blockers)
            time.sleep(15)
        for gpu in devices:
            broker.reserve(gpu, owner)
            acquired.append(gpu)
            broker.attach(
                gpu, owner, os.getpid(), source_record=study / "process.json", exit_receipt=study / "exit.json"
            )
        initial_banks = {}
        for name, path in manifest["experiments"].items():
            from omegaconf import OmegaConf

            resolved = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
            bank_key = json.dumps(
                {
                    "student": resolved["actor_rollout_ref"]["model"]["path"],
                    "protocols": resolved["trainer"]["evaluation"]["protocols"],
                    "datasets": resolved["trainer"]["evaluation"]["datasets"],
                    "rollout": resolved["actor_rollout_ref"]["rollout"],
                },
                sort_keys=True,
            )
            reused = manifest.get("reuse_bootstrap", {}).get(name)
            if reused:
                verify_reused_bootstrap(OmegaConf.load(path), reused["sha256"])
                dump(study / f"{name}-bank-reuse.json", reused)
            for producer in () if reused else ("initial", "teacher", "finalize"):
                producer_config = manifest["teacher_configs"][name] if producer == "teacher" else path
                command = [
                    "-m",
                    "verl.trainer.evaluation.bootstrap",
                    "--config",
                    producer_config,
                    "--producer",
                    producer,
                ]
                if producer == "initial" and bank_key in initial_banks:
                    command.extend(["--resume-generation", initial_banks[bank_key]])
                execute(f"{name}-{producer}", command)
                if producer == "initial":
                    initial_banks[bank_key] = str(study / "bootstrap" / name / "initial")
            execute(
                f"{name}-train",
                ["-m", "verl.trainer.main_ppo", "--config-path", Path(path).parent, "--config-name", name],
            )
            root = ArtifactStore(study / "training" / name / "evaluation")
            root.manifest(complete=True)
            for outer in range(41):
                event = ArtifactStore(root.root / f"events/outer-{outer:03d}").manifest(complete=True)
                if not event["metadata"]["coverage"]["scientific_complete"]:
                    raise ValueError("Incomplete bank scoring coverage")
                if event["metadata"]["optimizer_steps"] != outer * 16:
                    raise ValueError("Optimizer update budget differs")
                if any(k.startswith("eval/accuracy/") for k in event["metadata"]["metrics"]):
                    raise ValueError("Unexpected accuracy")
            for outer in SNAPSHOTS:
                location = root.root / ("final-model" if outer == 40 else f"snapshots/outer-{outer:03d}")
                audit = ArtifactStore(location / "export-audit").manifest(complete=True)
                if audit["metadata"]["native_reload_rkl"] != "passed":
                    raise ValueError("Snapshot native reload failed")
            dump(study / f"{name}-complete.json", dict(outer=40, updates=640, bank_cap=7168, accuracy=False))
        publish(state="complete", current=None)
        dump(study / "exit.json", dict(returncode=0, finished_at=time.time()))
    except BaseException as error:
        publish(state="stopped" if stopped() else "failed", error=repr(error))
        dump(study / "exit.json", dict(returncode=1, error=repr(error), finished_at=time.time()))
        raise
    finally:
        # Attached PID leases are released automatically only after this process
        # exits and the guard's CUDA contexts are gone. Never release other owners.
        publish(acquired=acquired, restoration="broker_auto_release_on_supervisor_exit")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "preflight", "partition", "retry"))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[5])
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--broker", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--from-study", type=Path)
    parser.add_argument("--pairs", nargs="+")
    parser.add_argument("--reuse-from", type=Path)
    args = parser.parse_args()
    study = args.study.resolve()
    if args.action == "prepare":
        prepare(args.repo.resolve(), study, args.pairs)
    elif args.action == "partition":
        partition(args.from_study.resolve(), study)
    elif args.action == "retry":
        retry(args.from_study.resolve(), study, args.pairs, args.reuse_from.resolve() if args.reuse_from else None)
    elif args.action == "preflight":
        preflight(args.config)
    else:
        with (study / "supervisor.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(study, args.broker.resolve())


if __name__ == "__main__":
    main()
