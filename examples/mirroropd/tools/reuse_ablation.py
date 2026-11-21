# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Prepare matched-update reuse ablations; execute through the existing GPU queue.

The admitted reference owns the pair, data and evaluation identity. Every arm
uses the same explicit AdamW setting. Evaluation banks are reused for reporting;
the training controller never reads them.
"""

import argparse
import copy
import fcntl
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from examples.mirroropd.research.core.io import dump, read
from examples.mirroropd.research.runtime.source import freeze

ARMS = {"adaptive": 16, "fixed16": 16, "fixed8": 8, "fixed4": 4, "online": 1}


def normalized_prompt(value):
    messages = json.loads(value) if isinstance(value, str) else value
    return " ".join(" ".join(message["content"].split()).casefold() for message in messages)


def prepare_training_pool(config, root):
    """Exclude every configured evaluation question from one shared recycled pool."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    heldout = {
        normalized_prompt(row["prompt"])
        for dataset in config["trainer"]["evaluation"]["datasets"]
        for row in pq.read_table(dataset["path"], columns=["prompt"]).to_pylist()
    }
    table = pa.concat_tables([pq.read_table(path) for path in config["data"]["train_files"]])
    keep = [normalized_prompt(row["prompt"]) not in heldout for row in table.select(["prompt"]).to_pylist()]
    output = root / "questions/train.parquet"
    output.parent.mkdir()
    pq.write_table(table.filter(pa.array(keep)), output)
    from verl.trainer.evaluation.artifacts import file_sha256

    dump(
        root / "questions/provenance.json",
        dict(
            sources={path: file_sha256(path) for path in config["data"]["train_files"]},
            original_rows=table.num_rows,
            excluded_rows=keep.count(False),
            retained_rows=keep.count(True),
            output_sha256=file_sha256(output),
            policy="exclude normalized text of every evaluation dataset",
        ),
    )
    config["data"]["train_files"] = [str(output)]


def audit_prompt_pool(config):
    """Audit the entire recycled pool, including questions beyond the old 40-outer prefix."""
    import pyarrow.parquet as pq

    heldout = {
        normalized_prompt(row["prompt"])
        for dataset in config["trainer"]["evaluation"]["datasets"]
        for row in pq.read_table(dataset["path"], columns=["prompt"]).to_pylist()
    }
    training = [
        normalized_prompt(row["prompt"])
        for path in config["data"]["train_files"]
        for row in pq.read_table(path, columns=["prompt"]).to_pylist()
    ]
    if len(training) < config["data"]["train_batch_size"]:
        raise ValueError("Training pool must contain at least one full batch")
    if set(training).intersection(heldout):
        raise ValueError("Training pool overlaps evaluation questions")
    return dict(rows=len(training), unique_texts=len(set(training)), heldout_text_overlap=0)


def configure(reference, *, name, arm, root, source, original_source, updates, every, seed, lr=1e-6, weight_decay=0.01):
    """Preserve scientific inputs and use one explicit AdamW setting across arms."""
    from examples.mirroropd.tools.optimizer_campaign import rewrite

    if not math.isfinite(lr) or lr <= 0 or not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("Require positive finite AdamW LR and nonnegative finite weight decay")
    config = rewrite(copy.deepcopy(reference), [(original_source, str(source))])
    output = root / "training" / name
    trainer = config["trainer"]
    trainer.update(
        total_training_steps=updates,  # Outer safety cap, not the stopping clock.
        update_budget=dict(total=updates, eval_every=every, save_every=every),
        reuse_control=dict(
            mode="adaptive" if arm == "adaptive" else "fixed",
            minimum=1 if arm == "online" else 4,
            window=5,
            patience=2,
            threshold=0.9,
            warmup_updates=160,
            min_drop=1e-8,
        ),
        save_freq=1,  # The update budget owns the actual checkpoint cadence.
        test_freq=-1,  # The native evaluation coordinator owns validation.
        max_actor_ckpt_to_keep=2,
        project_name="mirroropd-reuse",
        experiment_name=name,
        logger=["console", "file", "wandb"],
        default_local_dir=str(output / "runtime"),
        resume_mode="disable",
        resume_from_path=None,
    )
    actor = config["actor_rollout_ref"]["actor"]
    actor.update(ppo_epochs=ARMS[arm], data_loader_seed=seed)
    actor["optim"].update(
        optimizer="AdamW",
        optimizer_impl="torch.optim",
        lr=lr,
        betas=[0.9, 0.999],
        weight_decay=weight_decay,
        lr_scheduler_type="constant",
        lr_warmup_steps=0,
        lr_warmup_steps_ratio=0.0,
        override_optimizer_config=dict(eps=1e-8),
    )
    actor["checkpoint"].update(
        save_contents=["model", "optimizer", "extra", "hf_model"],
        load_contents=["model", "optimizer", "extra"],
        preserve_hf_master_weights=True,
        async_save=False,
    )
    config["data"]["seed"] = seed
    config["actor_rollout_ref"]["rollout"]["seed"] = seed
    evaluation = trainer["evaluation"]
    # Bank identity includes dataset paths and protocols; never rewrite them.
    evaluation["datasets"] = copy.deepcopy(reference["trainer"]["evaluation"]["datasets"])
    evaluation.update(
        output=str(output / "evaluation"),
        schedules={"banks": dict(every=every), "bench": dict(at=[], final=False)},
        export_outers=[],
        final_export=True,
        verify_export=True,
    )
    ray = config["ray_kwargs"]["ray_init"]
    ray["address"] = "local"
    ray["_temp_dir"] = "${oc.env:SMOKE_RAY_TMPDIR}"
    ray["runtime_env"]["env_vars"].update(
        PYTHONPATH=str(source),
        WANDB_MODE="offline",
        WANDB_DIR=str(output),
        VERL_FILE_LOGGER_PATH=str(output / "metrics.jsonl"),
        VERL_ROLLOUT_IPC_NAMESPACE="${oc.env:VERL_ROLLOUT_IPC_NAMESPACE}",
    )
    return config


def prepare(
    repo, root, origin, *, devices=(4, 5, 6, 7), updates=1280, every=160, seeds=(42,), lr=1e-6, weight_decay=0.01
):
    import yaml
    from omegaconf import OmegaConf

    from examples.mirroropd.tools.absorption_queue import preflight, verify_reused_bootstrap
    from verl.trainer.evaluation.artifacts import file_sha256
    from verl.trainer.ppo.v1.reuse import UpdateBudget, build_training_control

    UpdateBudget(total=updates, eval_every=every, save_every=every)
    if len(devices) != 4 or len(set(devices)) != 4 or any(type(d) is not int or d < 0 for d in devices):
        raise ValueError("Exactly four distinct GPU IDs are required")
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("Require distinct nonnegative training seeds")
    original = read(origin / "prepared.json")
    if len(original["experiments"]) != 1:
        raise ValueError("Require one admitted reference experiment")
    for path, digest in original["configs"].items():
        if file_sha256(path) != digest:
            raise ValueError("Reference config changed")
    reference_path = Path(next(iter(original["experiments"].values())))
    reference = yaml.safe_load(reference_path.read_text())
    domains = {d["name"]: d["domain"] for d in reference["trainer"]["evaluation"]["datasets"]}
    if any(domains.get(name) != name for name in ("math", "code", "if")):
        raise ValueError("Require separate Math, Code and IF held-out datasets")
    student = reference["actor_rollout_ref"]["model"]["path"]
    teacher = reference["distillation"]["teacher_models"]["teacher_model"]["model_path"]
    if "/Qwen--Qwen3-1.7B/" not in student or "/Qwen--Qwen3-4B-Instruct-2507/" not in teacher:
        raise ValueError("This ablation requires the admitted Qwen3-1.7B <- Qwen3-4B-Instruct-2507 pair")
    bootstrap = Path(reference["trainer"]["evaluation"]["bootstrap"])
    digest = file_sha256(bootstrap / "manifest.json")
    verify_reused_bootstrap(OmegaConf.create(reference), digest)
    root.mkdir(parents=True, exist_ok=False)
    prepare_training_pool(reference, root)
    pool = audit_prompt_pool(reference)
    source = freeze(repo, root / "source")
    (root / "configs").mkdir()
    manifest = dict(
        repo=str(repo),
        source=str(source),
        devices=list(devices),
        experiments={},
        teacher_configs={},
        configs={},
        code={str(p.relative_to(source)): file_sha256(p) for p in source.rglob("*.py")},
        reuse_bootstrap={},
        reference=dict(path=str(reference_path), sha256=file_sha256(reference_path)),
        environment={
            **{k: os.environ[k] for k in ("PATH", "LD_LIBRARY_PATH", "HF_HOME", "HF_HUB_CACHE") if k in os.environ},
            "CUDA_VISIBLE_DEVICES": ",".join(map(str, devices)),
            "PYTHONPATH": str(source),
            "MIRROROPD_WORKSPACE": str(repo.parent),
            "WANDB_MODE": "offline",
            "VERL_USE_UV": "0",
            "OMP_NUM_THREADS": "1",
        },
    )
    plan = dict(
        pair=dict(student=student, teacher=teacher),
        updates=updates,
        evaluation_updates=sorted({0, *range(every, updates + 1, every), updates}),
        devices=list(devices),
        batch=reference["data"]["train_batch_size"],
        optimizer=dict(name="AdamW", lr=lr, betas=[0.9, 0.999], eps=1e-8, weight_decay=weight_decay),
        seeds=list(seeds),
        data_policy="cycle the same shuffled prompt pool; generate new on-policy responses on every visit",
        prompt_pool=pool,
        runs=[],
        controller="native pre-update RKL only; completed-batch decisions apply to next rollout batch",
        restart="controller checkpoints are durable; unified evaluation currently requires fresh runs",
    )
    for seed in seeds:
        for arm, reuse in ARMS.items():
            name = f"reuse-{arm}-s{seed}"
            config = configure(
                reference,
                name=name,
                arm=arm,
                root=root,
                source=source,
                original_source=original["source"],
                updates=updates,
                every=every,
                seed=seed,
                lr=lr,
                weight_decay=weight_decay,
            )
            typed = OmegaConf.create(config)
            build_training_control(typed)
            verify_reused_bootstrap(typed, digest)
            if config["trainer"]["n_gpus_per_node"] != 4:
                raise ValueError("Reference must already use four-GPU shared-sync execution")
            if config["trainer"]["evaluation"].get("source_enabled") or config["trainer"]["evaluation"].get(
                "history_enabled"
            ):
                raise ValueError("Use the admitted fixed reporting-bank protocol")
            path = root / "configs" / f"{name}.yaml"
            path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
            preflight(path)
            manifest["experiments"][name] = str(path)
            manifest["configs"][str(path)] = file_sha256(path)
            manifest["reuse_bootstrap"][name] = dict(path=str(bootstrap), sha256=digest)
            plan["runs"].append(
                dict(name=name, arm=arm, reuse=reuse, adaptive=arm == "adaptive", seed=seed, config=str(path))
            )
    dump(root / "prepared.json", manifest)
    dump(root / "plan.json", plan)


def run(root, broker):
    """Use one existing four-GPU queue; periodically refresh CPU-only reports."""
    plotting_python = Path(sys.executable).parents[2] / "plotting/bin/python"
    if not plotting_python.is_file():
        raise ValueError("Missing isolated plotting environment")

    def render():
        with (root / "report.log").open("a") as log:
            subprocess.run(
                [str(plotting_python), "-m", "examples.mirroropd.tools.reuse_report", "--root", str(root)],
                cwd=manifest["source"],
                env=manifest["environment"],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )

    with (root / "campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "launch.json").exists():
            raise ValueError("Campaign already launched; never silently replay a training run")
        manifest = read(root / "prepared.json")
        command = [
            sys.executable,
            "-m",
            "examples.mirroropd.tools.absorption_queue",
            "run",
            "--study",
            str(root),
            "--broker",
            str(broker),
        ]
        dump(root / "launch.json", dict(pid=os.getpid(), command=command, started_at=time.time()))
        with (root / "supervisor.log").open("x") as log:
            child = subprocess.Popen(
                command, cwd=manifest["source"], env=manifest["environment"], stdout=log, stderr=subprocess.STDOUT
            )
            dump(root / "queue-process.json", dict(pid=child.pid, command=command))
            next_report = 0
            while child.poll() is None:
                if time.monotonic() >= next_report:
                    try:
                        render()
                    except Exception as error:
                        # Reporting never changes the training trajectory; preserve the original failure.
                        dump(root / "report-error.json", dict(error=repr(error), time=time.time()))
                    next_report = time.monotonic() + 60
                time.sleep(5)
            code = child.wait()
        render()
        if code:
            raise RuntimeError(f"Reuse queue failed ({code}); inspect supervisor.log and logs/")


def verify_run(root, config):
    """Reject missing/duplicate updates, unmatched evaluation budgets and incomplete exports."""
    from verl.trainer.evaluation.artifacts import ArtifactStore

    budget = config["trainer"]["update_budget"]
    total, every = budget["total"], budget["eval_every"]
    events = [ArtifactStore(path).manifest(complete=True) for path in sorted(root.root.glob("events/outer-*"))]
    observed = []
    for event in events:
        metadata = event["metadata"]
        if not metadata["coverage"]["scientific_complete"]:
            raise ValueError("Incomplete bank scoring coverage")
        observed.append(metadata["optimizer_steps"])
    if sorted(observed) != sorted({0, *range(every, total + 1, every), total}):
        raise ValueError(f"Evaluation update boundaries differ: {observed}")
    rows = [json.loads(line)["data"] for line in (root.root.parent / "metrics.jsonl").read_text().splitlines()]
    updates = [row for row in rows if "optim/actor/update_applied" in row]
    if (
        [row["step/optimizer"] for row in updates] != list(range(1, total + 1))
        or [row["step/update"] for row in updates] != list(range(1, total + 1))
        or any(row["optim/actor/update_applied"] != 1 for row in updates)
    ):
        raise ValueError("Native records do not prove the exact successful-update budget")
    audit = ArtifactStore(root.root / "final-model/export-audit").manifest(complete=True)
    if audit["metadata"]["native_reload_rkl"] != "passed":
        raise ValueError("Final export reload verification failed")
    return dict(
        updates=total,
        outer=max(row["step/outer"] for row in updates),
        fresh_responses=max(row["step/outer"] for row in updates) * config["data"]["train_batch_size"],
        switches=[row["step/optimizer"] for row in rows if row.get("reuse/switched") == 1],
        evaluation_updates=sorted(observed),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--origin", type=Path)
    parser.add_argument("--broker", type=Path)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--devices", type=int, nargs=4, default=(4, 5, 6, 7))
    parser.add_argument("--seeds", type=int, nargs="+", default=(42,))
    parser.add_argument("--updates", type=int, default=1280)
    parser.add_argument("--every", type=int, default=160)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    args = parser.parse_args()
    if args.action == "run":
        if args.broker is None:
            parser.error("run requires --broker")
        run(args.root.resolve(), args.broker.resolve())
        return
    if args.origin is None:
        parser.error("prepare requires --origin")
    prepare(
        args.repo.resolve(),
        args.root.resolve(),
        args.origin.resolve(),
        devices=tuple(args.devices),
        updates=args.updates,
        every=args.every,
        seeds=tuple(args.seeds),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )


if __name__ == "__main__":
    main()
