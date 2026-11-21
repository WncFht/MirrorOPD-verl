# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Matched Muon/AdamW long-budget probes, shared banks and bounded CPU reports."""

import argparse
import copy
import fcntl
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from examples.mirroropd.launchers.absorption import OPTIMIZER_PROBES, PAIRS
from examples.mirroropd.research.core.io import dump, read
from examples.mirroropd.research.runtime.source import freeze


def rewrite(value, replacements):
    """Relocate artifact paths recursively; leave external model/data paths intact."""
    if isinstance(value, str):
        for old, new in replacements:
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [rewrite(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: rewrite(item, replacements) for key, item in value.items()}
    return value


def configure(reference, *, name, study, source, original_source, origin, bootstrap, kind, lr, development):
    """Keep pair-specific semantics while matching optimizer, checkpoint and scoring policy."""
    replacements = [(original_source, str(source)), (str(origin), str(study))]
    config = rewrite(copy.deepcopy(reference), replacements)
    trainer = config["trainer"]
    output = study / f"training-{kind}" / name
    optim = config["actor_rollout_ref"]["actor"]["optim"]
    optim.update(
        optimizer="MuonAdamW" if kind == "muon" else "AdamW",
        optimizer_impl="verl.utils.fsdp_optimizers" if kind == "muon" else "torch.optim",
        lr=lr,
        weight_decay=0,
        lr_scheduler_type="constant",
        lr_warmup_steps=0,
        lr_warmup_steps_ratio=0.0,
        override_optimizer_config=(
            dict(
                aux_lr=1e-6,
                momentum=0.95,
                nesterov=True,
                ns_steps=5,
                adjust_lr_fn="match_rms_adamw",
                distributed_mode="owner",
            )
            if kind == "muon"
            else {}
        ),
    )
    config["actor_rollout_ref"]["actor"]["checkpoint"].update(
        save_contents=["model", "optimizer", "extra", "hf_model"],
        load_contents=["model", "optimizer", "extra"],
        preserve_hf_master_weights=True,
        async_save=False,
    )
    trainer.update(
        total_training_steps=40,
        save_freq=5,
        max_actor_ckpt_to_keep=2,
        project_name="mirroropd-optimizer-platform",
        experiment_name=f"{name}-{kind}-lr-{lr:.8g}",
        logger=["console", "file", "wandb"],
        default_local_dir=str(output / "runtime"),
        resume_mode="disable",
        resume_from_path=None,
    )
    evaluation = trainer["evaluation"]
    # Preserve the reference dataset paths: path is part of the bootstrap identity.
    evaluation["datasets"] = copy.deepcopy(reference["trainer"]["evaluation"]["datasets"])
    if not any(dataset["name"] == "development_math" for dataset in evaluation["datasets"]):
        evaluation["datasets"].append(copy.deepcopy(development))
    evaluation.update(
        output=str(output / "evaluation"),
        bootstrap=str(bootstrap),
        export_outers=[20, 40],
        final_export=True,
        verify_export=True,
    )
    ray_env = config["ray_kwargs"]["ray_init"]["runtime_env"]["env_vars"]
    ray_env.update(
        PYTHONPATH=str(source),
        WANDB_MODE="offline",
        WANDB_DIR=str(output),
        VERL_FILE_LOGGER_PATH=str(output / "metrics.jsonl"),
    )
    return config


def prepare(repo, root, references, dependency, probes=None):
    import yaml
    from omegaconf import OmegaConf

    from examples.mirroropd.tools.absorption_queue import preflight, verify_reused_bootstrap
    from verl.trainer.evaluation.artifacts import file_sha256

    root.mkdir(parents=True, exist_ok=False)
    source = freeze(repo, root / "source")
    code = {str(path.relative_to(source)): file_sha256(path) for path in source.rglob("*.py")}
    qwen = references["E102"]
    qmanifest = read(Path(qwen["study"]) / "prepared.json")
    qconfig = yaml.safe_load(Path(qmanifest["experiments"][qwen["experiment"]]).read_text())
    development = next(d for d in qconfig["trainer"]["evaluation"]["datasets"] if d["name"] == "development_math")
    # Existing split receipt was produced by development_split: formal prefix and held-out disjoint.
    split = Path(development["path"]).parent / "split.json"
    receipt = read(split)
    if receipt["exact_normalized_text_overlap"] != 0 or receipt["final_heldout_used_for_selection"]:
        raise ValueError("Development split is not independent")
    dump(root / "development-split.json", dict(source=str(split), sha256=file_sha256(split), receipt=receipt))
    banks = {102: Path(qconfig["trainer"]["evaluation"]["bootstrap"])}
    pair_rows = {row[0]: row for row in PAIRS}
    jobs = []
    for number, (pair, kind, lr) in (probes or OPTIMIZER_PROBES).items():
        number, pair = int(number), int(pair)
        if kind not in ("muon", "adamw") or not math.isfinite(lr) or lr <= 0:
            raise ValueError("Require muon/adamw and a positive finite LR")
        name = f"E{number}"
        origin = Path(references[f"E{pair}"]["study"])
        old_name = references[f"E{pair}"]["experiment"]
        original = read(origin / "prepared.json")
        path = Path(original["experiments"][old_name])
        if file_sha256(path) != original["configs"][str(path)]:
            raise ValueError(f"Reference config changed: {old_name}")
        study = root / f"{name.lower()}-{kind}-lr-{lr:.8g}"
        (study / "configs").mkdir(parents=True)
        (study / "preflight").mkdir()
        banks.setdefault(pair, root / "banks" / f"E{pair}")
        bootstrap = banks[pair]
        devices = [0, 1, 2, 3] if pair in (102, 117) else [4, 5, 6, 7]
        manifest = copy.deepcopy(original)
        manifest.update(
            repo=str(repo),
            source=str(source),
            code=code,
            devices=devices,
            experiments={},
            teacher_configs={},
            configs={},
            reuse_bootstrap={},
            optimizer=kind,
            calibration={name: dict(lr=lr, aux_lr=1e-6, status="full_budget_probe")},
            settings=dict(outer=40, reuse=16, batch=64, cap=7168, accuracy=False, snapshots=[20, 40]),
            models=[dict(model, experiment=name) for model in original["models"] if model["experiment"] == old_name],
        )
        manifest.pop("retry_of", None)
        manifest["environment"] = rewrite(
            original["environment"], [(original["source"], str(source)), (str(origin), str(study))]
        )
        manifest["environment"].update(
            PYTHONPATH=str(source), WANDB_MODE="offline", CUDA_VISIBLE_DEVICES=",".join(map(str, devices))
        )
        for field in ("experiments", "teacher_configs"):
            reference = yaml.safe_load(Path(original[field][old_name]).read_text())
            config = configure(
                reference,
                name=name,
                study=study,
                source=source,
                original_source=original["source"],
                origin=origin,
                bootstrap=bootstrap,
                kind=kind,
                lr=lr,
                development=development,
            )
            # Training data are shared read-only; relocation must never invent missing data paths.
            config["data"]["train_files"] = reference["data"]["train_files"]
            config["custom_reward_function"] = reference["custom_reward_function"]
            target = study / "configs" / f"{name}{'-teacher' if field == 'teacher_configs' else ''}.yaml"
            target.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
            preflight(target)
            manifest[field][name] = str(target)
            manifest["configs"][str(target)] = file_sha256(target)
            if pair == 102:
                verify_reused_bootstrap(OmegaConf.create(config), file_sha256(bootstrap / "manifest.json"))
        if pair == 102:
            manifest["reuse_bootstrap"][name] = dict(
                path=str(bootstrap), sha256=file_sha256(bootstrap / "manifest.json")
            )
        dump(study / "prepared.json", manifest)
        dump(study / "status.json", dict(state="queued", updated_at=time.time()))
        jobs.append(
            dict(
                experiment=name,
                study=str(study),
                pair=f"E{pair}",
                optimizer=kind,
                lr=lr,
                aux_lr=1e-6 if kind == "muon" else None,
                devices=devices,
                lane=0 if pair in (102, 117) else 1,
                bootstrap=str(bootstrap),
                student=pair_rows[pair][1],
                teacher=pair_rows[pair][2],
                budget=40,
                reuse=16,
            )
        )
    dump(
        root / "plan.json",
        dict(
            jobs=jobs,
            dependency=str(dependency),
            check_interval_seconds=7200,
            selection="development_math: mean relative reduction on both banks, outers31-40",
            plateau="report last10 mean, normalized slope and variability; no best-point selection",
            weight_decay=0,
            references=references,
        ),
    )


def retry(repo, root, origin, replacements):
    """Retain failed attempts and retry only jobs that never reached model workers."""
    import yaml
    from omegaconf import OmegaConf

    from examples.mirroropd.tools.absorption_queue import preflight, verify_reused_bootstrap
    from verl.trainer.evaluation.artifacts import file_sha256

    plan = read(origin / "plan.json")
    if read(origin / "controller-status.json")["state"] != "failed":
        raise ValueError("Retry requires a terminal failed controller")
    jobs = plan["jobs"]
    if set(replacements) != {job["experiment"] for job in jobs} or len(set(replacements.values())) != len(jobs):
        raise ValueError("Require a unique new registered ID for every original job")
    catalog = {f"E{row[0]}" for row in PAIRS}
    if not set(replacements.values()) <= catalog or set(replacements.values()) & set(replacements):
        raise ValueError("Retry IDs must be new and registered")
    for job in jobs:
        study = Path(job["study"])
        status = read(study / "status.json")
        if status["state"] not in ("queued", "failed", "stopped"):
            raise ValueError("Original jobs must be inactive")
        entered = any(stage["label"].endswith("-train") for stage in status.get("stages", []))
        if entered:
            # A missing metrics row alone cannot establish that no update occurred.
            log = (study / "logs" / f"{job['experiment']}-train.log").read_text()
            training = study / f"training-{job['optimizer']}"
            output = training / job["experiment"] / "evaluation"
            allowed = {output / "manifest.json", output / ".writer.lock"}
            unstarted = not training.exists()
            if (output / "manifest.json").exists():
                receipt = read(output / "manifest.json")
                unstarted = (
                    receipt["status"] == "active"
                    and not receipt["files"]
                    and all(path in allowed for path in training.rglob("*") if path.is_file())
                )
            if not unstarted or "create worker group dict_keys" not in log:
                raise ValueError("Training may have updated parameters; require verified full checkpoint recovery")
            if "Training Progress" in log or "step/optimizer" in log:
                raise ValueError("Refuse restarting training after possible parameter updates")
    root.mkdir(parents=True, exist_ok=False)
    source = freeze(repo, root / "source")
    code = {str(path.relative_to(source)): file_sha256(path) for path in source.rglob("*.py")}
    retried = []
    for job in jobs:
        old_study = Path(job["study"])
        original = read(old_study / "prepared.json")
        name = replacements[job["experiment"]]
        study = root / f"{name.lower()}-{job['optimizer']}-lr-{job['lr']:.8g}"
        (study / "configs").mkdir(parents=True)
        bank = Path(job["bootstrap"]) if job["pair"] == "E102" else root / "banks" / job["pair"]
        manifest = copy.deepcopy(original)
        manifest.update(
            source=str(source), code=code, configs={}, experiments={}, teacher_configs={}, reuse_bootstrap={}
        )
        manifest["retry_of"] = {name: str(old_study)}
        manifest["environment"] = rewrite(
            original["environment"], [(original["source"], str(source)), (str(old_study), str(study))]
        )
        manifest["environment"].update(
            MIRROROPD_RAY_PREWARM="1", RAY_raylet_start_wait_time_s="300", RAY_agent_register_timeout_ms="300000"
        )
        for field in ("experiments", "teacher_configs"):
            previous = Path(original[field][job["experiment"]])
            if file_sha256(previous) != original["configs"][str(previous)]:
                raise ValueError("Original configuration changed")
            reference = yaml.safe_load(previous.read_text())
            config = rewrite(
                reference, [(original["source"], str(source)), (str(old_study), str(study)), (job["experiment"], name)]
            )
            config["trainer"]["evaluation"]["datasets"] = reference["trainer"]["evaluation"]["datasets"]
            config["trainer"]["evaluation"]["bootstrap"] = str(bank)
            target = study / "configs" / f"{name}{'-teacher' if field == 'teacher_configs' else ''}.yaml"
            target.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
            preflight(target)
            manifest[field][name] = str(target)
            manifest["configs"][str(target)] = file_sha256(target)
            if job["pair"] == "E102":
                digest = file_sha256(bank / "manifest.json")
                verify_reused_bootstrap(OmegaConf.create(config), digest)
                manifest["reuse_bootstrap"][name] = dict(path=str(bank), sha256=digest)
        manifest["calibration"] = {name: original["calibration"][job["experiment"]]}
        manifest["models"] = [dict(model, experiment=name) for model in original["models"]]
        dump(study / "prepared.json", manifest)
        dump(study / "status.json", dict(state="queued", updated_at=time.time()))
        retried.append(dict(job, experiment=name, study=str(study), bootstrap=str(bank), retry_of=job["experiment"]))
    dump(root / "plan.json", dict(plan, jobs=retried, retry_of=str(origin)))


def metrics_summary(path):
    """Only completed JSONL events enter scoring; held-out never selects candidates."""
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line)["data"])
            except ValueError:
                continue
    output = dict(updates=max((int(row.get("step/optimizer", 0)) for row in rows), default=0), banks={})
    for bank in ("initial", "teacher"):
        for dataset in ("math", "development_math"):
            prefix = f"eval/rkl/public_{bank}/dataset/{dataset}/"
            points = {
                int(row["step/outer"]): row[prefix + "question_mean"] for row in rows if prefix + "question_mean" in row
            }
            if not points:
                continue
            last = max(points)
            indices = list(range(max(1, last - 9), last + 1)) if last else [0]
            tail = [(index, points[index]) for index in indices if index in points]
            finite = all(math.isfinite(value) for _, value in tail)
            if not finite:
                output["banks"][f"{bank}/{dataset}"] = dict(error="nonfinite RKL")
                continue
            mean = sum(value for _, value in tail) / len(tail)
            xmean = sum(index for index, _ in tail) / len(tail)
            variance = sum((index - xmean) ** 2 for index, _ in tail)
            slope = sum((index - xmean) * (value - mean) for index, value in tail) / variance if variance else None
            complete = all(
                any(
                    int(row.get("step/outer", -1)) == index
                    and row.get(prefix + "invalid_rows") == 0
                    and row.get(prefix + "valid_rows", 0) == (32 if dataset == "development_math" else 128)
                    for row in rows
                    if prefix + "question_mean" in row
                )
                for index in indices
            )
            output["banks"][f"{bank}/{dataset}"] = dict(
                outer=last,
                initial=points.get(0),
                latest=points[last],
                tail_mean=mean,
                slope=slope,
                relative_tail=mean / points[0] if points.get(0, 0) > 0 else None,
                tail_std=(sum((value - mean) ** 2 for _, value in tail) / len(tail)) ** 0.5,
                coverage_complete=complete,
                final_window=last == 40 and len(tail) == 10,
                still_falling=bool(slope is not None and mean > 0 and -10 * slope / mean > 0.01),
            )
    updates = [row for row in rows if "optim/actor/update_applied" in row]
    output["skipped"] = sum(row.get("optim/actor/update_skipped", 0) for row in updates)
    output["nonfinite"] = sum(row.get("optim/actor/loss_nonfinite", 0) for row in updates)
    return output


def report(root):
    plan = read(root / "plan.json")
    records, selected = [], {}
    for job in [*plan["jobs"], *plan.get("external_jobs", [])]:
        study = Path(job["study"])
        state = read(study / "status.json") if (study / "status.json").exists() else dict(state="queued")
        summary = metrics_summary(study / f"training-{job['optimizer']}" / job["experiment"] / "metrics.jsonl")
        records.append(dict(**job, state=state["state"], error=state.get("error"), **summary))
    for pair in {job["pair"] for job in records}:
        for kind in ("muon", "adamw"):
            candidates = []
            for job in records:
                if job["pair"] != pair or job["optimizer"] != kind or job["state"] != "complete":
                    continue
                banks = [job["banks"].get(f"{bank}/development_math", {}) for bank in ("initial", "teacher")]
                if not all(
                    bank.get("final_window") and bank.get("coverage_complete") and bank.get("relative_tail") is not None
                    for bank in banks
                ):
                    continue
                candidates.append((sum(bank["relative_tail"] for bank in banks) / 2, job))
            if candidates:
                score, winner = min(candidates, key=lambda item: item[0])
                selected[f"{pair}/{kind}"] = dict(
                    experiment=winner["experiment"],
                    lr=winner["lr"],
                    score=score,
                    provisional=True,
                    still_falling=any(
                        winner["banks"][f"{b}/development_math"]["still_falling"] for b in ("initial", "teacher")
                    ),
                )
    result = dict(time=time.time(), jobs=records, development_selection=selected)
    dump(root / "analysis.json", result)
    return result


def _run_lane(root, broker, plan, index):
    """Execute queued jobs only; completed receipts permit explicit lane recovery."""
    from omegaconf import OmegaConf

    from examples.mirroropd.tools.absorption_queue import verify_reused_bootstrap
    from verl.trainer.evaluation.artifacts import file_sha256

    for job in [item for item in plan["jobs"] if item["lane"] == index]:
        state_path = Path(job["study"]) / "status.json"
        if state_path.exists():
            state = read(state_path)["state"]
            if state == "complete" and (Path(job["study"]) / f"{job['experiment']}-complete.json").exists():
                continue
            if state != "queued":
                raise ValueError(f"Refuse relaunching non-queued job: {job['experiment']} ({state})")
        if (root / "STOP").exists():
            return
        study = Path(job["study"])
        manifest = read(study / "prepared.json")
        bank = Path(job["bootstrap"])
        if (bank / "manifest.json").exists():
            digest = file_sha256(bank / "manifest.json")
            verify_reused_bootstrap(OmegaConf.load(manifest["experiments"][job["experiment"]]), digest)
            manifest["reuse_bootstrap"] = {job["experiment"]: dict(path=str(bank), sha256=digest)}
            dump(study / "prepared.json", manifest)
        command = [
            sys.executable,
            "-m",
            "examples.mirroropd.tools.absorption_queue",
            "run",
            "--study",
            str(study),
            "--broker",
            str(broker),
        ]
        with (study / "supervisor.log").open("x") as stream:
            child = subprocess.Popen(
                command,
                cwd=manifest["source"],
                env=manifest["environment"],
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
            dump(study / "supervisor.json", dict(pid=child.pid, command=command))
            while child.poll() is None:
                if (root / "STOP").exists():
                    (study / "STOP").touch()
                time.sleep(5)
            if child.returncode:
                raise RuntimeError(f"{job['experiment']} failed; preserve source and checkpoints")
        report(root)


def run_lane(root, broker, plan, index):
    """Publish lane failures immediately, independently of the other healthy lane."""
    try:
        _run_lane(root, broker, plan, index)
    except BaseException as error:
        dump(root / f"lane-{index}-error.json", dict(time=time.time(), error=repr(error)))
        raise


def run(root, broker):
    with (root / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "launch.json").exists():
            raise ValueError("Already launched; preserve evidence and use an explicit retry")
        plan = read(root / "plan.json")
        dump(root / "launch.json", dict(pid=os.getpid(), time=time.time()))
        dependency = Path(plan["dependency"])
        dump(root / "controller-status.json", dict(state="waiting_for_dependency", dependency=str(dependency)))
        while True:
            if (root / "STOP").exists():
                raise InterruptedError("Campaign STOP")
            status = read(dependency / "controller-status.json")
            if status["state"] == "finished":
                break
            if status["state"] in ("failed", "preflight_failed"):
                raise RuntimeError("Preceding sweep failed; repair it before continuing")
            time.sleep(20)

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run_lane, root, broker, plan, index) for index in (0, 1)]
            next_check = 0
            while not all(future.done() for future in futures):
                if time.time() >= next_check:
                    report(root)
                    dump(root / "controller-status.json", dict(state="running", time=time.time()))
                    next_check = time.time() + plan["check_interval_seconds"]
                time.sleep(10)
            failures = [str(future.exception()) for future in futures if future.exception()]
            report(root)
            dump(
                root / "controller-status.json",
                dict(state="failed" if failures else "finished", time=time.time(), errors=failures),
            )
            if failures:
                raise RuntimeError(failures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "report", "retry", "run-lane"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--references", type=Path)
    parser.add_argument("--dependency", type=Path)
    parser.add_argument("--probes", type=Path, help="Optional JSON mapping new E numbers to [pair, kind, lr]")
    parser.add_argument("--from-root", type=Path)
    parser.add_argument("--replacements", type=Path, help="JSON mapping original IDs to new registered IDs")
    parser.add_argument("--broker", type=Path)
    parser.add_argument("--lane", type=int, choices=(0, 1))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    if args.action == "prepare":
        if args.references is None or args.dependency is None:
            parser.error("prepare requires --references and --dependency")
        prepare(
            args.repo.resolve(),
            args.root.resolve(),
            read(args.references),
            args.dependency.resolve(),
            read(args.probes) if args.probes else None,
        )
    elif args.action == "retry":
        if args.from_root is None or args.replacements is None:
            parser.error("retry requires --from-root and --replacements")
        retry(args.repo.resolve(), args.root.resolve(), args.from_root.resolve(), read(args.replacements))
    elif args.action == "run-lane":
        if args.lane is None or args.broker is None:
            parser.error("run-lane requires --lane and --broker")
        with (args.root / f"lane-{args.lane}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run_lane(args.root.resolve(), args.broker.resolve(), read(args.root / "plan.json"), args.lane)
    elif args.action == "run":
        if args.broker is None:
            parser.error("run requires --broker")
        run(args.root.resolve(), args.broker.resolve())
    else:
        report(args.root.resolve())


if __name__ == "__main__":
    main()
