# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Run immutable optimizer probes from an admitted pair and shared fixed banks."""

import argparse
import copy
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from examples.mirroropd.launchers.absorption import MUON_LR_PRESETS
from examples.mirroropd.research.core.io import dump, read
from examples.mirroropd.research.runtime.source import freeze


def prepare(repo, root, origin):
    """Reuse the exact accepted data/banks; change only optimizer and output identity."""
    import yaml
    from omegaconf import OmegaConf

    from examples.mirroropd.tools.absorption_queue import preflight, verify_reused_bootstrap
    from verl.trainer.evaluation.artifacts import file_sha256

    original = read(origin / "prepared.json")
    if len(original["experiments"]) != 1:
        raise ValueError("Require a single admitted reference experiment")
    old_name = next(iter(original["experiments"]))
    for path, digest in original["configs"].items():
        if file_sha256(path) != digest:
            raise ValueError("Reference configuration changed")
    reference = yaml.safe_load(Path(original["experiments"][old_name]).read_text())
    bootstrap = Path(reference["trainer"]["evaluation"]["bootstrap"])
    digest = file_sha256(bootstrap / "manifest.json")
    verify_reused_bootstrap(OmegaConf.create(reference), digest)
    root.mkdir(parents=True, exist_ok=False)
    source = freeze(repo, root / "source")
    code = {str(p.relative_to(source)): file_sha256(p) for p in source.rglob("*.py")}
    lanes = []
    for index, (number, lr) in enumerate(MUON_LR_PRESETS.items()):
        name = f"E{number}"
        study = root / f"{name.lower()}-muon-lr-{lr:.8g}"
        study.mkdir()
        (study / "configs").mkdir()
        (study / "preflight").mkdir()
        shutil.copytree(origin / "questions", study / "questions")
        for path in (study / "questions").glob("*.json"):
            path.write_text(path.read_text().replace(str(origin), str(study)))
        manifest = copy.deepcopy(original)
        manifest.update(
            experiments={},
            teacher_configs={},
            configs={},
            source=str(source),
            code=code,
            devices=[0, 1, 2, 3] if index % 2 == 0 else [4, 5, 6, 7],
            optimizer="muon",
            calibration={name: dict(lr=lr, aux_lr=1e-6, status="full_budget_probe")},
            reuse_bootstrap={name: dict(path=str(bootstrap), sha256=digest)},
            reference=str(origin),
        )
        manifest.pop("retry_of", None)
        env = {
            k: v.replace(original["source"], str(source))
            .replace(str(origin), str(study))
            .replace(old_name, name)
            .replace("training-sgd", "training-muon")
            for k, v in original["environment"].items()
        }
        env.update(
            PYTHONPATH=str(source), CUDA_VISIBLE_DEVICES=",".join(map(str, manifest["devices"])), WANDB_MODE="offline"
        )
        manifest["environment"] = env
        manifest["models"] = [dict(m, experiment=name) for m in original["models"]]
        for field in ("experiments", "teacher_configs"):
            text = Path(original[field][old_name]).read_text()
            text = text.replace(original["source"], str(source)).replace(str(origin), str(study))
            text = text.replace(old_name, name).replace("training-sgd", "training-muon")
            config = yaml.safe_load(text)
            config["trainer"]["evaluation"]["bootstrap"] = str(bootstrap)
            config["trainer"]["project_name"] = "mirroropd-optimizer-platform"
            config["trainer"]["experiment_name"] = f"{name}-muon-lr-{lr:.8g}"
            config["trainer"]["logger"] = ["console", "file", "wandb"]
            config["actor_rollout_ref"]["actor"]["optim"].update(
                optimizer="MuonAdamW",
                optimizer_impl="verl.utils.fsdp_optimizers",
                lr=lr,
                weight_decay=0,
                override_optimizer_config=dict(
                    aux_lr=1e-6,
                    momentum=0.95,
                    nesterov=True,
                    ns_steps=5,
                    adjust_lr_fn="match_rms_adamw",
                    distributed_mode="owner",
                ),
            )
            config["trainer"]["default_local_dir"] = str(study / "training-muon" / name / "runtime")
            config["trainer"]["evaluation"]["output"] = str(study / "training-muon" / name / "evaluation")
            path = study / "configs" / f"{name}{'-teacher' if field == 'teacher_configs' else ''}.yaml"
            path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
            verify_reused_bootstrap(OmegaConf.create(config), digest)
            preflight(path)
            manifest[field][name] = str(path)
            manifest["configs"][str(path)] = file_sha256(path)
        # Any residual training-sgd path must belong to the read-only reference bootstrap.
        for field in ("experiments", "teacher_configs"):
            config = yaml.safe_load(Path(manifest[field][name]).read_text())
            assert config["data"]["train_files"] == [str(study / "questions/train.parquet")]
        dump(study / "prepared.json", manifest)
        lanes.append(
            dict(experiment=name, study=str(study), lr=lr, aux_lr=1e-6, devices=manifest["devices"], lane=index % 2)
        )
    dump(
        root / "plan.json",
        dict(
            optimizer="MuonAdamW",
            outer=40,
            reuse=16,
            batch=64,
            reference=str(origin),
            bank=str(bootstrap),
            lanes=lanes,
            selection="development banks; held-out is reporting only",
        ),
    )


def observe(root, lanes, processes, state):
    records = []
    for lane in lanes:
        study = Path(lane["study"])
        status = read(study / "status.json") if (study / "status.json").exists() else dict(state="queued")
        latest = {}
        path = study / "training-muon" / lane["experiment"] / "metrics.jsonl"
        if path.exists():
            for line in path.read_text().splitlines()[-3:]:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                latest = dict(
                    step=row.get("step"),
                    data={k: v for k, v in row["data"].items() if k.startswith("step/") or k.endswith("question_mean")},
                )
        records.append(dict(**lane, state=status["state"], error=status.get("error"), latest=latest))
    event = dict(time=time.time(), state=state, runs=records, active={str(p.pid): p.poll() for p in processes})
    dump(root / "controller-status.json", event)
    with (root / "hourly-observations.jsonl").open("a") as file:
        file.write(json.dumps(event, ensure_ascii=False) + "\n")


def run(root, broker):
    """Two sequential four-GPU lanes; native queue owns all resource leases."""
    with (root / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "launch.json").exists():
            raise ValueError("Already launched; never silently restart an optimizer sweep")
        lanes = read(root / "plan.json")["lanes"]
        processes = []
        dump(root / "launch.json", dict(pid=os.getpid(), started_at=time.time()))

        def lane_run(index):
            for lane in [item for item in lanes if item["lane"] == index]:
                if (root / "STOP").exists():
                    raise InterruptedError("Sweep STOP exists")
                study = Path(lane["study"])
                manifest = read(study / "prepared.json")
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
                with (study / "supervisor.log").open("x") as log:
                    child = subprocess.Popen(
                        command,
                        cwd=manifest["source"],
                        env=manifest["environment"],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                    processes.append(child)
                    dump(study / "supervisor.json", dict(pid=child.pid, command=command))
                    code = child.wait()
                if code:
                    (root / "STOP").touch()
                    raise RuntimeError(f"{lane['experiment']} failed ({code}); inspect original logs")

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(lane_run, index) for index in range(2)]
            next_observation = 0
            while not all(future.done() for future in futures):
                if (root / "STOP").exists():
                    for lane in lanes:
                        (Path(lane["study"]) / "STOP").touch()
                if time.time() >= next_observation:
                    observe(root, lanes, processes, "running")
                    next_observation = time.time() + 3600
                time.sleep(10)
            failed = any(future.exception() is not None for future in futures)
            observe(root, lanes, processes, "failed" if failed else "finished")
            for future in futures:
                future.result()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--origin", type=Path)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--broker", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        if args.origin is None:
            parser.error("prepare requires --origin")
        prepare(args.repo.resolve(), args.root.resolve(), args.origin.resolve())
    else:
        if args.broker is None:
            parser.error("run requires --broker")
        run(args.root.resolve(), args.broker.resolve())


if __name__ == "__main__":
    main()
