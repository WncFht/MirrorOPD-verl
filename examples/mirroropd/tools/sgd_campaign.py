# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Calibrate E94–E114 on disjoint development prompts, then run both full lanes.

Every candidate starts from the original student. Development banks are shared
only within a pair; permanent held-out banks never select the learning rate.
Nonzero native-job exits stop the campaign and preserve the original evidence.
"""

import argparse
import copy
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from examples.mirroropd.research.core.io import dump, read
from examples.mirroropd.research.runtime.processes import identity as process_identity
from examples.mirroropd.research.studies.absorption_dynamics.sgd_protocol import (
    candidate_score,
    development_split,
    next_lr,
)
from examples.mirroropd.tools import absorption_queue as queue

INITIAL_LRS = (0.003, 0.01, 0.03)
CALIBRATION_OUTERS = 4


def prepare(repo, root):
    import pyarrow as pa
    import pyarrow.parquet as pq

    root.mkdir(parents=True, exist_ok=False)
    original = repo / "examples/mirroropd/research-evidence/qwen-prefix-six/prepared-03"
    training = pq.read_table(original / "train.parquet").to_pylist()
    cal_train, dev = development_split(training, read(original / "heldout-questions.json"))
    questions = root / "development"
    questions.mkdir()
    pq.write_table(pa.Table.from_pylist(cal_train), questions / "train.parquet")
    pq.write_table(
        pa.Table.from_pylist(
            [dict(id=f"sgd-dev-{i:03d}", prompt=json.dumps(r["prompt"], ensure_ascii=False)) for i, r in enumerate(dev)]
        ),
        questions / "math.parquet",
    )
    dump(
        questions / "split.json",
        dict(
            full_training_prefix=2560,
            calibration_training_ids=[r["extra_info"]["index"] for r in cal_train],
            development_ids=[r["extra_info"]["index"] for r in dev],
            exact_normalized_text_overlap=0,
            final_heldout_used_for_selection=False,
        ),
    )
    names = [f"E{i}" for i in range(94, 115)]
    # These are configuration templates, never accepted learning-rate selections.
    records = {name: dict(lr=INITIAL_LRS[1], status="unselected_template") for name in names}
    queue.prepare(repo, root / "templates", names, optimizer="sgd", calibration=records)
    dump(
        root / "campaign.json",
        dict(
            initial_lrs=INITIAL_LRS,
            max_edge_expansions=2,
            calibration_outer=CALIBRATION_OUTERS,
            reuse=16,
            batch=64,
            response_cap=7168,
            selection="mean relative development RKL reduction, both fixed banks, outers 3 and 4",
            full_outer=40,
            full_updates=640,
            experiments=names,
            caveat="SGD weight_decay=0; historical AdamW weight_decay=0.01, not a pure optimizer-only contrast",
        ),
    )


def materialize_trial(root, name, lr, devices, trial, bootstrap=None):
    """Freeze one ordinary native queue job with its own logs and outputs."""
    import yaml

    from verl.trainer.evaluation.artifacts import file_sha256

    original_root = root / "templates"
    original = read(original_root / "prepared.json")
    manifest = copy.deepcopy(original)
    trial.mkdir(parents=True, exist_ok=False)
    manifest.update(devices=list(devices), experiments={}, teacher_configs={}, configs={}, reuse_bootstrap={})
    manifest["environment"].update(CUDA_VISIBLE_DEVICES=",".join(map(str, devices)))
    manifest["settings"].update(outer=CALIBRATION_OUTERS, snapshots=[])
    manifest["calibration"] = {name: dict(lr=lr, status="candidate")}
    content = Path(original["experiments"][name]).read_text().replace(str(original_root), str(trial))
    config = yaml.safe_load(content.replace(str(trial / "source"), original["source"]))
    config["data"]["train_files"] = [str(root / "development/train.parquet")]
    config["trainer"].update(total_training_steps=CALIBRATION_OUTERS, save_freq=-1)
    config["trainer"]["project_name"] = "mirroropd-sgd-calibration"
    config["trainer"]["experiment_name"] = f"{name}-sgd-lr-{lr:.8g}"
    config["actor_rollout_ref"]["actor"]["optim"]["lr"] = lr
    evaluation = config["trainer"]["evaluation"]
    evaluation.update(
        bootstrap=str(trial / "bootstrap" / name) if bootstrap is None else bootstrap["path"],
        final_export=False,
        verify_export=False,
        export_outers=[],
        datasets=[
            dict(
                name="math",
                domain="math",
                path=str(root / "development/math.parquet"),
                verifier="math",
                n=1,
                protocol="public",
                panel_ids=[f"sgd-dev-{i:03d}" for i in range(32)],
            )
        ],
    )
    if bootstrap:
        manifest["reuse_bootstrap"][name] = bootstrap
    configs = trial / "configs"
    configs.mkdir()
    path = configs / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    manifest["experiments"][name] = str(path)
    teacher = yaml.safe_load(Path(original["teacher_configs"][name]).read_text())
    config["actor_rollout_ref"]["rollout"]["max_num_seqs"] = teacher["actor_rollout_ref"]["rollout"]["max_num_seqs"]
    config["trainer"]["evaluation"]["generation_concurrency"] = teacher["trainer"]["evaluation"][
        "generation_concurrency"
    ]
    path = configs / f"{name}-teacher.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    manifest["teacher_configs"][name] = str(path)
    manifest["configs"] = {
        p: file_sha256(p) for p in (*manifest["experiments"].values(), *manifest["teacher_configs"].values())
    }
    dump(trial / "prepared.json", manifest)
    return manifest


def native_job(study, broker, root):
    """The existing queue owns GPU leases; the campaign itself owns no GPUs."""
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
    env = dict(os.environ, PYTHONPATH=read(root / "templates/prepared.json")["source"])
    with (study / "supervisor.log").open("x") as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        dump(study / "supervisor.json", dict(pid=child.pid, command=command))
        while child.poll() is None:
            if (root / "STOP").exists():
                (study / "STOP").touch()
            time.sleep(2)
        if child.returncode:
            raise RuntimeError(f"Native job failed ({child.returncode}): {study}; inspect original logs")


def adopt_job(study, root):
    """Await an existing native supervisor, checking PID/start-time identity."""
    record = read(study / "process.json")
    while not (study / "exit.json").exists():
        if process_identity(record["pid"]) != record["identity"]:
            raise RuntimeError(f"Native supervisor exited without an exit receipt: {study}")
        if (root / "STOP").exists():
            (study / "STOP").touch()
        time.sleep(2)
    if read(study / "exit.json")["returncode"]:
        raise RuntimeError(f"Refuse restarting a failed native job: {study}")


def calibrate_lane(root, names, devices, broker):
    from verl.trainer.evaluation.artifacts import file_sha256

    label = "front4" if devices[0] == 0 else "back4"
    selected = {}
    for name in names:
        pair_root = root / "calibration" / name
        pair_root.mkdir(parents=True, exist_ok=True)
        bootstrap = None
        results = []
        candidates = list(INITIAL_LRS)
        expansions = 0
        while candidates:
            if (root / "STOP").exists():
                raise InterruptedError("Campaign STOP exists")
            lr = candidates.pop(0)
            trial = pair_root / f"trial-{len(results):02d}"
            dump(root / f"{label}-status.json", dict(state="calibrating", experiment=name, lr=lr, trial=str(trial)))
            if trial.exists():
                import yaml

                from verl.trainer.evaluation.artifacts import file_sha256

                existing = read(trial / "prepared.json")
                config = yaml.safe_load(Path(existing["experiments"][name]).read_text())
                if existing["devices"] != list(devices) or config["actor_rollout_ref"]["actor"]["optim"]["lr"] != lr:
                    raise ValueError("Existing trial has a different LR or GPU allocation")
                for path, digest in existing["configs"].items():
                    if file_sha256(path) != digest:
                        raise ValueError("Existing trial configuration changed")
                adopt_job(trial, root)
            else:
                materialize_trial(root, name, lr, devices, trial, bootstrap)
                native_job(trial, broker, root)
            evaluation = trial / "training-sgd" / name / "evaluation"
            events = [
                read(evaluation / f"events/outer-{outer:03d}/manifest.json")["metadata"]["metrics"]
                for outer in range(CALIBRATION_OUTERS + 1)
            ]
            results.append(dict(lr=lr, score=candidate_score(events), trial=str(trial)))
            dump(pair_root / "results.json", results)
            if bootstrap is None:
                path = trial / "bootstrap" / name
                bootstrap = dict(path=str(path), sha256=file_sha256(path / "manifest.json"))
            if not candidates:
                extra = next_lr(results, expansions)
                if extra is not None:
                    candidates.append(extra)
                    expansions += 1
        best = max(results, key=lambda row: (row["score"], -row["lr"]))
        selected[name] = dict(
            **best,
            status="selected",
            criterion=read(root / "campaign.json")["selection"],
            edge_winner=best["lr"] in (min(r["lr"] for r in results), max(r["lr"] for r in results)),
        )
        dump(pair_root / "selected.json", selected[name])
        dump(root / f"{label}-selected.json", selected)
    dump(root / f"{label}-status.json", dict(state="calibrated", experiments=list(selected)))
    return selected


def prepare_full(root, selected):
    """Populate measured LRs before splitting the original full-budget templates."""
    import yaml

    from verl.trainer.evaluation.artifacts import file_sha256

    origin = root / "templates"
    full = root / "full-prepared"
    full.mkdir()
    shutil.copytree(origin / "source", full / "source", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(origin / "questions", full / "questions")
    datasets = full / "questions/datasets.json"
    datasets.write_text(datasets.read_text().replace(str(origin), str(full)))
    (full / "configs").mkdir()
    (full / "preflight").mkdir()
    manifest = read(origin / "prepared.json")
    for field in ("experiments", "teacher_configs"):
        for name, previous in manifest[field].items():
            config = yaml.safe_load(Path(previous).read_text().replace(str(origin), str(full)))
            config["actor_rollout_ref"]["actor"]["optim"]["lr"] = selected[name]["lr"]
            config["trainer"].update(save_freq=5, max_actor_ckpt_to_keep=2)
            target = full / "configs" / Path(previous).name
            target.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
            manifest[field][name] = str(target)
    for name in selected:
        shutil.copy2(origin / "preflight" / f"{name}.log", full / "preflight" / f"{name}.log")
    manifest["calibration"] = selected
    manifest["source"] = str(full / "source")
    manifest["code"] = {str(p.relative_to(full / "source")): file_sha256(p) for p in (full / "source").rglob("*.py")}
    manifest["environment"] = {k: v.replace(str(origin), str(full)) for k, v in manifest["environment"].items()}
    manifest["configs"] = {
        p: file_sha256(p) for p in (*manifest["experiments"].values(), *manifest["teacher_configs"].values())
    }
    dump(full / "prepared.json", manifest)
    queue.partition(full, root / "full")


def run(root, broker, *, resume=False):
    manifest = read(root / "templates/prepared.json")
    if (root / "status.json").exists() and not resume:
        raise ValueError("Existing campaign requires --resume; trials will be adopted, never restarted")
    if (root / "STOP").exists():
        raise InterruptedError("Remove a diagnosed STOP explicitly before resuming")
    if (root / "full").exists() or (root / "full-prepared").exists():
        raise ValueError("Full training was already prepared; inspect and use its native supervisors directly")
    front, back = queue.partition_numbers(manifest["experiments"])
    dump(root / "status.json", dict(state="calibrating", pid=os.getpid(), started_at=time.time()))
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(calibrate_lane, root, [f"E{n}" for n in ids], devices, broker)
                for ids, devices in ((front, (0, 1, 2, 3)), (back, (4, 5, 6, 7)))
            ]
            # Stop the peer lane after a failed worker; running native jobs observe STOP.
            selected = {}
            try:
                for future in as_completed(futures):
                    selected.update(future.result())
            except BaseException:
                (root / "STOP").touch()
                raise
        dump(root / "selected-lrs.json", selected)
        prepare_full(root, selected)
        dump(
            root / "status.json",
            dict(state="full_training", pid=os.getpid(), selected_lrs=str(root / "selected-lrs.json")),
        )
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(native_job, root / "full" / f"absorption-{label}", broker, root)
                for label in ("front4", "back4")
            ]
            try:
                for future in as_completed(futures):
                    future.result()
            except BaseException:
                (root / "STOP").touch()
                raise
        dump(root / "status.json", dict(state="complete", finished_at=time.time(), experiments=21))
    except BaseException as error:
        dump(root / "status.json", dict(state="failed", error=repr(error), finished_at=time.time()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--broker", type=Path)
    parser.add_argument("--resume", action="store_true", help="Adopt existing successful/running calibration jobs")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "prepare":
        prepare(args.repo.resolve(), root)
    else:
        with (root / "campaign.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(root, args.broker.resolve(), resume=args.resume)


if __name__ == "__main__":
    main()
