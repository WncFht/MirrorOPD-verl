# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Conditional, bounded MDPO sweep after the paired pilot has restored GPUs 0-3.

The gate is a budget-allocation heuristic on fixed pilot questions, not a test
of statistical significance. Preparation and unsuccessful decisions use no GPU.
"""

import argparse
import fcntl
import importlib.util
import json
import math
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

from examples.mirroropd.research.core.io import dump as write
from examples.mirroropd.research.core.paths import REPO
from examples.mirroropd.research.studies.termination.run_e54_recovery_eval import execute

ARMS = [(f"k05-eta{label}", 0.5, eta) for label, eta in [("005", 0.05), ("01", 0.1), ("02", 0.2), ("03", 0.3)]] + [
    ("k1-eta005", 1.0, 0.05),
    ("k1-eta03", 1.0, 0.3),
]
GATE = {
    "correct_total_delta_min": 0,
    "correct_stopped_delta_min": 4,
    "cap_count_delta_max": -4,
    "mean_length_delta_max": 0,
    "repeat8_mean_delta_max": 0,
    "correct_per_dataset_delta_min": -2,
    "interpretation": "Resource-allocation screen on 128 pilot questions, not statistical significance",
}


def decide(summary):
    indexed = {}
    for row in summary["evaluation"]:
        key = (row["arm"], row["dataset"])
        if key in indexed:
            raise ValueError(f"Duplicate evaluation row: {key}")
        indexed[key] = row
    expected = {
        (arm, dataset) for arm in ("k0-eta01", "k1-eta01", "k0-eta02", "k1-eta02") for dataset in ("math", "dapo")
    }
    if set(indexed) != expected or summary.get("paired_training_prompts") is not True:
        raise ValueError("Expected all four paired arms on both fixed datasets")
    for row in indexed.values():
        values = [row[k] for k in ("accuracy", "mean_length", "repeat8_mean", "correct_and_stopped", "cap_count")]
        if row["responses"] != 64 or not all(math.isfinite(v) for v in values):
            raise ValueError("Incomplete or nonfinite pilot metrics")
        correct = row["accuracy"] * 64
        if not (0 <= row["accuracy"] <= 1 and abs(correct - round(correct)) < 1e-7):
            raise ValueError("Invalid accuracy or response denominator")
        if not (0 <= row["repeat8_mean"] <= 1 and 0 <= row["mean_length"] <= 31744):
            raise ValueError("Invalid repeat or length metric")
        if not (0 <= row["cap_count"] <= 64 and 0 <= row["correct_and_stopped"] <= min(correct, 64 - row["cap_count"])):
            raise ValueError("Invalid stopping counts")

    comparisons = []
    for eta in ("01", "02"):
        candidate = [indexed[(f"k1-eta{eta}", d)] for d in ("math", "dapo")]
        control = [indexed[(f"k0-eta{eta}", d)] for d in ("math", "dapo")]
        correct_deltas = [round((a["accuracy"] - b["accuracy"]) * 64) for a, b in zip(candidate, control, strict=True)]
        deltas = {
            key: sum(a[key] - b[key] for a, b in zip(candidate, control, strict=True))
            for key in ("correct_and_stopped", "cap_count", "mean_length", "repeat8_mean")
        }
        deltas["mean_length"] /= 2
        deltas["repeat8_mean"] /= 2
        passed = (
            sum(correct_deltas) >= GATE["correct_total_delta_min"]
            and min(correct_deltas) >= GATE["correct_per_dataset_delta_min"]
            and deltas["correct_and_stopped"] >= GATE["correct_stopped_delta_min"]
            and deltas["cap_count"] <= GATE["cap_count_delta_max"]
            and deltas["mean_length"] <= GATE["mean_length_delta_max"]
            and deltas["repeat8_mean"] <= GATE["repeat8_mean_delta_max"] + 1e-12
        )
        comparisons.append(
            {"eta": float("0." + eta[1:]), "eligible": passed, "correct_deltas_math_dapo": correct_deltas, **deltas}
        )
    return {"launch": any(c["eligible"] for c in comparisons), "comparisons": comparisons, "gate": GATE}


def prepare(parent, root):
    repo = REPO
    assert parent.parent == repo / "outputs" and root.parent == parent.parent and parent != root
    root.mkdir(exist_ok=False)
    source = root / "source"
    # The model/loss/trainer code is exactly the pilot snapshot, not today's worktree.
    shutil.copytree(
        parent / "source", source, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "wandb", "node_modules")
    )
    for name in (
        "examples/mirroropd/research/studies/termination/run_e54_topd_mdpo.py",
        "examples/mirroropd/research/studies/termination/queue_e54_mdpo_sweep.py",
        "tests/trainer/distillation/test_mdpo_termination_on_gpu.py",
    ):
        shutil.copy2(repo / name, source / name)
    shutil.copy2(parent / "controller.py", root / "controller.py")
    pilot = json.loads((parent / "protocol.json").read_text())
    original = json.loads((parent / "k1-eta01/launch.json").read_text())
    jobs = []
    gpu_check = root / "gpu-check"
    gpu_check.mkdir()
    check = json.loads((parent / "gpu-check/launch.json").read_text())
    check = json.loads(json.dumps(check).replace(str(parent), str(root)))
    write(gpu_check / "launch.json", check)
    jobs.append({"preset": "gpu-check", "directory": str(gpu_check)})
    for label, kappa, eta in ARMS:
        arm = root / label
        arm.mkdir()
        run_id = "s" + uuid.uuid4().hex[:7]

        def translate(value, label=label, run_id=run_id):
            return (
                value.replace(str(parent / "k1-eta01"), str(root / label))
                .replace(str(parent), str(root))
                .replace(original["environment"]["WANDB_RUN_ID"], run_id)
                .replace(parent.name, root.name)
            )

        env = {k: translate(v) for k, v in original["environment"].items()}
        env["EXPERIMENT_NAME"] = root.name + "-" + label
        command = [translate(v) for v in original["command"]]
        command[command.index("--kappa") + 1] = str(kappa)
        command[command.index("--eta") + 1] = str(eta)
        write(arm / "launch.json", {"environment": env, "command": command, "cwd": str(source)})
        execute(command, arm / "config.log", os.environ | env | {"CONFIG_ONLY": "1", "CUDA_VISIBLE_DEVICES": ""}, 120)
        jobs.append({"preset": label, "directory": str(arm), "wandb_run_id": run_id})
    evaluation = root / "evaluation"
    evaluation.mkdir()
    for dataset in ("math", "dapo"):
        shutil.copy2(parent / f"evaluation/prompts-{dataset}.json", evaluation / f"prompts-{dataset}.json")
    launch = json.loads((parent / "evaluation/launch.json").read_text())
    launch = json.loads(json.dumps(launch).replace(str(parent), str(root)))
    write(evaluation / "launch.json", launch)
    jobs.append({"preset": "evaluation", "directory": str(evaluation)})
    write(root / "plan.json", {"jobs": jobs, "restore_physical_gpus": [0, 1, 2, 3]})
    write(
        root / "protocol.json",
        pilot
        | {
            "arms": [{"label": label, "kappa": kappa, "eta": eta} for label, kappa, eta in ARMS],
            "conditional_parent": str(parent),
            "gate": GATE,
            "source": "Pilot model/trainer/loss snapshot; only generalized evaluation and CUDA coverage added",
            "existing_grid_points": ["k1-eta01", "k1-eta02"],
            "selection_scope": "Same 64+64 pilot questions reused; winner requires independent confirmation",
        },
    )
    print(json.dumps({"prepared": str(root), "new_training_arms": len(ARMS), "gate": GATE}, indent=2))


def watch(root):
    with (root / "conditional.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (root / "started.json").exists() and not (root / "conditional-decision.json").exists()
        parent = Path(json.loads((root / "protocol.json").read_text())["conditional_parent"])
        spec = importlib.util.spec_from_file_location("pilot_controller", parent / "controller.py")
        controller = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(controller)
        write(root / "conditional-state.json", {"phase": "waiting", "pid": os.getpid(), "dependency": str(parent)})
        while True:
            if (parent / "recovered.json").exists():
                state = json.loads((parent / "status.json").read_text())
                owner = json.loads((parent / "started.json").read_text())["supervisor"]
                if state["phase"] == "complete":
                    break
                if state["phase"] == "failed" or not controller.alive(owner):
                    write(
                        root / "conditional-state.json", {"phase": "skipped", "reason": "Pilot failed; no GPU action"}
                    )
                    return
            time.sleep(30)
        try:
            jobs = json.loads((parent / "plan.json").read_text())["jobs"]
            if not all(json.loads((Path(j["directory"]) / "exit.json").read_text())["exit_code"] == 0 for j in jobs):
                raise ValueError("Pilot job failure")
            decision = decide(json.loads((parent / "evaluation/summary.json").read_text()))
        except (OSError, KeyError, TypeError, ValueError) as exc:
            write(root / "conditional-state.json", {"phase": "skipped", "reason": f"Invalid pilot evidence: {exc}"})
            return
        write(root / "conditional-decision.json", decision)
        if not decision["launch"]:
            write(
                root / "conditional-state.json",
                {"phase": "skipped", "reason": "Pilot did not meet predeclared screen; no GPU action"},
            )
            return
        write(root / "conditional-state.json", {"phase": "launching", "pid": os.getpid()})
        # The supervisor starts its independent recovery guard before releasing occupation.
        os.execv(sys.executable, [sys.executable, str(root / "controller.py"), "run"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "watch"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        if args.parent is None:
            parser.error("prepare requires --parent")
        prepare(args.parent.resolve(), args.root.resolve())
    else:
        watch(args.root.resolve())


if __name__ == "__main__":
    main()
