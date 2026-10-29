# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Wait for bounded training and GPU recovery, then evaluate four checkpoints."""

import argparse
import concurrent.futures
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def execute(command, log_path, env, timeout):
    with log_path.open("w") as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            raise
    if code:
        raise RuntimeError(f"Command failed with code {code}; see {log_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    evaluation = root / "evaluation"
    old = root.parent / "e54-rope-debug-20261028-hppk13/run"
    while not (root / "recovered.json").exists():
        time.sleep(10)
    jobs = json.loads((root / "plan.json").read_text())["jobs"]
    assert all(json.loads((Path(j["directory"]) / "exit.json").read_text())["exit_code"] == 0 for j in jobs)
    cpu_env = os.environ | {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2"}
    merge_specs = [("continue-r16", 15), ("branch-r4", 14)]
    for name, step in merge_specs:
        command = [
            sys.executable,
            "-m",
            "verl.model_merger",
            "merge",
            "--backend",
            "fsdp",
            "--local_dir",
            str(root / name / f"training/checkpoints/global_step_{step}/actor"),
            "--target_dir",
            str(evaluation / f"checkpoint-{name}"),
        ]
        execute(command, evaluation / f"merge-{name}.log", cpu_env, 300)
    memory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True
    )
    assert all(int(line.split(",")[1]) < 100 for line in memory.splitlines() if int(line.split(",")[0]) < 4)
    models = [
        ("outer2", evaluation / "checkpoint-2"),
        ("outer5", old / "probes/checkpoint-5"),
        ("r16-outer15", evaluation / "checkpoint-continue-r16"),
        ("r4-outer14", evaluation / "checkpoint-branch-r4"),
    ]

    def one(item):
        gpu, (label, model) = item
        output = evaluation / label
        output.mkdir(exist_ok=False)
        env = os.environ | {"CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "2"}
        base = [sys.executable, "-m", "examples.mirroropd.research.studies.termination.evaluate_e54_recovery"]
        execute(
            base
            + [
                "generate",
                "--model",
                str(model),
                "--prompts",
                str(evaluation / "prompts.json"),
                "--output",
                str(output),
            ],
            output / "generate.log",
            env,
            1800,
        )
        execute(base + ["grade", "--output", str(output)], output / "grade.log", cpu_env, 300)
        print(json.dumps({"model": label, "metrics": json.loads((output / "metrics.json").read_text())}), flush=True)
        return label

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        labels = list(pool.map(one, enumerate(models)))
    (evaluation / "complete.json").write_text(json.dumps({"models": labels, "time": time.time()}) + "\n")


if __name__ == "__main__":
    main()
