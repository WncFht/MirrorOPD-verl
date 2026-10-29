# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Run teacher on GPUs 0-1 and two frozen continuation cohorts on GPUs 2/3."""

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from examples.mirroropd.research.core.paths import REPO
from examples.mirroropd.research.studies.termination.run_e54_recovery_eval import execute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    repo = REPO
    memory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True
    )
    assert all(int(line.split(",")[1]) < 100 for line in memory.splitlines() if int(line.split(",")[0]) < 4)
    jobs = [
        ("teacher", "0,1", 2, 64, repo.parent / "model/Qwen3-30B-A3B-Instruct-2507"),
        ("student5", "2", 1, 16, repo / "outputs/e54-rope-debug-20261028-hppk13/run/probes/checkpoint-5"),
        (
            "student10",
            "3",
            1,
            16,
            repo / "outputs/e54-recovery-controls-r3-20261028-hppk13/evaluation/checkpoint-continue10",
        ),
    ]

    def one(job):
        label, devices, tp, seqs, model = job
        output = root / label
        output.mkdir(exist_ok=False)
        base = [
            sys.executable,
            "-u",
            "-m",
            "examples.mirroropd.research.studies.termination.measure_e54_training_lengths",
        ]
        command = base + [
            "generate",
            "--model",
            str(model),
            "--requests",
            str(root / f"{label}-requests.json"),
            "--output",
            str(output),
            "--tp",
            str(tp),
            "--max-num-seqs",
            str(seqs),
        ]
        env = os.environ | {
            "CUDA_VISIBLE_DEVICES": devices,
            "OMP_NUM_THREADS": "2",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        }
        (output / "launch.json").write_text(
            json.dumps({"command": command, "physical_gpus": devices, "time": time.time()}, indent=2) + "\n"
        )
        try:
            execute(command, output / "generate.log", env, 3600)
            execute(
                base + ["grade", "--model", str(repo.parent / "model/Qwen3-1.7B-Base"), "--output", str(output)],
                output / "grade.log",
                os.environ | {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2"},
                900,
            )
        except BaseException as exc:
            (output / "exit.json").write_text(json.dumps({"success": False, "error": repr(exc)}) + "\n")
            raise
        (output / "exit.json").write_text(json.dumps({"success": True, "time": time.time()}) + "\n")
        print(f"Completed {label}", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(one, jobs))
    (root / "complete.json").write_text(json.dumps({"time": time.time(), "jobs": 3}) + "\n")


if __name__ == "__main__":
    main()
