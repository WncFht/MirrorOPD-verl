#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Execute an explicit sequential queue on one user-assigned GPU."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--gpu", required=True)
p.add_argument("--jobs", type=Path, required=True)
p.add_argument("--models", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--cache", type=Path, required=True)
p.add_argument("--warm", type=Path, required=True)
a = p.parse_args()
env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu, OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8")
for j in json.loads(a.jobs.read_text()):
    name = f"{j['initial']}-{j['optimizer']}-{j['precision']}-fold{j['fold']}"
    if "lr" in j:
        name += "-fixedlr"
    if j.get("accum_fp32"):
        name += "-accum32"
    if (a.out / name / "complete.json").exists():
        print("already completed", name, flush=True)
        continue
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("run_study.py")),
        "train",
        "--models",
        str(a.models),
        "--out",
        str(a.out),
        "--cache",
        str(a.cache),
        "--optimizer",
        j["optimizer"],
        "--precision",
        j["precision"],
        "--fold",
        str(j["fold"]),
        "--init-tag",
        j["initial"],
        "--steps",
        str(j["steps"]),
    ]
    if j["initial"] == "warm64":
        cmd += ["--initial", str(a.warm)]
    if "lr" in j:
        cmd += ["--fixed-lr", str(j["lr"])]
    if j.get("accum_fp32"):
        cmd += ["--accum-fp32"]
    with (a.out / f"{name}.log").open("w") as f:
        subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
    print("completed", name, flush=True)
