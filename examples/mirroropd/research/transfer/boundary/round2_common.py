# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""October 29 scope: three continuations, three new teachers, Math baseline only."""

from . import common as previous
from .common import WORK

DATASETS = previous.DATASETS
GROUPS = previous.GROUPS
REPO = previous.REPO
dump = previous.dump
read = previous.read

OLD_OUT = previous.OUT
OLD_CACHE = previous.CACHE
OUT = WORK / "wandb-logs/MirrorOPD-verl/analysis/teacher-boundary-20261029"
CACHE = WORK / "wandb-logs/derived/MirrorOPD-verl/teacher-boundary-20261029"
NEW_MODELS = {
    "deepscaler": ("agentica-org/DeepScaleR-1.5B-Preview", "e3f524ce413a296b4d388e7560dd5c82c1c56725"),
    "brorl": ("nvidia/Nemotron-Research-Reasoning-Qwen-1.5B", "3441fcdf8c6e81a2959e6352ff50122e3c677d72"),
    "deepcoder": ("agentica-org/DeepCoder-1.5B-Preview", "103033da54f104b0d0a3570547fb3721fba99962"),
    "math15": ("Qwen/Qwen2.5-Math-1.5B", "4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2"),
}
MODELS = previous.MODELS | NEW_MODELS
ARMS = {
    "CONT-J": dict(teacher="justrl", old_arm="TB1-J", start=64, stop=128),
    "CONT-N": dict(teacher="nemotron", old_arm="TB1-N", start=64, stop=128),
    "CONT-M": dict(teacher="minimath", old_arm="TB2-M1", start=64, stop=128),
    "NEW-DS": dict(teacher="deepscaler", old_arm=None, start=0, stop=64),
    "NEW-BRO": dict(teacher="brorl", old_arm=None, start=0, stop=64),
    "NEW-DC": dict(teacher="deepcoder", old_arm=None, start=0, stop=64),
}


def model_path(name):
    return WORK / "model/teacher-boundary" / name / MODELS[name][1]


def initial_path(arm):
    config = ARMS[arm]
    if config["old_arm"]:
        return OLD_CACHE / "checkpoints" / config["old_arm"] / "step064"
    return model_path("r1")


def final_path(arm):
    return CACHE / "models" / arm / f"step{ARMS[arm]['stop']:03d}"


def eval_specs():
    """Original models and endpoints use a common four-sample protocol per row."""
    specs = {}
    for name in NEW_MODELS:
        specs[f"teacher-{name}"] = dict(
            path=str(model_path(name)),
            teacher=name,
            kind="teacher",
            student=None,
            template="math_native" if name == "math15" else "shared",
            response_cap=3072 if name == "math15" else 31744,
            model_length=4096 if name == "math15" else 32768,
        )
    for arm, (student, teacher) in previous.ARMS.items():
        specs[f"after-{arm}-064"] = dict(
            path=str(OLD_CACHE / "checkpoints" / arm / "step064"),
            teacher=teacher,
            student=student,
            kind="old-endpoint",
            arm=arm,
            step=64,
            template="shared",
            response_cap=31744,
            model_length=32768,
        )
    for arm, config in ARMS.items():
        specs[f"after-{arm}-{config['stop']:03d}"] = dict(
            path=str(final_path(arm)),
            teacher=config["teacher"],
            student="r1",
            kind="new-endpoint",
            arm=arm,
            step=config["stop"],
            template="shared",
            response_cap=31744,
            model_length=32768,
        )
    return specs
