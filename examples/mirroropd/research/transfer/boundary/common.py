# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Explicit identities and filesystem layout for the authorized TB0--TB2 run."""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
WORK = REPO.parent
OUT = WORK / "wandb-logs/MirrorOPD-verl/analysis/teacher-boundary-20261028"
CACHE = WORK / "wandb-logs/derived/MirrorOPD-verl/teacher-boundary-20261028"
MODELS = {
    "r1": ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"),
    "r7": ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "916b56a44061fd5cd7d6a8fb632557ed4f724f60"),
    "justrl": ("hbx/JustRL-DeepSeek-1.5B", "0637e4096c789c67f9eecbe8355e0bdeddede1c2"),
    "polaris": ("POLARIS-Project/Polaris-7B-Preview", "faea72cf7123681069668e8da45c1c074e378830"),
    "skywork": ("Skywork/Skywork-OR1-Math-7B", "fcef970f1d10682f13c182b280146a194ed57d1e"),
    "nemotron": ("nvidia/Nemotron-Research-Reasoning-Qwen-1.5B", "b89048893f95246c6b5749b287f0049e6df42ee9"),
    "light": ("qihoo360/Light-R1-7B-DS", "06c9338cdbec28416e15f28a325b1439729ec237"),
    "minimath": ("oumi-ai/MiniMath-R1-1.5B", "9fcff43e7b6739a86f1e443e3b8d8b85fb00a998"),
}
ARMS = {
    "TB1-J": ("r1", "justrl"),
    "TB1-P": ("r1", "polaris"),
    "TB1-R": ("r1", "r7"),
    "TB1-S": ("r1", "skywork"),
    "TB1-N": ("r1", "nemotron"),
    "TB2-P7": ("r7", "polaris"),
    "TB2-L1": ("r1", "light"),
    "TB2-L7": ("r7", "light"),
    "TB2-M1": ("r1", "minimath"),
}
DATASETS = {"MATH-500": 500, "AMC23": 40, "AIME24": 30, "AIME25": 30, "AIME26": 30}
GROUPS = list("ABCDEFGH") + [f"M{i:02d}" for i in range(1, 5)] + ["Code01", "Code02", "IF01", "IF02"]


def model_path(name):
    return WORK / "model/teacher-boundary" / name / MODELS[name][1]


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())
