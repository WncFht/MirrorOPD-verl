# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Record inference source identities even before a delivery commit exists."""

import hashlib
import subprocess
from pathlib import Path


def source_fingerprint():
    root = Path(__file__).resolve().parents[4]
    files = []
    for relative in (
        "verl/experimental/rollout_policy",
        "verl/workers/rollout/hf_fusion",
        "verl/workers/rollout/vllm_fusion",
        "verl/workers/rollout/vllm_rollout",
        "examples/mirroropd/benchmarks/fusion",
    ):
        files.extend(sorted((root / relative).glob("*.py")))
    files.extend(
        root / relative
        for relative in (
            "verl/workers/engine_workers.py",
            "verl/workers/rollout/weight_target.py",
            "verl/trainer/ppo/v1/trainer_base.py",
            "verl/trainer/ppo/v1/trainer_sync.py",
            "verl/trainer/config/rollout_policy/local.yaml",
            "examples/mirroropd/assets/paths.py",
            "examples/mirroropd/launchers/math_experiment.py",
        )
    )
    return {
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
    }
