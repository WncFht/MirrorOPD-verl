# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""One fail-closed scientific worker per allocated GPU; never share a live engine."""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

from examples.mirroropd.benchmarks.fusion.check_prefix_study_config import check
from examples.mirroropd.research.core.artifacts import file_hash
from examples.mirroropd.research.core.io import dump, read
from examples.mirroropd.research.studies.prefix_fusion.protocol import ARMS, protocol_identity


def frozen_revision(repo, expected):
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if actual != expected or subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=repo).returncode:
        raise RuntimeError("Frozen scientific source changed; inspect before continuing")


def execution(pair, gpu, root):
    return dict(
        FUSION_GPUS=str(gpu),
        FUSION_TP="1",
        FUSION_MAX_NUM_SEQS="32",
        FUSION_KV_BYTES=str((24 if pair == "J" else 32) * 1024**3),
        PREFIX_FULL_KL_BACKEND="precompute_fused",
        PREFIX_TOKEN_CHUNK="1024",
        PREFIX_PPO_TOKEN_BUDGET="32768",
        PREFIX_TEACHER_TOKEN_BUDGET="32768",
        PREFIX_ROLLOUT_SEQS="32",
        PREFIX_ROLLOUT_MEMORY_FRACTION=".65",
        PREFIX_RAY_CPUS="32",
        PREFIX_AGENT_WORKERS="8",
        PREFIX_REWARD_WORKERS="8",
        PREFIX_STORAGE_UNITS="8",
        PREFIX_OBJECT_STORE_BYTES=str(16 * 1024**3),
        PREFIX_TEACHER_LAYER_OFFLOAD="false",
        PREFIX_SCIENTIFIC_ROOT=str(root),
        GRADIENT_CHECKPOINTING="True",
        WANDB_MODE="offline",
        OMP_NUM_THREADS="1",
        PYTHONPYCACHEPREFIX="/root/.cache/mirroropd-fusion/pycache",
    )


def worker(root, pair, gpu, revision, label, *, config_only=False):
    repo = Path(__file__).resolve().parents[5]
    root = Path(root).resolve()
    workspace = repo.parent
    if (pair, gpu) not in [("J", 6), ("P", 7)] or not re.fullmatch(r"[A-Za-z0-9_-]+", label):
        raise ValueError("Invalid worker allocation or label")
    frozen_revision(repo, revision)
    bench = repo / "examples/mirroropd/benchmarks/fusion"
    reference_path = workspace / "wandb-logs/MirrorOPD-verl/analysis/batch-transfer-20261027/E20/config.json"
    reference = read(reference_path)
    dest = root / "queues" / f"{pair}-{label}"
    dest.mkdir(parents=True, exist_ok=False)
    settings = execution(pair, gpu, root)
    env = dict(os.environ, **settings, PYTHONPATH=str(repo))
    for key in ["CONFIG_ONLY", "DRY_RUN", "PREFIX_TORCH_TRACE", "PREFIX_RESUME_FROM"]:
        env.pop(key, None)
    manifest = dict(
        pair=pair,
        gpu=gpu,
        code_revision=revision,
        protocol=protocol_identity(),
        label=label,
        arms=list(ARMS),
        execution=settings,
        reference_sha256=file_hash(reference_path),
        pid=os.getpid(),
        started_at=time.time(),
        config_only=config_only,
    )
    dump(dest / "manifest.json", manifest)

    def event(**value):
        with (dest / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(time=time.time(), **value)) + "\n")

    def run(command, name, environment=None):
        frozen_revision(repo, revision)
        event(stage=name, state="start", command=list(map(str, command)))
        with (dest / f"{name}.log").open("x") as log:
            result = subprocess.run(
                list(map(str, command)), cwd=repo, env=environment or env, stdout=log, stderr=subprocess.STDOUT
            )
        dump(dest / f"{name}-exit.json", dict(exit_code=result.returncode, time=time.time()))
        event(stage=name, state="exit", exit_code=result.returncode)
        frozen_revision(repo, revision)
        if result.returncode:
            raise RuntimeError(f"{name} failed with exit {result.returncode}; queue stopped")

    # All six exact launch configs are checked before this worker allocates its GPU.
    for arm in ARMS:
        command = ["bash", bench / "run_fusion_prefix_study_fsdp2.sh", pair, arm, label, "science"]
        with (dest / f"{arm}.yaml").open("x") as out, (dest / f"{arm}-config.log").open("x") as log:
            result = subprocess.run(
                list(map(str, command)),
                cwd=repo,
                env=dict(env, CONFIG_ONLY="1", CUDA_VISIBLE_DEVICES=""),
                stdout=out,
                stderr=log,
            )
        if result.returncode:
            raise RuntimeError(f"{arm} config resolution failed")
        report = check(yaml.safe_load((dest / f"{arm}.yaml").read_text()), reference, pair, arm, scientific=True)
        dump(dest / f"{arm}-config-audit.json", report)
        if not report["passed"]:
            raise ValueError(report["errors"])
    if config_only:
        dump(dest / "config-complete.json", dict(passed=True, conditions=6, protocol=protocol_identity()))
        return
    for target in ["J", "P"]:
        gate = read(root / "gates" / f"{target}-readout.json")
        if not gate["passed"] or gate["protocol"] != protocol_identity():
            raise ValueError("Scientific lifecycle gate has not passed")
    # Baselines share the exact final accuracy entry and per-answer seeds.
    for source in ["S0", "J"] if pair == "J" else ["P"]:
        run(
            [
                "bash",
                bench / "run_fusion_local_fsdp2.sh",
                sys.executable,
                "-m",
                "examples.mirroropd.research.studies.prefix_fusion.tasks",
                "--prepared",
                root / "prepared-03",
                "--assets",
                root / "assets",
                "--weights",
                root / "assets" / source,
                "--out",
                root / "baseline-accuracy" / source,
            ],
            f"baseline-{source}",
        )
    for arm in ARMS:
        name = f"prefix-science-{pair}-{arm}-{label}"
        run(["bash", bench / "run_fusion_prefix_study_fsdp2.sh", pair, arm, label, "science"], arm)
        run(
            [
                sys.executable,
                bench / "audit_prefix_study.py",
                repo / "outputs" / name,
                "--reference",
                reference_path,
                "--pair",
                pair,
                "--arm",
                arm,
                "--mode",
                "science",
                "--exit-code",
                "0",
                "--out",
                dest / f"{arm}-audit.json",
            ],
            f"{arm}-audit",
            dict(env, CUDA_VISIBLE_DEVICES=""),
        )
    dump(
        dest / "training-readouts-complete.json",
        dict(
            passed=True,
            pair=pair,
            conditions=6,
            updates=3840,
            pending=["fresh endpoint cross scoring", "paired statistics"],
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pair", choices=["J", "P"], required=True)
    parser.add_argument("--gpu", type=int, choices=[6, 7], required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--config-only", action="store_true")
    args = parser.parse_args()
    worker(args.root, args.pair, args.gpu, args.revision, args.label, config_only=args.config_only)


if __name__ == "__main__":
    main()
