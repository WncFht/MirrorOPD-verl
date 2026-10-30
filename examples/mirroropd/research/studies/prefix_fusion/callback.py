# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Synchronous scientific readouts, isolated from training RNG and optimizer updates."""

import os
import random
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from examples.mirroropd.research.core.artifacts import file_hash
from examples.mirroropd.research.core.io import dump, read
from examples.mirroropd.research.studies.prefix_fusion.protocol import PROTOCOL, protocol_identity


@contextmanager
def preserve_driver_rng():
    python, numpy, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:
        yield
    finally:
        random.setstate(python)
        np.random.set_state(numpy)
        torch.set_rng_state(cpu)


def wait_rpc(value):
    import ray

    if isinstance(value, ray.ObjectRef):
        return ray.get(value)
    if isinstance(value, list | tuple) and value and all(isinstance(v, ray.ObjectRef) for v in value):
        return ray.get(value)
    return value


class ScientificReadout:
    def __init__(self, config):
        self.config = config
        self.settings = config.trainer.scientific_evaluation
        self.root = Path(self.settings.root).resolve()
        self.prepared = self.root / "prepared-03"
        self.pair = self.settings.pair
        self.gpu = str(self.settings.physical_gpu)
        self.smoke = bool(self.settings.get("smoke", False))
        self.repo = Path(__file__).resolve().parents[5]
        if self.gpu not in {str(gpu) for gpu in range(8)} or self.pair not in {"J", "P"}:
            raise ValueError("Scientific readout has invalid model/resource binding")
        if (
            config.trainer.n_gpus_per_node != 1
            or config.trainer.nnodes != 1
            or config.trainer.v1.trainer_mode != "sync"
        ):
            raise ValueError("Scientific readouts currently require single-rank synchronous FSDP2")
        if (
            not config.actor_rollout_ref.actor.fsdp_config.param_offload
            or not config.actor_rollout_ref.actor.fsdp_config.optimizer_offload
        ):
            raise ValueError("Readout capacity requires actor parameters and optimizer offload")
        data = read(self.prepared / "complete.json")
        if data["inputs"]["protocol"] != protocol_identity():
            raise ValueError("Prepared questions belong to another scientific protocol")
        configured = [Path(p).resolve() for p in config.data.train_files]
        if configured != [(self.prepared / "train.parquet").resolve()]:
            raise ValueError("Scientific training must use the frozen disjoint training pool")
        if file_hash(configured[0]) != data["artifacts"]["train.parquet"]:
            raise ValueError("Scientific training data changed")
        for source in ["S0", self.pair]:
            if not (self.root / "banks" / source / "complete.json").exists():
                raise ValueError("Fixed evaluation bank is not ready")
        if not (self.root / "teacher-cache" / self.pair / "complete.json").exists():
            raise ValueError("Teacher cache is not ready")
        self.output = Path(config.trainer.default_local_dir).parent / "science"
        self.output.mkdir(parents=True, exist_ok=True)
        dump(
            self.output / "protocol.json",
            dict(protocol=PROTOCOL, smoke=self.smoke, pair=self.pair, physical_gpu=self.gpu),
        )

    def _run(self, module, arguments, log):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=self.gpu, PYTHONPATH=str(self.repo), OMP_NUM_THREADS="1")
        for key in ["RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]:
            env.pop(key, None)
        with log.open("x") as stream:
            subprocess.run(
                [sys.executable, "-m", module, *map(str, arguments)],
                cwd=self.repo,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )

    def on_readout(self, trainer, outer):
        every = 1 if self.smoke else 5
        if outer % every:
            return {}
        expected = outer * 16
        if trainer.actor_optimizer_step != expected or trainer.actor_optimizer_step_attempts != expected:
            raise ValueError("Readout is not at the expected successful-update boundary")
        dest = self.output / f"outer-{outer:03d}"
        if dest.exists():
            raise ValueError("Readout already exists; preserve prior attempts and use a new run directory")
        dest.mkdir()
        before = (trainer.actor_optimizer_step, trainer.actor_optimizer_step_attempts)
        started = time.monotonic()
        snapshot = dest / "weights"
        with preserve_driver_rng():
            if outer == 0:
                trainer.checkpoint_manager.sleep_replicas()
            wait_rpc(trainer.actor_rollout_wg.export_scientific_readout(str(snapshot), outer, verify_state=self.smoke))
            wait_rpc(trainer.actor_rollout_wg.park_fusion_actor())
            weights = snapshot / "huggingface"
            risk_args = [
                "score",
                "--assets",
                self.root / "assets",
                "--banks",
                self.root / "banks",
                "--cache",
                self.root / "teacher-cache" / self.pair,
                "--weights",
                weights,
                "--out",
                dest / "fixed",
                "--outer",
                outer,
            ]
            if self.smoke:
                risk_args += ["--limit-per-source", 2, "--horizon", 32]
            self._run("examples.mirroropd.research.studies.prefix_fusion.scoring", risk_args, dest / "fixed.log")
            if not self.smoke and outer in [0, 20, 40]:
                long_args = list(risk_args)
                long_args[long_args.index("--out") + 1] = dest / "long"
                self._run(
                    "examples.mirroropd.research.studies.prefix_fusion.scoring",
                    long_args + ["--long-panel"],
                    dest / "long.log",
                )
            final = int(self.config.trainer.total_training_steps)
            if outer == final or (not self.smoke and outer == 20):
                panel = "test" if outer == final else "dev"
                args = [
                    "--prepared",
                    self.prepared,
                    "--assets",
                    self.root / "assets",
                    "--weights",
                    weights,
                    "--out",
                    dest / "accuracy",
                    "--panel",
                    panel,
                ]
                if self.smoke:
                    args += ["--smoke"]
                self._run("examples.mirroropd.research.studies.prefix_fusion.tasks", args, dest / "accuracy.log")
        if before != (trainer.actor_optimizer_step, trainer.actor_optimizer_step_attempts):
            raise ValueError("Evaluation changed optimizer counters")
        retained = outer in ([0, 1, 2] if self.smoke else [20, 40])
        fixed = read(dest / "fixed/per-question.json")
        summary = {}
        horizon = 32 if self.smoke else 1024
        for source in ["S0", self.pair]:
            for domain in ["math", "code", "if"]:
                rows = [r for r in fixed if r["source"] == source and r["domain"] == domain]
                if rows:
                    for metric in ["rkl", "js"]:
                        summary[f"science/fixed/{source}/{domain}/{metric}"] = sum(
                            r["stats"]["all"][metric] for r in rows
                        ) / (len(rows) * horizon)
        dump(
            dest / "complete.json",
            dict(
                outer=outer,
                optimizer_step=expected,
                smoke=self.smoke,
                protocol=protocol_identity(),
                elapsed_seconds=time.monotonic() - started,
                fixed_receipt=file_hash(dest / "fixed/complete.json"),
                retained_weights=retained,
                weights_directory=str(weights),
                metrics=summary,
            ),
        )
        if not retained:
            shutil.rmtree(snapshot)
        return summary | {"science/readout_seconds": time.monotonic() - started, "science/readout_outer": outer}
