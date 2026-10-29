# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Immutable scientific identities; execution choices are recorded separately."""

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from examples.mirroropd.research.core.artifacts import fingerprint
from examples.mirroropd.research.core.paths import workspace

WORK = workspace()


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else WORK / path


@dataclass(frozen=True)
class Model:
    name: str
    repo: str
    revision: str
    path: str
    parent: str | None = None
    method: str = "unknown"
    sources: tuple[str, ...] = ()
    lineage_note: str = ""

    def __post_init__(self):
        if not re.fullmatch(r"[a-f0-9]{40}", self.revision):
            raise ValueError(f"Pin an immutable Hub revision for {self.name}")

    @property
    def identity(self):
        return fingerprint(dict(repo=self.repo, revision=self.revision))

    @property
    def directory(self):
        return resolve(self.path)


@dataclass(frozen=True)
class Training:
    batch_size: int = 16
    reuse: int = 16
    horizon: int = 1024
    learning_rate: float = 1e-6
    betas: tuple[float, float] = (0.9, 0.999)
    epsilon: float = 1e-8
    weight_decay: float = 0.01
    clip: float = 1.0
    seed: int = 42
    bank_seed: int = 52000000
    score_interval: int = 16

    def __post_init__(self):
        if min(self.batch_size, self.reuse, self.horizon, self.score_interval) <= 0:
            raise ValueError("Batch, reuse, horizon and score interval must be positive")

        if self.learning_rate <= 0 or self.epsilon <= 0 or self.clip <= 0 or self.weight_decay < 0:
            raise ValueError("Optimizer scales must be positive; weight decay must be nonnegative")
        if len(self.betas) != 2 or any(not 0 <= value < 1 for value in self.betas):
            raise ValueError("Adam betas must lie in [0,1)")

    @property
    def denominator(self):
        return self.batch_size * self.horizon

    def score_steps(self, updates, readouts=()):
        """Always audit reuse boundaries and exported readouts, plus requested extra probes."""
        return tuple(
            sorted(
                {
                    0,
                    updates,
                    *readouts,
                    *range(self.reuse, updates + 1, self.reuse),
                    *range(self.score_interval, updates + 1, self.score_interval),
                }
            )
        )


@dataclass(frozen=True)
class Evaluation:
    samples: int = 4
    temperature: float = 0.7
    top_p: float = 0.9
    response_cap: int = 31744
    model_length: int = 32768
    seed: int = 72000000
    thinking: bool = True
    stop_ids: tuple[int, ...] = ()

    def __post_init__(self):
        if self.samples != 4:
            raise ValueError("This study series requires avg@4 for every candidate")
        if self.response_cap >= self.model_length or not self.stop_ids:
            raise ValueError("Evaluation needs prompt space and explicit stop IDs")


@dataclass(frozen=True)
class Execution:
    microbatch: int = 1
    token_chunk: int = 512
    activation_checkpointing: bool = True
    evaluation_concurrency: int = 16
    gpu_memory_fraction: float = 0.88
    parameter_dtype: str = "float32"
    backbone_dtype: str = "bfloat16"
    head_dtype: str = "float32"
    tf32: bool = False

    def __post_init__(self):
        if (self.parameter_dtype, self.backbone_dtype, self.head_dtype, self.tf32) != (
            "float32",
            "bfloat16",
            "float32",
            False,
        ):
            raise ValueError("The audited numerical contract is FP32 master/head, BF16 backbone, TF32 off")
        if min(self.microbatch, self.token_chunk, self.evaluation_concurrency) <= 0:
            raise ValueError("Execution batch sizes must be positive")


@dataclass(frozen=True)
class Arm:
    name: str
    teacher: str
    updates: int = 64
    evaluate_at: tuple[int, ...] = (64,)

    def __post_init__(self):
        if self.updates <= 0 or any(s <= 0 or s > self.updates for s in self.evaluate_at):
            raise ValueError(f"Invalid update/readout budget: {self.name}")
        if self.updates not in self.evaluate_at:
            raise ValueError("Every arm must evaluate its final endpoint")
        if len(set(self.evaluate_at)) != len(self.evaluate_at):
            raise ValueError("Duplicate readout step")


@dataclass(frozen=True)
class Study:
    name: str
    student: str
    models: dict[str, Model]
    arms: tuple[Arm, ...]
    question_source: str
    benchmark_source: str
    output: str
    cache: str
    training: Training = field(default_factory=Training)
    evaluation: Evaluation = field(default_factory=Evaluation)
    execution: Execution = field(default_factory=Execution)
    gpus: tuple[int, ...] = (4, 5, 6, 7)

    def __post_init__(self):
        if self.student not in self.models or any(a.teacher not in self.models for a in self.arms):
            raise ValueError("Every student and teacher must have a pinned model identity")
        if len({a.name for a in self.arms}) != len(self.arms):
            raise ValueError("Duplicate arm names")
        if any(a.updates % self.training.reuse for a in self.arms):
            raise ValueError("Complete batch-reuse blocks are required")
        if self.training.batch_size % self.execution.microbatch:
            raise ValueError("Microbatch must divide the effective batch")
        if not self.gpus or len(set(self.gpus)) != len(self.gpus) or any(g < 0 for g in self.gpus):
            raise ValueError("GPU allocation requires distinct nonnegative device IDs")

    @classmethod
    def load(cls, path):
        raw = json.loads(Path(path).read_text())
        raw["models"] = {k: Model(**v) for k, v in raw["models"].items()}
        raw["arms"] = tuple(Arm(**v) for v in raw["arms"])
        for key, kind in [("training", Training), ("evaluation", Evaluation), ("execution", Execution)]:
            raw[key] = kind(**raw.get(key, {}))
        return cls(**raw)

    @property
    def directory(self):
        return resolve(self.output)

    @property
    def cache_directory(self):
        return resolve(self.cache)

    @property
    def identity(self):
        data = asdict(self)
        # Location and hardware scheduling do not define the scientific protocol.
        for key in ["output", "cache", "gpus", "execution"]:
            data.pop(key)
        for model in data["models"].values():
            model.pop("path")
        return fingerprint(data)
