# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Explicit fixed-batch settings, shared by preparation, scoring and analysis."""

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FixedBatchProtocol:
    horizon: int = 1024
    batch_size: int = 16
    reuse: int = 16
    train_batches: int = 20
    total_batches: int = 40
    seed: int = 42
    lr: float = 1e-6
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.01
    clip: float = 1.0

    def __post_init__(self):
        if min(self.horizon, self.batch_size, self.reuse, self.train_batches) < 1:
            raise ValueError("Batch settings must be positive")
        if self.total_batches <= self.train_batches:
            raise ValueError("The evaluation bank must include held-out batches")
        if not math.isfinite(self.lr) or self.lr <= 0 or not math.isfinite(self.clip) or self.clip <= 0:
            raise ValueError("Learning rate and clip norm must be finite and positive")

    @property
    def normalizer(self):
        return self.batch_size * self.horizon

    @property
    def examples(self):
        return self.total_batches * self.batch_size

    @property
    def updates(self):
        return self.train_batches * self.reuse

    def bands(self):
        return (
            ("all", 0, self.horizon),
            ("early", 0, min(256, self.horizon)),
            ("middle", min(256, self.horizon), min(512, self.horizon)),
            ("late", min(512, self.horizon), self.horizon),
        )

    def as_record(self):
        return asdict(self)


# L is a frozen historical protocol, not the new dual-source R experiment.
LONG_PASS = FixedBatchProtocol()


def validate_long_pass(record):
    expected = dict(
        cap=LONG_PASS.horizon,
        prompts_per_batch=LONG_PASS.batch_size,
        consecutive_updates_per_batch=LONG_PASS.reuse,
        train_batches=LONG_PASS.train_batches,
        total_batches=LONG_PASS.total_batches,
        training_seed=LONG_PASS.seed,
        lr=LONG_PASS.lr,
        betas=list(LONG_PASS.betas),
        weight_decay=LONG_PASS.weight_decay,
        clip=LONG_PASS.clip,
    )
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"Historical L protocol mismatch: {key}={record.get(key)!r}, expected {value!r}")
    return LONG_PASS


def validate_resume(payload, identity, stop_after):
    if payload.get("experiment") != identity:
        raise ValueError(
            "Checkpoint lacks matching teacher, protocol and full trajectory identity; unsafe resume refused"
        )
    completed = payload["completed_batches"]
    reuse = identity["settings"]["reuse"]
    if payload["step"] != completed * reuse or not 0 <= completed < stop_after:
        raise ValueError("Checkpoint update counter or requested continuation is inconsistent")
