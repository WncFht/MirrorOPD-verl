# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Immutable global-batch routing, independent of Ray, GPU and completion order."""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


def stable_seed(*parts) -> int:
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") % (2**63 - 1)


@dataclass(frozen=True)
class ModelBinding:
    kind: str
    path: str

    def __post_init__(self):
        if self.kind not in {"current", "checkpoint"}:
            raise ValueError(f"Unknown model binding: {self.kind}")
        if not Path(self.path).is_dir():
            raise ValueError(f"Rollout requires a local checkpoint directory: {self.path}")


@dataclass(frozen=True)
class CoefficientRule:
    kind: str = "constant"
    start: float = 0.5
    end: float = 0.5
    steps: int = 1
    clock: str = "outer_step"

    def __post_init__(self):
        if self.kind not in {"constant", "linear"} or self.clock not in {"outer_step", "optimizer_step"}:
            raise ValueError("Only constant/linear coefficients and outer_step/optimizer_step clocks are supported")
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in (self.start, self.end)):
            raise ValueError("teacher_weight must be finite and in [0, 1]")
        if type(self.steps) is not int or self.steps < 1:
            raise ValueError("Schedule steps must be a positive integer")

    @classmethod
    def parse(cls, value):
        if isinstance(value, int | float) and not isinstance(value, bool):
            return cls(start=float(value), end=float(value))
        if not isinstance(value, Mapping):
            raise ValueError("teacher_weight must be a number or a coefficient mapping")
        return cls(**value)

    def resolve(self, outer_step: int, optimizer_step: int | None) -> float:
        if self.kind == "constant":
            return self.start
        coordinate = outer_step if self.clock == "outer_step" else optimizer_step
        if coordinate is None or coordinate < 0:
            raise ValueError(f"Cannot resolve schedule with unknown {self.clock}")
        return self.start + (self.end - self.start) * min(coordinate / self.steps, 1.0)


@dataclass(frozen=True)
class FusionExecutionConfig:
    backend: str = "hf_fusion"
    tensor_parallel_size: int = 1
    max_num_seqs: int = 2
    kv_cache_memory_bytes: int = 1073741824
    enforce_eager: bool = True

    def __post_init__(self):
        if self.backend not in {"hf_fusion", "vllm_fusion"}:
            raise ValueError("Unknown fusion execution backend")
        if self.tensor_parallel_size not in {1, 2} or (self.backend == "hf_fusion" and self.tensor_parallel_size != 1):
            raise ValueError("HF fusion supports TP1; native fusion supports TP1/TP2")
        for name in ("max_num_seqs", "kv_cache_memory_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.enforce_eager) is not bool:
            raise ValueError("enforce_eager must be a boolean")


@dataclass(frozen=True)
class RolloutPolicyConfig:
    mode: str = "single"
    source: str = "student"
    student: str = "current"
    teacher_weight: CoefficientRule = CoefficientRule()
    teacher_model_path: str | None = None
    model_path: str | None = None
    micro_batch_size: int = 4
    deadline_seconds: float = 3600.0
    drain_seconds: float = 60.0
    execution: FusionExecutionConfig = FusionExecutionConfig()

    def __post_init__(self):
        if self.mode not in {"single", "trajectory_mixture", "token_mixture"}:
            raise ValueError(f"Unknown rollout policy mode: {self.mode}")
        if self.source not in {"student", "teacher", "checkpoint"} or self.student not in {"current", "initial"}:
            raise ValueError("Invalid rollout source or student binding")
        if self.mode == "single" and self.teacher_weight.kind != "constant":
            raise ValueError("Coefficient schedules require a mixture mode")
        if self.mode != "single" and self.source != "student":
            raise ValueError("Mixture modes require source=student")
        if (self.source == "checkpoint") != (self.model_path is not None):
            raise ValueError("model_path is required only for source=checkpoint")
        if type(self.micro_batch_size) is not int or self.micro_batch_size < 1:
            raise ValueError("micro_batch_size must be a positive integer")
        if any(not math.isfinite(v) or v <= 0 for v in (self.deadline_seconds, self.drain_seconds)):
            raise ValueError("Generation and drain deadlines must be finite and positive")

    @classmethod
    def parse(cls, values=None):
        values = dict(values or {})
        values["teacher_weight"] = CoefficientRule.parse(values.get("teacher_weight", 0.5))
        values["execution"] = FusionExecutionConfig(**values.get("execution", {}))
        return cls(**values)

    @property
    def default_path(self):
        return self.mode == "single" and self.source == "student" and self.student == "current"


@dataclass(frozen=True)
class RolloutRequest:
    row: int
    stable_id: str
    uid: str
    response_index: int
    seed: int
    source: str

    @property
    def key(self):
        return self.uid, self.response_index


@dataclass(frozen=True)
class RolloutPhase:
    decoder: str
    requests: tuple[RolloutRequest, ...]


@dataclass(frozen=True)
class BatchRolloutPlan:
    collection_step: int
    actor_version: tuple[int, int | None]
    student_binding: ModelBinding
    teacher_binding: ModelBinding | None
    source_binding: ModelBinding | None
    coefficient_rule: CoefficientRule
    resolved_control: float
    weight_clock: str
    requests: tuple[RolloutRequest, ...]
    phases: tuple[RolloutPhase, ...]
    expected_sessions: tuple[tuple[str, tuple[int, ...]], ...]
    sampling: tuple[tuple[str, object], ...]
    termination: tuple[int, ...]

    @property
    def effective_teacher_fraction(self):
        return sum(r.source == "teacher" for r in self.requests) / len(self.requests)


def build_plan(
    policy,
    *,
    uids,
    stable_ids,
    n,
    collection_step,
    optimizer_step,
    seed,
    student_path,
    teacher_path=None,
    sampling=None,
    stop_ids=(),
):
    if not uids or len(uids) != len(stable_ids) or len(set(uids)) != len(uids):
        raise ValueError("A rollout batch requires unique prompt uids and matching stable row identities")
    if type(n) is not int or n < 1 or collection_step < 0:
        raise ValueError("Invalid response count or collection step")
    student = ModelBinding("current" if policy.student == "current" else "checkpoint", student_path)
    teacher_path = policy.teacher_model_path or teacher_path
    teacher = ModelBinding("checkpoint", teacher_path) if teacher_path else None
    if (policy.mode != "single" or policy.source == "teacher") and teacher is None:
        raise ValueError("A local rollout teacher checkpoint is required")
    completed_outer = max(collection_step - 1, 0)
    rule = policy.teacher_weight
    if policy.mode == "single":
        weight = float(policy.source == "teacher")
        rule = CoefficientRule(start=weight, end=weight)
    alpha = rule.resolve(completed_outer, optimizer_step)
    source_binding = ModelBinding("checkpoint", policy.model_path) if policy.source == "checkpoint" else None
    identities = [
        (row, str(sid), uid, j) for row, (sid, uid) in enumerate(zip(stable_ids, uids, strict=True)) for j in range(n)
    ]
    # Hash ordering gives a global permutation without depending on Python's hash seed,
    # UUID, worker count, source phase order, or a framework's RNG state.
    order = sorted(
        range(len(identities)),
        key=lambda i: (
            stable_seed(seed, collection_step, identities[i][1], identities[i][0], identities[i][3], "route"),
            i,
        ),
    )
    teachers = set(order[: math.floor(alpha * len(identities) + 0.5)])
    requests = []
    for i, (row, sid, uid, j) in enumerate(identities):
        if policy.mode == "single":
            source = policy.source
        elif policy.mode == "token_mixture":
            source = "student" if alpha == 0 else ("teacher" if alpha == 1 else "fusion")
        else:
            source = "teacher" if i in teachers else "student"
        requests.append(
            RolloutRequest(row, sid, uid, j, stable_seed(seed, collection_step, sid, row, j, "sample"), source)
        )
    requests = tuple(requests)
    if policy.mode == "token_mixture":
        # Endpoints skip the inactive model inside the same numerical executor.
        # Switching to another backend here would also change inference precision.
        phases = (RolloutPhase("fusion", requests),)
    else:
        sources = (policy.source,) if policy.mode == "single" else ("student", "teacher")
        phases = tuple(
            RolloutPhase(source, tuple(r for r in requests if r.source == source))
            for source in sources
            if any(r.source == source for r in requests)
        )
    return BatchRolloutPlan(
        collection_step,
        (completed_outer, optimizer_step),
        student,
        teacher,
        source_binding,
        rule,
        alpha,
        rule.clock,
        requests,
        phases,
        tuple((uid, tuple(range(n))) for uid in uids),
        tuple(sorted((sampling or {}).items())),
        tuple(stop_ids),
    )
