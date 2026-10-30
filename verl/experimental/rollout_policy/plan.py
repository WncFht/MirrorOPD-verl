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
    identity: str | None = None

    def same_model(self, other):
        if self.kind != other.kind:
            return False
        if self.identity is not None and other.identity is not None:
            return self.identity == other.identity
        return self.path == other.path

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
            raise ValueError("right_weight must be finite and in [0, 1]")
        if type(self.steps) is not int or self.steps < 1:
            raise ValueError("Schedule steps must be a positive integer")

    @classmethod
    def parse(cls, value):
        if isinstance(value, int | float) and not isinstance(value, bool):
            return cls(start=float(value), end=float(value))
        if not isinstance(value, Mapping):
            raise ValueError("right_weight must be a number or a coefficient mapping")
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
    backend: str = "vllm_fusion"
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
class ModelRef:
    kind: str = "current"
    ref: str | None = None

    def __post_init__(self):
        if self.kind not in {"current", "initial", "teacher", "checkpoint"}:
            raise ValueError(f"Unknown model reference: {self.kind}")
        if (self.kind == "checkpoint") != (isinstance(self.ref, str) and bool(self.ref)):
            raise ValueError("Only checkpoint references require a nonempty ref")
        if self.kind != "checkpoint" and self.ref is not None:
            raise ValueError("current/initial/teacher references do not accept ref")

    def resolve(self, learner_path, teacher_path=None):
        if self.kind in {"current", "initial"}:
            path = learner_path
        elif self.kind == "teacher":
            if teacher_path is None:
                raise ValueError("teacher requires exactly one supervision teacher; use checkpoint explicitly")
            path = teacher_path
        else:
            path = self.ref
            if not Path(path).expanduser().is_dir():
                from examples.mirroropd.assets.paths import registered_model

                path = registered_model(path)
        from examples.mirroropd.assets.paths import require_current_location

        path = require_current_location(str(path))
        return ModelBinding(
            "current" if self.kind == "current" else "checkpoint", str(Path(path).expanduser().resolve())
        )


@dataclass(frozen=True)
class RolloutPolicyConfig:
    mode: str = "single"
    model: ModelRef | None = None
    left: ModelRef | None = None
    right: ModelRef | None = None
    right_weight: CoefficientRule = CoefficientRule()
    micro_batch_size: int = 4
    deadline_seconds: float = 3600.0
    drain_seconds: float = 60.0
    execution: FusionExecutionConfig = FusionExecutionConfig()
    fusion_space: str = "logit"

    def __post_init__(self):
        if self.mode not in {"single", "trajectory_mixture", "token_mixture"}:
            raise ValueError(f"Unknown rollout policy mode: {self.mode}")
        if self.fusion_space not in {"logit", "probability"}:
            raise ValueError("fusion_space must be logit or probability")
        if self.fusion_space != "logit" and self.mode != "token_mixture":
            raise ValueError("Probability fusion requires token_mixture")
        if self.mode == "single":
            if self.left is not None or self.right is not None:
                raise ValueError("single uses model; remove left/right")
            if self.right_weight.kind != "constant":
                raise ValueError("Coefficient schedules require a mixture mode")
        elif self.model is not None:
            raise ValueError("Mixtures use left/right; remove model")
        if type(self.micro_batch_size) is not int or self.micro_batch_size < 1:
            raise ValueError("micro_batch_size must be a positive integer")
        if any(not math.isfinite(v) or v <= 0 for v in (self.deadline_seconds, self.drain_seconds)):
            raise ValueError("Generation and drain deadlines must be finite and positive")

    @classmethod
    def parse(cls, values=None):
        values = dict(values or {})
        retired = {"source", "student", "teacher_weight", "teacher_model_path", "model_path"} & values.keys()
        if retired:
            raise ValueError(f"Retired rollout fields {sorted(retired)}; migrate to model/left/right/right_weight")
        for name in ("model", "left", "right"):
            if values.get(name) is not None:
                if not isinstance(values[name], Mapping):
                    raise ValueError(f"{name} must be a model reference mapping")
                values[name] = ModelRef(**values[name])
        values["right_weight"] = CoefficientRule.parse(values.get("right_weight", 0.5))
        values["execution"] = FusionExecutionConfig(**values.get("execution", {}))
        return cls(**values)

    @property
    def requires_fusion(self):
        return self.mode == "token_mixture" and not (
            self.right_weight.kind == "constant" and self.right_weight.start in (0, 1)
        )

    @property
    def references(self):
        if self.mode == "single":
            return {"model": self.model or ModelRef()}
        return {"left": self.left or ModelRef(), "right": self.right or ModelRef("teacher")}

    def bindings(self, learner_path, teacher_path=None):
        return {slot: ref.resolve(learner_path, teacher_path) for slot, ref in self.references.items()}

    @property
    def default_path(self):
        return self.mode == "single" and self.references["model"].kind == "current"


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
    bindings: tuple[tuple[str, ModelBinding], ...]
    requested_mode: str
    resolved_mode: str
    normalization: str | None
    coefficient_rule: CoefficientRule
    resolved_control: float
    weight_clock: str
    requests: tuple[RolloutRequest, ...]
    phases: tuple[RolloutPhase, ...]
    expected_sessions: tuple[tuple[str, tuple[int, ...]], ...]
    sampling: tuple[tuple[str, object], ...]
    termination: tuple[int, ...]
    fusion_space: str = "logit"

    def binding(self, source):
        return dict(self.bindings)[source]

    def has_current(self, source):
        if source == "fusion":
            return any(binding.kind == "current" for _, binding in self.bindings)
        return self.binding(source).kind == "current"

    @property
    def fusion_components(self):
        # The native receiver updates its first component. Routing and quotas
        # remain in declared slots; only token arithmetic is mapped here.
        left, right = self.binding("left"), self.binding("right")
        if right.kind == "current":
            return right, left, 1.0 - self.resolved_control
        return left, right, self.resolved_control

    @property
    def effective_right_fraction(self):
        return sum(r.source == "right" for r in self.requests) / len(self.requests)


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
    bindings=None,
):
    if not uids or len(uids) != len(stable_ids) or len(set(uids)) != len(uids):
        raise ValueError("A rollout batch requires unique prompt uids and matching stable row identities")
    if type(n) is not int or n < 1 or collection_step < 0:
        raise ValueError("Invalid response count or collection step")
    bindings = bindings if bindings is not None else policy.bindings(student_path, teacher_path)
    completed_outer = max(collection_step - 1, 0)
    rule = policy.right_weight if policy.mode != "single" else CoefficientRule(start=0, end=0)
    alpha = rule.resolve(completed_outer, optimizer_step)
    identities = [
        (row, str(sid), uid, j) for row, (sid, uid) in enumerate(zip(stable_ids, uids, strict=True)) for j in range(n)
    ]
    order = sorted(
        range(len(identities)),
        key=lambda i: (
            stable_seed(seed, collection_step, identities[i][1], identities[i][0], identities[i][3], "route"),
            i,
        ),
    )
    right_rows = set(order[: math.floor(alpha * len(identities) + 0.5)])
    single, reason = None, None
    if policy.mode == "single":
        single = "model"
    elif alpha in (0, 1):
        single, reason = ("left" if alpha == 0 else "right"), "endpoint"
    elif bindings["left"].same_model(bindings["right"]):
        single, reason = "left", "same_binding"
    elif policy.mode == "trajectory_mixture" and len(right_rows) in (0, len(identities)):
        single, reason = ("right" if right_rows else "left"), "empty_quota"
    requests = tuple(
        RolloutRequest(
            row,
            sid,
            uid,
            j,
            stable_seed(seed, collection_step, sid, row, j, "sample"),
            single or ("fusion" if policy.mode == "token_mixture" else ("right" if i in right_rows else "left")),
        )
        for i, (row, sid, uid, j) in enumerate(identities)
    )
    sources = (single,) if single else (("fusion",) if policy.mode == "token_mixture" else ("left", "right"))
    phases = tuple(RolloutPhase(source, tuple(r for r in requests if r.source == source)) for source in sources)
    return BatchRolloutPlan(
        collection_step,
        (completed_outer, optimizer_step),
        tuple(bindings.items()),
        policy.mode,
        "single" if single else policy.mode,
        reason,
        rule,
        alpha,
        rule.clock,
        requests,
        phases,
        tuple((uid, tuple(range(n))) for uid in uids),
        tuple(sorted((sampling or {}).items())),
        tuple(stop_ids),
        policy.fusion_space,
    )
