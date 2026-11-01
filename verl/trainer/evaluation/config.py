# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Typed scientific protocols and execution budgets for native evaluation."""

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields

from .artifacts import identity
from .schedule import Schedule


def strict_record(cls, value):
    if not isinstance(value, Mapping):
        raise ValueError(f"{cls.__name__} requires an options mapping")
    unknown = set(value) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**value)


@dataclass(frozen=True)
class GenerationProtocol:
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = -1
    max_new_tokens: int = 32768
    prompt_limit: int = 8192
    repetition_penalty: float = 1.0
    seed: int = 42
    model_family: str = "auto"
    template_json: str = "{}"
    stop_token_ids: tuple[int, ...] = ()
    answer_channel: str = "after_thinking"

    def __post_init__(self):
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("Invalid sampling temperature")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if type(self.top_k) is not int or self.top_k == 0 or self.top_k < -1:
            raise ValueError("top_k must be -1 or positive")
        if any(type(x) is not int or x <= 0 for x in (self.max_new_tokens, self.prompt_limit)):
            raise ValueError("Positive prompt and response budgets required")
        if (
            type(self.seed) is not int
            or self.seed < 0
            or not math.isfinite(self.repetition_penalty)
            or self.repetition_penalty <= 0
        ):
            raise ValueError("Invalid seed or repetition penalty")
        if self.answer_channel not in {"after_thinking", "whole_response"}:
            raise ValueError("Unknown answer channel")
        if not isinstance(json.loads(self.template_json), dict):
            raise ValueError("Template arguments must be a mapping")
        object.__setattr__(self, "stop_token_ids", tuple(self.stop_token_ids))
        if any(type(x) is not int or x < 0 for x in self.stop_token_ids):
            raise ValueError("Invalid stop token IDs")

    @classmethod
    def parse(cls, value):
        value = dict(value)
        if "template_kwargs" in value:
            if "template_json" in value:
                raise ValueError("Specify template_kwargs only once")
            value["template_json"] = json.dumps(value.pop("template_kwargs"), sort_keys=True, allow_nan=False)
        return strict_record(cls, value)

    def metadata(self):
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def sampling_metadata(self):
        return {k: v for k, v in self.metadata().items() if k != "answer_channel"}

    @property
    def protocol_id(self):
        return identity(self.sampling_metadata())

    def sample_seed(self, dataset, question_id, sample_index):
        # Neither n nor current checkpoint enters this seed: sample0 is stable as
        # budgets grow, and the same eval sample is paired across checkpoints.
        return int(identity([self.seed, "evaluation", dataset, question_id, sample_index])[:15], 16)

    def sampling(self, dataset, question_id, sample_index):
        result = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "max_tokens": self.max_new_tokens,
            "seed": self.sample_seed(dataset, question_id, sample_index),
        }
        if self.stop_token_ids:
            result["stop_token_ids"] = list(self.stop_token_ids)
        return result


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    domain: str
    path: str
    verifier: str
    n: int = 1
    protocol: str = "public"
    panel_ids: tuple[str, ...] = ()

    def __post_init__(self):
        if self.domain not in {"math", "code", "science", "if"} or self.verifier not in {
            "math",
            "lcb",
            "gpqa",
            "ifeval",
        }:
            raise ValueError("Unknown evaluation domain/verifier")
        if not self.name or not self.path or type(self.n) is not int or self.n < 1:
            raise ValueError("Dataset identity, path and positive n required")
        object.__setattr__(self, "panel_ids", tuple(self.panel_ids))
        if len(set(self.panel_ids)) != len(self.panel_ids):
            raise ValueError("Duplicate panel question IDs")


@dataclass(frozen=True)
class ScoreBudget:
    payload_bytes: int = 16 * 1024**3
    host_budget_bytes: int = 64 * 1024**3
    copy_reserve_factor: float = 2.5
    token_chunk: int = 256
    audit_state: bool = False
    objective_space: bool = False

    def __post_init__(self):
        if any(type(x) is not int or x <= 0 for x in (self.payload_bytes, self.host_budget_bytes, self.token_chunk)):
            raise ValueError("Positive RKL execution budgets required")
        if type(self.audit_state) is not bool:
            raise ValueError("audit_state must be a boolean")
        if not math.isfinite(self.copy_reserve_factor) or self.copy_reserve_factor < 1:
            raise ValueError("Reserve factor must cover at least the logical payload")

    def available_payload(self, live_payload_bytes=0):
        available = int(self.host_budget_bytes / self.copy_reserve_factor) - live_payload_bytes
        if available <= 0:
            raise ValueError("No evaluation payload budget remains after live training data")
        return min(self.payload_bytes, available)


@dataclass(frozen=True)
class SourceHistoryConfig:
    """Source-only scientific protocol with the shared scoring budget."""

    output: str
    source_file: str
    source_count: int = 128
    history_size: int = 128
    source_every: int = 1
    history_every: int = 1
    seed: int = 42
    initial_bank: str | None = None
    score: ScoreBudget = field(default_factory=ScoreBudget)

    def __post_init__(self):
        if any(not isinstance(path, str) or not path for path in (self.output, self.source_file)):
            raise ValueError("Source/history output and source_file must be nonempty paths")
        if self.initial_bank is not None and (not isinstance(self.initial_bank, str) or not self.initial_bank):
            raise ValueError("initial_bank must be a nonempty path or null")
        for name in ("source_count", "history_size", "source_every", "history_every"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("Source/history seed must be a nonnegative integer")

    @classmethod
    def parse(cls, value):
        value = dict(value)
        value["score"] = strict_record(ScoreBudget, value.get("score", {}))
        return strict_record(cls, value)


@dataclass(frozen=True)
class EvaluationConfig:
    output: str
    source_file: str
    bootstrap: str
    datasets: tuple[DatasetSpec, ...]
    protocols: dict[str, GenerationProtocol]
    training_domain: str = "math"
    source_count: int = 128
    history_size: int = 128
    seed: int = 42
    generation_concurrency: int = 64
    generation_chunk_size: int = 32
    verifier_workers: int = 8
    request_timeout: float = 1800.0
    score: ScoreBudget = field(default_factory=ScoreBudget)
    schedules: dict[str, Schedule] = field(default_factory=dict)
    final_export: bool = True
    verify_export: bool = True

    @classmethod
    def parse(cls, value):
        value = dict(value)
        value["datasets"] = tuple(strict_record(DatasetSpec, d) for d in value["datasets"])
        value["protocols"] = {k: GenerationProtocol.parse(v) for k, v in value["protocols"].items()}
        value["score"] = strict_record(ScoreBudget, value.get("score", {}))
        defaults = {
            "source": {"at": (0,), "every": 1},
            "history": {"every": 1},
            "banks": {"at": (0, 1, 2), "every": 5},
            "bench": {"at": (0, 25, 50)},
        }
        configured = value.pop("schedules", {})
        if set(configured) - set(defaults):
            raise ValueError("Unknown evaluation schedule")
        value["schedules"] = {k: strict_record(Schedule, configured.get(k, v)) for k, v in defaults.items()}
        result = strict_record(cls, value)
        if result.training_domain not in {"math", "code", "science", "if"}:
            raise ValueError("Unknown training domain")
        if len({d.name for d in result.datasets}) != len(result.datasets):
            raise ValueError("Dataset names must be unique")
        if "source" not in result.protocols or any(d.protocol not in result.protocols for d in result.datasets):
            raise ValueError("Unresolved generation protocol")
        if any(
            type(x) is not int or x <= 0
            for x in (
                result.source_count,
                result.history_size,
                result.generation_concurrency,
                result.generation_chunk_size,
                result.verifier_workers,
            )
        ):
            raise ValueError("Positive counts and concurrency required")
        if not math.isfinite(result.request_timeout) or result.request_timeout <= 0:
            raise ValueError("Positive request timeout required")
        return result
