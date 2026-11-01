# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Probability-space definitions shared by objectives and value-only readouts.

Head rows and tokenizer IDs are different bases. No implicit min-vocabulary or
legal-token intersection is performed. Index selection retains autograd's scatter
into the original head; the ordinary native path never copies a weight matrix.
"""

import hashlib
import json
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class DistributionSpec:
    mode: str = "native"
    common_vocab_size: int | None = None
    student_ids: tuple[int, ...] = ()
    teacher_ids: tuple[int, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "student_ids", tuple(self.student_ids))
        object.__setattr__(self, "teacher_ids", tuple(self.teacher_ids))
        if self.mode not in {"native", "common_conditioned"}:
            raise ValueError("Distribution mode must be native or common_conditioned")
        explicit = bool(self.student_ids or self.teacher_ids)
        if self.mode == "native":
            if explicit or self.common_vocab_size is not None:
                raise ValueError("Native distribution cannot specify a conditional support")
        elif (self.common_vocab_size is not None) == explicit:
            raise ValueError("Specify either common_vocab_size or paired student_ids/teacher_ids")
        if self.common_vocab_size is not None and (
            type(self.common_vocab_size) is not int or self.common_vocab_size < 1
        ):
            raise ValueError("common_vocab_size must be a positive integer")
        if explicit:
            if not self.student_ids or len(self.student_ids) != len(self.teacher_ids):
                raise ValueError("Paired supports must have equal nonzero lengths")
            for ids in (self.student_ids, self.teacher_ids):
                if len(set(ids)) != len(ids) or any(type(i) is not int or i < 0 for i in ids):
                    raise ValueError("Support IDs must be distinct nonnegative integers")

    @classmethod
    def parse(cls, value=None):
        return value if isinstance(value, cls) else cls(**dict(value or {}))

    def compile(self, student_width, teacher_width, termination_ids=()):
        if min(student_width, teacher_width) < 1:
            raise ValueError("Empty model output support")
        if self.mode == "native":
            if student_width > teacher_width:
                raise ValueError(
                    "Native RKL requires teacher support to cover student; explicitly choose common_conditioned"
                )
            student, teacher = (), ()
        elif self.common_vocab_size is not None:
            if self.common_vocab_size > min(student_width, teacher_width):
                raise ValueError("Declared common support exceeds a model output head")
            student = teacher = tuple(range(self.common_vocab_size))
        else:
            student, teacher = self.student_ids, self.teacher_ids
            if max(student) >= student_width or max(teacher) >= teacher_width:
                raise ValueError("Declared support ID exceeds a model output head")
        mapped = tuple(termination_ids)
        if mapped and student:
            retained = set(mapped) & set(student)
            if retained and retained != set(mapped):
                raise ValueError("Conditional support must retain or exclude the complete termination group")
            mapped = tuple(student.index(i) for i in mapped) if retained else ()
        return CompiledDistribution(self, student_width, teacher_width, student, teacher, mapped)


@dataclass(frozen=True)
class CompiledDistribution:
    spec: DistributionSpec
    student_width: int
    teacher_width: int
    student_ids: tuple[int, ...]
    teacher_ids: tuple[int, ...]
    termination_ids: tuple[int, ...]

    @property
    def metadata(self):
        # A compact range stays compact on disk even for C151936.
        return {
            "schema": 1,
            "basis": "head_rows",
            "spec": asdict(self.spec),
            "student_width": self.student_width,
            "teacher_width": self.teacher_width,
            "termination_ids": self.termination_ids,
            "temperature": 1.0,
        }

    @property
    def measure_id(self):
        return hashlib.sha256(json.dumps(self.metadata, sort_keys=True).encode()).hexdigest()

    def select(self, weight, role):
        ids = self.student_ids if role == "student" else self.teacher_ids
        if not ids:
            return weight
        if self.spec.common_vocab_size is not None:
            return weight[: self.spec.common_vocab_size]
        return weight.index_select(0, torch.tensor(ids, device=weight.device))

    def outside(self, matrix, role):
        ids = self.student_ids if role == "student" else self.teacher_ids
        if not ids:
            return matrix[:0]
        if self.spec.common_vocab_size is not None:
            return matrix[self.spec.common_vocab_size :]
        keep = torch.ones(len(matrix), device=matrix.device, dtype=torch.bool)
        keep[torch.tensor(ids, device=matrix.device)] = False
        return matrix[keep]

    def excluded_from_logits(self, logits, role):
        outside = self.outside(logits.T, role).T
        if outside.shape[-1] == 0:
            return logits.new_zeros(len(logits))
        inside = self.select(logits.T, role).T
        return torch.sigmoid(outside.logsumexp(-1) - inside.logsumexp(-1))

    @torch.no_grad()
    def excluded_mass(self, hidden, weight, role, *, token_chunk, vocab_chunk=8192):
        """Stable partition ratio, without a token-by-full-vocabulary cache."""
        from verl.utils.linear import project

        ids = self.student_ids if role == "student" else self.teacher_ids
        dtype = torch.float64 if hidden.dtype == torch.float64 else torch.float32
        result = hidden.new_zeros(len(hidden), dtype=dtype)
        if not ids or len(ids) == len(weight):
            return result
        selected = self.select(weight, role)
        outside = self.outside(weight, role)
        for start in range(0, len(hidden), token_chunk):
            h = hidden[start : start + token_chunk]
            partitions = []
            for matrix in (outside, selected):
                z = h.new_full((len(h),), -torch.inf, dtype=dtype)
                for v in range(0, len(matrix), vocab_chunk):
                    z = torch.logaddexp(z, project(h, matrix[v : v + vocab_chunk]).logsumexp(-1))
                partitions.append(z)
            result[start : start + len(h)] = torch.sigmoid(partitions[0] - partitions[1])
        return result
