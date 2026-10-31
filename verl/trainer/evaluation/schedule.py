# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Pure evaluation schedules on completed outer boundaries."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Schedule:
    at: tuple[int, ...] = ()
    every: int | None = None
    start: int = 1
    final: bool = True

    def __post_init__(self):
        if self.every is not None and (type(self.every) is not int or self.every <= 0):
            raise ValueError("every must be a positive integer")
        if type(self.start) is not int or self.start < 0:
            raise ValueError("start must be a nonnegative integer")
        if any(type(step) is not int or step < 0 for step in self.at):
            raise ValueError("Explicit evaluation steps must be nonnegative integers")
        object.__setattr__(self, "at", tuple(sorted(set(self.at))))

    def due(self, outer, total):
        if type(outer) is not int or type(total) is not int or not 0 <= outer <= total or total < 1:
            raise ValueError("Evaluation requires a boundary within a positive training budget")
        return (
            outer in self.at
            or (self.every is not None and outer >= self.start and outer % self.every == 0)
            or (self.final and outer == total)
        )
