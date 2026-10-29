# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Explicit actor-rank to fusion-replica/TP-rank mapping for local execution."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FusionTopology:
    actor_world_size: int
    tensor_parallel_size: int

    def __post_init__(self):
        if self.actor_world_size not in {1, 2} or self.tensor_parallel_size not in {1, 2}:
            raise ValueError("Native fusion currently supports one or two local GPUs")
        if self.actor_world_size % self.tensor_parallel_size:
            raise ValueError("Fusion TP must divide the actor GPU pool")

    @property
    def replica_count(self):
        return self.actor_world_size // self.tensor_parallel_size

    def receiver(self, actor_rank):
        if not 0 <= actor_rank < self.actor_world_size:
            raise ValueError("Actor rank is outside the fusion topology")
        return actor_rank // self.tensor_parallel_size, actor_rank % self.tensor_parallel_size

    def worker_slice(self, replica):
        start = replica * self.tensor_parallel_size
        return slice(start, start + self.tensor_parallel_size)
