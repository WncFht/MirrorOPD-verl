# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Exclusive wall-time phases for a probe session, including I/O and idle gaps."""

import time
from collections import defaultdict
from contextlib import contextmanager

from examples.mirroropd.research.core.io import dump


class PerformanceLedger:
    """Measure sequential phases; report all remaining time as unattributed.

    Callers supply a CUDA synchronization barrier at phase boundaries when GPU
    work is included. Nested phases are rejected to prevent double counting.
    Each resume session has its own immutable filename and cumulative snapshot.
    """

    def __init__(self, directory, *, clock=time.perf_counter):
        self.clock = clock
        self.started = clock()
        self.seconds = defaultdict(float)
        self.counts = defaultdict(int)
        self.active = None
        self.path = directory / f"performance-{time.time_ns()}.json"

    @contextmanager
    def phase(self, name, *, synchronize=None):
        if self.active is not None:
            raise RuntimeError("Performance phases must not overlap")
        if synchronize:
            synchronize()
        started = self.clock()
        self.active = name
        try:
            yield
        finally:
            if synchronize:
                synchronize()
            self.seconds[name] += self.clock() - started
            self.counts[name] += 1
            self.active = None

    def snapshot(self, *, sites, complete=False):
        elapsed = self.clock() - self.started
        measured = sum(self.seconds.values())
        value = dict(
            session_wall_seconds=elapsed,
            phases_seconds=dict(self.seconds),
            phase_calls=dict(self.counts),
            unattributed_seconds=max(0.0, elapsed - measured),
            session_sites=sites,
            sites_per_minute=sites * 60 / elapsed if elapsed else None,
            complete=complete,
        )
        dump(self.path, value)
        return value
