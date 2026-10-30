# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Explicit phase ownership and bounded operations shared by execution backends."""

import asyncio
from dataclasses import dataclass, field
from typing import Any

from .collector import UnsafeDrainError


@dataclass(frozen=True)
class RequestRoutes:
    assignments: tuple
    counts: tuple[int, ...]

    @classmethod
    def build(cls, requests, replica_count):
        if replica_count < 1:
            raise ValueError("A phase requires at least one replica")
        assignments = tuple((r.key, r.ordinal % replica_count) for r in requests)
        if len({key for key, _ in assignments}) != len(assignments):
            raise ValueError("Duplicate request route")
        counts = tuple(sum(rank == index for _, rank in assignments) for index in range(replica_count))
        return cls(assignments, counts)

    def replica(self, request):
        return dict(self.assignments)[request.key]


@dataclass(frozen=True)
class ComponentLayout:
    first: Any
    second: Any
    reversed: bool

    @classmethod
    def from_plan(cls, plan):
        left, right = plan.binding("left"), plan.binding("right")
        return cls(right, left, True) if right.kind == "current" else cls(left, right, False)


@dataclass(frozen=True)
class PreparedPhase:
    source: str
    client: Any
    routes: RequestRoutes
    owner: Any
    loaded_versions: tuple
    token: str | None = None
    resource_key: Any = None

    async def release(self):
        await self.owner.release(self)


@dataclass(frozen=True)
class ExecutionContext:
    config: Any
    policy: Any
    actor: Any
    checkpoint_manager: Any = None
    tokenizer: Any = None
    pending: set = field(default_factory=set, compare=False, repr=False)

    async def guarded(self, awaitable, timeout=None):
        task = asyncio.ensure_future(awaitable)
        self.pending.add(task)

        def settled(done):
            self.pending.discard(done)
            if not done.cancelled():
                done.exception()

        task.add_done_callback(settled)
        try:
            _, pending = await asyncio.wait([task], timeout=timeout or self.policy.deadline_seconds)
        except asyncio.CancelledError as exc:
            raise UnsafeDrainError("Lifecycle awaiter cancelled; remote work may still own GPU resources") from exc
        if pending:
            task.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
            raise UnsafeDrainError("Lifecycle deadline exceeded; GPU ownership is uncertain, stop this run")
        return task.result()

    async def wait(self, refs):
        refs = refs if isinstance(refs, list) else [refs]
        # gather(return_exceptions=True) waits for every rank before reporting a failure.
        results = await self.guarded(asyncio.gather(*refs, return_exceptions=True))
        errors = [value for value in results if isinstance(value, BaseException)]
        if errors:
            raise errors[0]
        return results

    async def set_policy_sampling(self, replicas, enabled):
        await self.wait(
            [
                replica._server_handle.collective_rpc.remote("configure_rollout_policy_sampler", args=(enabled,))
                for replica in replicas
            ]
        )


@dataclass
class OwnedReplicas:
    """Register handles before launch; failed preparations never lose their owner."""

    context: ExecutionContext
    replicas: list = field(default_factory=list)
    ready: bool = False
    awake: bool = False

    async def launch(self, start):
        # All launches settle before cleanup; a timeout is fail-stop, never close-and-retry.
        try:
            await self.context.wait([start(replica) for replica in self.replicas])
        except UnsafeDrainError:
            raise
        except BaseException:
            await self.close()
            raise
        self.ready = self.awake = True

    async def wake(self):
        if not self.ready:
            raise RuntimeError("Cannot wake an incompletely prepared replica group")
        if not self.awake:
            try:
                await self.context.wait([replica.wake_up() for replica in self.replicas])
            except UnsafeDrainError:
                raise
            except BaseException:
                await self.close()
                raise
            self.awake = True

    async def sleep(self):
        if self.ready and self.awake:
            await self.context.wait([replica.sleep() for replica in self.replicas])
            self.awake = False

    async def close(self):
        import ray

        # Only servers owned by these replicas; colocated actor workers are borrowed.
        for replica in self.replicas:
            for server in getattr(replica, "servers", []):
                ray.kill(server, no_restart=True)
        self.replicas.clear()
        self.ready = self.awake = False


def loaded_versions(plan, source):
    return tuple(
        (slot, plan.actor_version if binding.kind == "current" else binding.identity or binding.path)
        for slot, binding in plan.bindings
        if source == "fusion" or slot == source
    )
