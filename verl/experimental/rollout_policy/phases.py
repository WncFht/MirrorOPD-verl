# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Generator selection and explicit phase transitions for one global batch."""

import time

from verl.workers.rollout.vllm_rollout.policy_execution import SingleVLLMExecutor


class ExecutionPool:
    def __init__(self, context):
        self.context = context
        self.single = SingleVLLMExecutor(context)
        self.fusion = None

    async def prepare(self, phase, plan):
        if phase.generator == "single":
            return await self.single.prepare(phase, plan)
        if self.fusion is None:
            if self.context.policy.execution.backend == "vllm_fusion":
                from verl.workers.rollout.vllm_fusion.execution import NativeVLLMExecutor

                self.fusion = NativeVLLMExecutor(self.context)
            else:
                from verl.workers.rollout.hf_fusion.execution import HFReferenceExecutor

                self.fusion = HFReferenceExecutor(self.context)
        return await self.fusion.prepare(plan)

    async def release_all(self):
        await self.single.release()
        if self.fusion is not None:
            await self.fusion.release()


class BatchExecution:
    def __init__(self, pool, sessions):
        self.pool = pool
        self.sessions = sessions

    async def prepare(self, phase, plan):
        tick = time.monotonic()
        prepared = await self.pool.prepare(phase, plan)
        self.sessions.metrics.update(prepared.owner.timings)
        self.sessions.metrics[f"perf/timing/rollout_policy_{phase.source}_prepare_seconds"] = time.monotonic() - tick
        return prepared

    async def generate(self, phase, plan, prepared):
        return await self.sessions.generate(phase, plan, prepared)

    async def release(self, prepared):
        tick = time.monotonic()
        await prepared.release()
        self.sessions.metrics[f"perf/timing/rollout_policy_{prepared.source}_release_seconds"] = time.monotonic() - tick

    async def drain(self):
        await self.sessions.drain()

    async def release_all(self):
        await self.pool.release_all()
