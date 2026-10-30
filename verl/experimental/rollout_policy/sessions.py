# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""AgentLoop/TQ transport and writer settlement, independent of model residency."""

import asyncio
import dataclasses
import time

import torch
import transfer_queue as tq
from tensordict import NonTensorData, NonTensorStack

from verl.workers.rollout.llm_server import LLMServerClient

from .collector import UnsafeDrainError


class LocalPhaseClient(LLMServerClient):
    def __init__(self, config, servers, *, fusion=False):
        super().__init__(config)
        self.servers = servers
        self.fusion = fusion

    async def abort(self, reason):
        if self.fusion:
            calls = [server.abort_pending.remote(reason) for server in self.servers]
            try:
                await asyncio.wait_for(asyncio.gather(*calls), timeout=60.0)
            except TimeoutError as exc:
                raise UnsafeDrainError("Cannot close fusion admission; terminate run") from exc

    async def generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):
        params = dict(sampling_params)
        route = params.pop("_rollout_replica")
        if type(route) is not int or not 0 <= route < len(self.servers):
            raise ValueError("Request has an invalid explicit replica route")
        output = await self.servers[route].generate.remote(
            request_id=request_id,
            prompt_ids=prompt_ids,
            sampling_params=params,
            priority=int(kwargs.get("priority", 0)),
        )
        output.extra_fields.setdefault("min_global_steps", output.extra_fields.get("global_steps"))
        output.extra_fields.setdefault("max_global_steps", output.extra_fields.get("global_steps"))
        return output


class TQPublisher:
    def __init__(self, deadline_seconds=3600):
        self.deadline_seconds = deadline_seconds

    async def publish(self, awaitable):
        task = asyncio.ensure_future(awaitable)
        _, pending = await asyncio.wait([task], timeout=self.deadline_seconds)
        if pending:
            task.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
            raise UnsafeDrainError("Parent publication deadline exceeded; terminate run without clearing keys")
        return task.result()

    async def running(self, plan):
        await self.publish(
            tq.async_kv_batch_put(
                keys=[uid for uid, _ in plan.expected_sessions],
                partition_id="train",
                tags=[{"status": "running"} for _ in plan.expected_sessions],
            )
        )

    async def finished(self, plan):
        await self.publish(
            tq.async_kv_batch_put(
                keys=[uid for uid, _ in plan.expected_sessions],
                partition_id="train",
                tags=[{"status": "finished"} for _ in plan.expected_sessions],
            )
        )

    async def failed_and_clear(self, plan):
        uids = [uid for uid, _ in plan.expected_sessions]
        await self.publish(
            tq.async_kv_batch_put(keys=uids, partition_id="train", tags=[{"status": "failure"} for _ in uids])
        )
        # single_turn_agent produces exactly one output per session.
        await self.publish(
            tq.async_kv_clear(
                partition_id="train", keys=uids + [f"{r.uid}_{r.response_index}_0" for r in plan.requests]
            )
        )


class SessionDriver:
    def __init__(self, context, workers, batch):
        self.context = context
        self.workers = workers
        self.batch = batch
        self.active_refs = []
        self.metrics = {}
        self.rank_metrics = []

    async def generate(self, phase, plan, prepared):
        tick = time.monotonic()
        sessions = []
        for request in phase.requests:
            prompt = {}
            for key, value in self.batch.items():
                if isinstance(value, torch.Tensor):
                    prompt[key] = value[request.row]
                elif isinstance(value, NonTensorStack):
                    prompt[key] = value[request.row].data
                elif isinstance(value, NonTensorData):
                    prompt[key] = value.data
                else:
                    raise TypeError(f"Unsupported prompt field {key}: {type(value)}")
            prompt["priority"] = request.ordinal
            params = dict(plan.sampling) | {"seed": request.seed, "_rollout_replica": prepared.routes.replica(request)}
            if phase.generator == "token_mixture":
                params["_fusion_phase_token"] = prepared.token
            metadata = {
                "mode": plan.requested_mode,
                "resolved_mode": plan.resolved_mode,
                "normalization": plan.normalization,
                "has_current": plan.has_current(request.source),
                "source": request.source,
                "actor_version": plan.actor_version,
                "collection_step": plan.collection_step,
                "right_weight": plan.resolved_control,
                "fusion_space": plan.fusion_space,
                "bindings": {slot: dataclasses.asdict(binding) for slot, binding in plan.bindings},
                "seed": request.seed,
                "temperature": params["temperature"],
                "stable_id": request.stable_id,
            }
            sessions.append({"request": request, "prompt": prompt, "sampling": params, "metadata": metadata})
        workers = self.workers
        chunks = [sessions[i :: len(workers)] for i in range(len(workers))]
        self.active_refs = [
            worker.generate_sessions.remote(
                chunk, prepared.client, self.context.policy.deadline_seconds, self.context.policy.drain_seconds
            )
            for worker, chunk in zip(workers, chunks, strict=True)
            if chunk
        ]
        results = await self.context.wait(self.active_refs)
        self.active_refs = []
        if phase.generator == "token_mixture":
            rank_metrics = await prepared.owner.audit()
            for rank, values in enumerate(rank_metrics):
                self.metrics[f"perf/memory/fusion_gpu_{rank}_allocated_peak_bytes"] = values["peak_allocated_bytes"]
                self.metrics[f"perf/memory/fusion_gpu_{rank}_reserved_peak_bytes"] = values["peak_reserved_bytes"]
                if "microbatches" in values:
                    self.metrics[f"perf/memory/fusion_gpu_{rank}_kv_peak_bytes"] = max(
                        (row["kv_peak_bytes"] for row in values["microbatches"]), default=0
                    )
            self.rank_metrics = rank_metrics
        self.metrics[f"perf/timing/rollout_policy_{phase.source}_generate_seconds"] = time.monotonic() - tick
        return [item for result in results for item in result]

    async def drain(self):
        # A cancelled Ray ObjectRef can settle before its actor coroutine has
        # finished cleanup. Obtain an independent acknowledgement from every
        # possible TQ writer before clearing keys or switching GPU ownership.
        workers = self.workers
        try:
            await self.context.guarded(
                asyncio.gather(
                    *(worker.drain_sessions.remote(self.context.policy.drain_seconds) for worker in workers)
                ),
                self.context.policy.drain_seconds,
            )
        except BaseException as exc:
            raise UnsafeDrainError("Cannot confirm all session writers stopped; terminate run") from exc
        self.active_refs = []
