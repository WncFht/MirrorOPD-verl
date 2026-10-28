# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Batch requests before one actor-worker RPC; no per-token RPC or vocabulary transfer."""

import asyncio

import ray

from verl.workers.rollout.replica import RolloutReplica


@ray.remote(num_cpus=1)
class FusionBatchServer:
    def __init__(self, worker, micro_batch_size, method="fusion_generate_batch", prefix=None):
        self.worker = worker
        self.micro_batch_size = micro_batch_size
        self.method = method
        self.prefix = prefix
        self.queue = []
        self.task = None
        self.accepting = False
        self.error = None
        self.remaining = None
        self.expected = None
        self.phase_token = None
        self.seen = set()
        self.arrived = asyncio.Event()

    async def prepare(self, expected_requests=None, phase_token=None):
        await self.drain()
        if self.error:
            raise RuntimeError(self.error)
        self.remaining = expected_requests
        self.expected = expected_requests
        self.phase_token = phase_token
        self.seen = set()
        self.arrived.clear()
        self.accepting = True

    async def generate(self, request_id, prompt_ids, sampling_params, **kwargs):
        if not self.accepting or self.error:
            raise RuntimeError("Fusion server is not accepting requests")
        if self.phase_token is not None and sampling_params.get("_fusion_phase_token") != self.phase_token:
            raise RuntimeError("Stale fusion phase request")
        if request_id in self.seen or (self.expected is not None and len(self.seen) >= self.expected):
            raise RuntimeError("Duplicate or excess fusion request")
        self.seen.add(request_id)
        future = asyncio.get_running_loop().create_future()
        self.queue.append(
            ({"request_id": request_id, "prompt_ids": prompt_ids, "sampling_params": sampling_params}, future)
        )
        self.arrived.set()
        if self.task is None:
            self.task = asyncio.create_task(self._run())
        return await asyncio.shield(future)

    async def _run(self):
        try:
            if self.remaining is None:
                # Standalone clients without a global plan retain a bounded flush window.
                await asyncio.sleep(0.01)
            while self.queue or (self.remaining is not None and self.remaining > 0):
                if self.error:
                    break
                needed = min(self.micro_batch_size, self.remaining) if self.remaining is not None else 1
                if len(self.queue) < needed:
                    self.arrived.clear()
                    await self.arrived.wait()
                    continue
                self.queue.sort(key=lambda item: len(item[0]["prompt_ids"]))
                pending, self.queue = self.queue[: self.micro_batch_size], self.queue[self.micro_batch_size :]
                rows = [item[0] for item in pending]
                if self.remaining is not None:
                    self.remaining -= len(pending)
                try:
                    method = getattr(self.worker, self.method)
                    outputs = await (method.remote(self.prefix, rows) if self.prefix else method.remote(rows))
                    if len(outputs) != len(pending):
                        raise RuntimeError("Fusion worker returned the wrong number of requests")
                    if any(
                        getattr(output, "extra_fields", {}).get("fusion_request_id") != row["request_id"]
                        for row, output in zip(rows, outputs, strict=True)
                    ):
                        raise RuntimeError("Fusion worker returned mismatched request identities")
                    for (_, future), output in zip(pending, outputs, strict=True):
                        future.set_result(output)
                except BaseException as exc:
                    self.error = str(exc)
                    for _, future in pending + self.queue:
                        if not future.done():
                            future.set_exception(RuntimeError(self.error))
                    self.queue = []
                    break
        finally:
            self.task = None

    async def abort_pending(self, reason):
        self.accepting = False
        self.error = reason
        for _, future in self.queue:
            if not future.done():
                future.set_exception(RuntimeError(reason))
        self.queue = []
        self.arrived.set()
        # An in-flight GPU microbatch is allowed to settle before drain returns.

    async def drain(self):
        self.accepting = False
        task = self.task
        if task is not None:
            await asyncio.shield(task)

    async def sleep(self):
        await self.drain()


class HFFusionReplica(RolloutReplica):
    async def launch_servers(self):
        if self.world_size != 1 or len(self.workers) != 1:
            raise ValueError("hf_fusion supports TP1 colocated worker replicas")
        settings = self.config.engine_kwargs["hf_fusion"]
        self._server_handle = FusionBatchServer.remote(
            self.workers[0],
            settings["micro_batch_size"],
            settings.get("worker_method", "fusion_generate_batch"),
            settings.get("worker_prefix"),
        )
        self.servers = [self._server_handle]
        self._server_address = f"hf-fusion-worker-{self.replica_rank}"

    async def sleep(self):
        await self._server_handle.drain.remote()

    async def wake_up(self):
        await self._server_handle.prepare.remote()
