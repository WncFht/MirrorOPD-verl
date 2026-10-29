# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Named student receiver using the existing bucketed IPC transport."""

import asyncio
import os

import ray

from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.ipc_identity import ipc_job_key
from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightSender


class ServerAdapter(BaseRollout):
    def __init__(self, config, model_config, device_mesh=None, *, settings):
        super().__init__(config, model_config, device_mesh)
        self.settings = settings
        self.server = settings["server"]
        self.loaded_version = None
        self.pending_version = None
        job = ipc_job_key(ray.get_runtime_context().get_job_id())
        self.address = f"ipc:///tmp/rl-colocate-zmq-{job}-replica-{settings['replica_rank']}-rank-0.sock"

    async def resume(self, tags=None):
        await self.server.wake_up.remote(tags=tags)

    async def release(self):
        await self.server.sleep.remote()

    async def update_weights(self, weights, wire_format="named_tensors", version=None, **kwargs):
        if wire_format != "named_tensors" or not self.settings["current"]:
            raise ValueError("Fusion receiver requires named tensors and current student")
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise ValueError("Native fusion transport currently supports one actor rank")
        self.loaded_version = self.pending_version = None
        result = self.server.collective_rpc.remote(
            "fusion_receive_weights", kwargs={"version": version, "use_shm": False}
        )
        sender = BucketedWeightSender(
            zmq_handle=self.address,
            bucket_size_mb=self.config.checkpoint_engine.update_weights_bucket_megabytes,
        )
        seen, errors = set(), []

        def validated_stream():
            for name, tensor in weights:
                if name in seen:
                    errors.append(f"Duplicate exported weight: {name}")
                    continue
                seen.add(name)
                yield name, tensor

        await sender.async_send_weights(validated_stream())
        received = await asyncio.wait_for(result, timeout=self.settings["deadline_seconds"])
        errors.extend(error for error in received if error)
        if not errors:
            self.pending_version = tuple(version)
        return "; ".join(errors) if errors else None

    async def commit_version(self, version):
        if self.pending_version != tuple(version):
            raise RuntimeError("Cannot commit incomplete native fusion weights")
        await self.server.collective_rpc.remote("fusion_commit_version", args=(version,))
        self.loaded_version = tuple(version)
