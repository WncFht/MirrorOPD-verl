# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Native fusion lifecycle without changing the global plan or TQ collector."""

import asyncio
import copy
import time

from omegaconf import open_dict

from verl.experimental.rollout_policy.plan import stable_seed
from verl.experimental.rollout_policy.topology import FusionTopology
from verl.workers.rollout.replica import RolloutMode


class NativeVLLMExecutor:
    def __init__(self, runtime):
        self.runtime = runtime
        self.replicas = []
        self.current_version = None
        self.awake = False
        self.fusion_space = None

    async def prepare(self, plan):
        from verl.experimental.rollout_policy.runtime import LocalPhaseClient
        from verl.workers.rollout.vllm_fusion.compat import decode_graph_config, fusion_hf_overrides
        from verl.workers.rollout.vllm_fusion.replica import VLLMFusionReplica

        rt = self.runtime
        if plan.fusion_space != rt.policy.fusion_space or self.fusion_space not in {None, plan.fusion_space}:
            raise ValueError("Fusion space changed without rebuilding the native executor")
        self.fusion_space = plan.fusion_space
        wg = rt.trainer.actor_rollout_wg
        execution = rt.policy.execution
        topology = FusionTopology(wg.world_size, execution.tensor_parallel_size)
        tick = time.monotonic()
        await rt.wait(wg.park_fusion_actor())
        rt.metrics["perf/timing/fusion_actor_park_seconds"] = time.monotonic() - tick
        tick = time.monotonic()
        if not self.replicas:
            rollout = copy.deepcopy(rt.config.actor_rollout_ref.rollout)
            rollout.load_format = "auto"
            rollout.tensor_model_parallel_size = topology.tensor_parallel_size
            rollout.expert_parallel_size = 1
            rollout.max_model_len = rollout.max_model_len or (rollout.prompt_length + rollout.response_length)
            rollout.max_num_seqs = execution.max_num_seqs
            rollout.max_num_batched_tokens = rollout.max_model_len
            rollout.enforce_eager = execution.enforce_eager
            rollout.enable_prefix_caching = False
            rollout.enable_chunked_prefill = False
            rollout.logprobs_mode = "processed_logprobs"
            rollout.gpu_memory_utilization = 0.9
            with open_dict(rollout.engine_kwargs.vllm):
                rollout.engine_kwargs.vllm.hf_overrides = fusion_hf_overrides(
                    plan.teacher_binding.path, plan.fusion_space
                )
                rollout.engine_kwargs.vllm.kv_cache_memory_bytes = execution.kv_cache_memory_bytes
                # Match the independently checked component/KV probes. Automatic
                # backend selection is not covered by their numerical evidence.
                rollout.engine_kwargs.vllm.attention_config = {"backend": "TRITON_ATTN"}
                if not execution.enforce_eager:
                    compilation = decode_graph_config(execution.max_num_seqs)
                    rollout.engine_kwargs.vllm.compilation_config = compilation
                    rollout["cudagraph_capture_sizes"] = compilation["cudagraph_capture_sizes"]
            for index in range(topology.replica_count):
                replica = VLLMFusionReplica(
                    4000 + index,
                    rollout,
                    rt.config.actor_rollout_ref.model,
                    gpus_per_node=rt.config.trainer.n_gpus_per_node,
                    name_suffix="policy_fusion",
                )
                replica.rollout_mode = RolloutMode.COLOCATED
                replica.workers = wg.workers[topology.worker_slice(index)]
                self.replicas.append(replica)
            await rt.guarded(asyncio.gather(*(replica.launch_servers() for replica in self.replicas)))
            self.awake = True
            servers = [replica._server_handle for replica in self.replicas]
            await rt.guarded(
                asyncio.gather(
                    *(
                        server.collective_rpc.remote("fusion_initialize", args=(rt.policy.student == "current",))
                        for server in servers
                    )
                )
            )
            await rt.wait(
                wg.configure_fusion(
                    {
                        "backend": "vllm_fusion",
                        "servers": servers,
                        "tp_size": topology.tensor_parallel_size,
                        "current": rt.policy.student == "current",
                        "deadline_seconds": rt.policy.deadline_seconds,
                    }
                )
            )
            rt.replicas["fusion"] = self.replicas
            rt.clients["fusion"] = LocalPhaseClient(rt.config, servers, fusion=True)
        elif not self.awake:
            await rt.guarded(asyncio.gather(*(replica.wake_up() for replica in self.replicas)))
            self.awake = True
        rt.metrics["perf/timing/fusion_prepare_residency_seconds"] = time.monotonic() - tick
        if rt.policy.student == "current" and self.current_version != plan.actor_version:
            tick = time.monotonic()
            await rt.wait(
                wg.update_weights(
                    global_steps=plan.collection_step, target="fusion_student", version=plan.actor_version
                )
            )
            self.current_version = plan.actor_version
            rt.metrics["perf/timing/fusion_current_refit_seconds"] = time.monotonic() - tick
        rt.phase_token = str(stable_seed("fusion-phase", plan.collection_step, [r.uid for r in plan.requests]))
        counts = [0] * topology.replica_count
        n = rt.config.actor_rollout_ref.rollout.n
        for request in plan.requests:
            counts[(request.row * n + request.response_index) % topology.replica_count] += 1
        await rt.guarded(
            asyncio.gather(
                *(
                    replica._server_handle.prepare.remote(
                        count, rt.phase_token, plan.resolved_control, plan.actor_version, plan.fusion_space
                    )
                    for replica, count in zip(self.replicas, counts, strict=True)
                )
            )
        )
        await rt.guarded(
            asyncio.gather(
                *(replica._server_handle.set_global_steps.remote(plan.actor_version[0]) for replica in self.replicas)
            )
        )
        return rt.clients["fusion"]

    async def release(self):
        if self.replicas and self.awake:
            await self.runtime.guarded(
                asyncio.gather(*(replica.sleep() for replica in self.replicas)), self.runtime.policy.drain_seconds
            )
            self.awake = False

    async def metrics(self):
        groups = await self.runtime.guarded(
            asyncio.gather(*(replica._server_handle.collective_rpc.remote("fusion_audit") for replica in self.replicas))
        )
        return [dict(row, replica=index) for index, group in enumerate(groups) for row in group]
