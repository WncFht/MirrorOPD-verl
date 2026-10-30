# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Native fusion lifecycle without changing the global plan or TQ collector."""

import copy
import time

from omegaconf import open_dict

from verl.experimental.rollout_policy.lifecycle import (
    ComponentLayout,
    OwnedReplicas,
    PreparedPhase,
    RequestRoutes,
    loaded_versions,
)
from verl.experimental.rollout_policy.plan import stable_seed
from verl.experimental.rollout_policy.topology import FusionTopology
from verl.workers.rollout.replica import RolloutMode


class NativeVLLMExecutor:
    def __init__(self, context):
        self.context = context
        self.resources = OwnedReplicas(context)
        self.timings = {}
        self.current_version = None
        self.fusion_space = None

    async def prepare(self, plan):
        from verl.experimental.rollout_policy.sessions import LocalPhaseClient
        from verl.workers.rollout.vllm_fusion.compat import decode_graph_config, fusion_hf_overrides
        from verl.workers.rollout.vllm_fusion.replica import VLLMFusionReplica

        layout = ComponentLayout.from_plan(plan)
        first, second, alpha = layout.first, layout.second, plan.resolved_control
        ctx = self.context
        self.timings = {}
        if plan.fusion_space != ctx.policy.fusion_space or self.fusion_space not in {None, plan.fusion_space}:
            raise ValueError("Fusion space changed without rebuilding the native executor")
        self.fusion_space = plan.fusion_space
        wg = ctx.actor
        execution = ctx.policy.execution
        topology = FusionTopology(wg.world_size, execution.tensor_parallel_size)
        tick = time.monotonic()
        await ctx.wait(wg.park_fusion_actor())
        self.timings["perf/timing/fusion_actor_park_seconds"] = time.monotonic() - tick
        tick = time.monotonic()
        if not self.resources.replicas:
            rollout = copy.deepcopy(ctx.config.actor_rollout_ref.rollout)
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
                rollout.engine_kwargs.vllm.hf_overrides = fusion_hf_overrides(second.path, plan.fusion_space)
                rollout.engine_kwargs.vllm.kv_cache_memory_bytes = execution.kv_cache_memory_bytes
                # Match the independently checked component/KV probes. Automatic
                # backend selection is not covered by their numerical evidence.
                rollout.engine_kwargs.vllm.attention_config = {"backend": "TRITON_ATTN"}
                if not execution.enforce_eager:
                    compilation = decode_graph_config(execution.max_num_seqs)
                    rollout.engine_kwargs.vllm.compilation_config = compilation
                    rollout["cudagraph_capture_sizes"] = compilation["cudagraph_capture_sizes"]
            from verl.workers.config import HFModelConfig

            model = HFModelConfig(path=first.path, trust_remote_code=False)
            for index in range(topology.replica_count):
                replica = VLLMFusionReplica(
                    4000 + index,
                    rollout,
                    model,
                    gpus_per_node=ctx.config.trainer.n_gpus_per_node,
                    name_suffix="policy_fusion",
                )
                replica.rollout_mode = RolloutMode.COLOCATED
                replica.workers = wg.workers[topology.worker_slice(index)]
                self.resources.replicas.append(replica)
            await self.resources.launch(lambda replica: replica.launch_servers())
            servers = [replica._server_handle for replica in self.resources.replicas]
            await ctx.wait(
                list(
                    server.collective_rpc.remote("fusion_initialize", args=(first.kind == "current",))
                    for server in servers
                )
            )
            await ctx.wait(
                wg.configure_fusion(
                    {
                        "backend": "vllm_fusion",
                        "servers": servers,
                        "tp_size": topology.tensor_parallel_size,
                        "current": first.kind == "current",
                        "deadline_seconds": ctx.policy.deadline_seconds,
                    }
                )
            )
            self.client = LocalPhaseClient(ctx.config, servers, fusion=True)
        else:
            await self.resources.wake()
        self.timings["perf/timing/fusion_prepare_residency_seconds"] = time.monotonic() - tick
        if first.kind == "current" and self.current_version != plan.actor_version:
            tick = time.monotonic()
            await ctx.wait(
                wg.update_weights(
                    global_steps=plan.collection_step, target="policy_current", version=plan.actor_version
                )
            )
            self.current_version = plan.actor_version
            self.timings["perf/timing/fusion_current_refit_seconds"] = time.monotonic() - tick
        phase_token = str(stable_seed("fusion-phase", plan.collection_step, [r.uid for r in plan.requests]))
        routes = RequestRoutes.build(plan.requests, topology.replica_count)
        await ctx.wait(
            list(
                (
                    replica._server_handle.prepare.remote(
                        count, phase_token, alpha, plan.actor_version, plan.fusion_space, layout.reversed
                    )
                    for replica, count in zip(self.resources.replicas, routes.counts, strict=True)
                )
            )
        )
        await ctx.wait(
            list(
                replica._server_handle.set_global_steps.remote(plan.actor_version[0])
                for replica in self.resources.replicas
            )
        )
        return PreparedPhase("fusion", self.client, routes, self, loaded_versions(plan), phase_token)

    async def release(self, prepared=None):
        await self.resources.sleep()

    async def audit(self):
        groups = await self.context.wait(
            list(replica._server_handle.collective_rpc.remote("fusion_audit") for replica in self.resources.replicas)
        )
        return [dict(row, replica=index) for index, group in enumerate(groups) for row in group]
