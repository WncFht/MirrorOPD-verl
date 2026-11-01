# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""One vLLM single-model generator; mutability is a property of its binding."""

import asyncio
import copy

from omegaconf import open_dict

from verl.experimental.rollout_policy.lifecycle import OwnedReplicas, PreparedPhase, RequestRoutes, loaded_versions
from verl.experimental.rollout_policy.sessions import LocalPhaseClient
from verl.workers.config import HFModelConfig
from verl.workers.rollout.replica import RolloutMode, get_rollout_replica_class


def training_generator_config(config):
    """Copy a private generator config without inheriting the validation window.

    The borrowed current/validation engine retains its original context. A
    smaller explicit engine limit stays in force; request caps are unchanged.
    """
    rollout = copy.deepcopy(config)
    training_context = rollout.prompt_length + rollout.response_length
    rollout.max_model_len = min(rollout.max_model_len or training_context, training_context)
    return rollout


class SingleVLLMExecutor:
    def __init__(self, context):
        self.context = context
        self.frozen = {}
        self.current_active = False
        self.timings = {}

    async def prepare(self, phase, plan):
        binding = plan.binding(phase.source)
        ctx = self.context
        routes = RequestRoutes.build(phase.requests, ctx.actor.world_size)
        if binding.kind == "current":
            self.current_active = True
            await ctx.wait(ctx.actor.update_weights(global_steps=plan.actor_version[0]))
            replicas = ctx.checkpoint_manager.replicas
            await ctx.set_policy_sampling(replicas, True)
            client = LocalPhaseClient(ctx.config, [r._server_handle for r in replicas])
            key = None
        else:
            await ctx.wait(ctx.actor.park_fusion_actor())
            key = (binding.kind, binding.identity or binding.path)
            client = await self.frozen_client(binding, max(routes.counts))
        return PreparedPhase(phase.source, client, routes, self, loaded_versions(plan, phase.source), resource_key=key)

    async def frozen_client(self, binding, max_requests):
        source, path = (binding.kind, binding.identity or binding.path), binding.path
        if source not in self.frozen:
            resources = self.frozen[source] = OwnedReplicas(self.context)
            rollout = training_generator_config(self.context.config.actor_rollout_ref.rollout)
            rollout.load_format = "auto"
            # Each frozen replica below owns exactly one actor worker. Keep its
            # engine topology local instead of inheriting the learner's layout.
            rollout.tensor_model_parallel_size = 1
            rollout.data_parallel_size = 1
            rollout.pipeline_model_parallel_size = 1
            rollout.expert_parallel_size = 1
            # The frozen model keeps a CPU weight backup across phases.
            model = HFModelConfig(path=path, trust_remote_code=False)
            rollout.max_num_seqs = min(rollout.max_num_seqs, max_requests)
            # Budget for the actual replica concurrency, rather than allocating
            # every free byte just because the learner's engine used a large ratio.
            hf = model.hf_config
            kv_heads = getattr(hf, "num_key_value_heads", None) or hf.num_attention_heads
            head_dim = getattr(hf, "head_dim", None) or hf.hidden_size // hf.num_attention_heads
            kv_bytes = 2 * 2 * hf.num_hidden_layers * kv_heads * head_dim * rollout.max_model_len * rollout.max_num_seqs
            granularity = 256 * 1024**2
            with open_dict(rollout.engine_kwargs.vllm):
                if self.context.config.trainer.get("evaluation"):
                    # This explicit budget belongs to the long-context current
                    # evaluation engine, not to a private fixed training source.
                    rollout.engine_kwargs.vllm.pop("kv_cache_memory_bytes", None)
                if rollout.engine_kwargs.vllm.get("kv_cache_memory_bytes") is None:
                    rollout.engine_kwargs.vllm.kv_cache_memory_bytes = max(
                        granularity, ((kv_bytes + granularity - 1) // granularity) * granularity
                    )
            if model.tokenizer.get_vocab() != self.context.tokenizer.get_vocab():
                raise ValueError("Frozen rollout source has a different token-ID mapping")
            replica_cls = get_rollout_replica_class("vllm")
            replicas = []
            for rank, worker in enumerate(self.context.actor.workers):
                # Frozen engines never share the default receiver's IPC identity.
                offset = len(self.frozen) * 1000
                replica = replica_cls(
                    offset + rank,
                    rollout,
                    model,
                    gpus_per_node=self.context.config.trainer.n_gpus_per_node,
                    name_suffix=f"policy_fixed_{len(self.frozen)}",
                )
                replica.rollout_mode = RolloutMode.COLOCATED
                replica.workers = [worker]
                replicas.append(replica)
            resources.replicas = replicas
            await resources.launch(lambda replica: replica.launch_servers())
        else:
            resources = self.frozen[source]
            await resources.wake()
        await self.context.set_policy_sampling(resources.replicas, True)
        return LocalPhaseClient(self.context.config, [r._server_handle for r in resources.replicas])

    async def release(self, prepared=None):
        if prepared is None or prepared.resource_key is None:
            if self.current_active:
                await self.context.set_policy_sampling(self.context.checkpoint_manager.replicas, False)
                await self.context.guarded(asyncio.to_thread(self.context.checkpoint_manager.sleep_replicas))
                self.current_active = False
        if prepared is None:
            for resources in self.frozen.values():
                await resources.sleep()
        elif prepared.resource_key is not None:
            await self.frozen[prepared.resource_key].sleep()

    async def validation_begin(self, version):
        self.current_active = True
        await self.context.wait(self.context.actor.update_weights(global_steps=version))
        await self.context.set_policy_sampling(self.context.checkpoint_manager.replicas, True)

    async def validation_end(self):
        # The initial fixed-bank evaluation can park an allocated default engine
        # before any current-policy refit has marked it active.
        if not self.current_active:
            await self.context.guarded(asyncio.to_thread(self.context.checkpoint_manager.sleep_replicas))
        await self.release()
