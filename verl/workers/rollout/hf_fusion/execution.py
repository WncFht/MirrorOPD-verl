# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""HF reference lifecycle; model residency stays inside this backend."""

import asyncio
import copy
import time

from omegaconf import OmegaConf, open_dict

from verl.experimental.rollout_policy.lifecycle import (
    ComponentLayout,
    OwnedReplicas,
    PreparedPhase,
    RequestRoutes,
    loaded_versions,
)
from verl.experimental.rollout_policy.plan import stable_seed
from verl.experimental.rollout_policy.sessions import LocalPhaseClient
from verl.workers.rollout.replica import get_rollout_replica_class


class HFReferenceExecutor:
    def __init__(self, context):
        self.context = context
        self.resources = OwnedReplicas(context)
        self.initialized_fusion = False
        self.configured = False
        self.fusion_version = None
        self.timings = {}

    async def prepare(self, plan):
        self.timings = {}
        layout = ComponentLayout.from_plan(plan)
        first, second, alpha = layout.first, layout.second, plan.resolved_control
        wg = self.context.actor
        await self.context.wait(wg.park_fusion_actor())
        if not self.initialized_fusion:
            setup_tick = time.monotonic()
            settings = {
                "current": first.kind == "current",
                "reverse_components": layout.reversed,
                "student_path": first.path,
                "fusion_space": plan.fusion_space,
                "teacher_path": second.path,
                "stop_ids": list(plan.termination),
                "deadline_seconds": self.context.policy.deadline_seconds,
                "micro_batch_size": self.context.policy.micro_batch_size,
                "compact": True,
                "compact_power_of_two": True,
                "reference": OmegaConf.select(
                    self.context.config, "actor_rollout_ref.rollout.engine_kwargs.hf_fusion.reference", default=False
                ),
            }
            self.configured = True
            await self.context.wait(wg.configure_fusion(settings))
            rollout = copy.deepcopy(self.context.config.actor_rollout_ref.rollout)
            rollout.name = "hf_fusion"
            # Validation remains on the native engine. Its native-only detector
            # must not enter the reference generator's RolloutConfig validation.
            if rollout.get("val_kwargs") is not None:
                with open_dict(rollout.val_kwargs):
                    rollout.val_kwargs.repetition_detection = None
            if wg.fused_worker_used:
                settings["worker_method"] = wg.fused_worker_execute_fn_name
                settings["worker_prefix"] = f"{wg.sub_cls_name}_fwmn_fusion_generate_batch"
            with open_dict(rollout.engine_kwargs):
                rollout.engine_kwargs.hf_fusion = settings
            cls = get_rollout_replica_class("hf_fusion")
            replicas = [
                cls(
                    rank,
                    rollout,
                    self.context.config.actor_rollout_ref.model,
                    gpus_per_node=self.context.config.trainer.n_gpus_per_node,
                )
                for rank in range(wg.world_size)
            ]
            self.resources.replicas = replicas
            await self.resources.launch(lambda replica: replica.init_hybrid(wg))
            self.client = LocalPhaseClient(self.context.config, [r._server_handle for r in replicas], fusion=True)
            self.initialized_fusion = True
            self.timings["perf/timing/fusion_server_setup_seconds"] = time.monotonic() - setup_tick
        await self.context.guarded(
            asyncio.gather(*(r.sleep() for r in self.resources.replicas)), self.context.policy.drain_seconds
        )
        self.phase_token = str(stable_seed("fusion-phase", plan.collection_step, [r.uid for r in plan.requests]))
        if first.kind == "current" and self.fusion_version != plan.actor_version:
            tick = time.monotonic()
            await self.context.wait(wg.initialize_rollout_receiver(target="policy_current"))
            self.timings["perf/timing/fusion_student_initialize_seconds"] = time.monotonic() - tick
            tick = time.monotonic()
            await self.context.wait(
                wg.update_weights(
                    global_steps=plan.collection_step, target="policy_current", version=plan.actor_version
                )
            )
            self.timings["perf/timing/fusion_current_refit_seconds"] = time.monotonic() - tick
            self.fusion_version = plan.actor_version
        tick = time.monotonic()
        await self.context.wait(wg.prepare_fusion(alpha, plan.actor_version, phase_token=self.phase_token))
        self.timings["perf/timing/fusion_prepare_residency_seconds"] = time.monotonic() - tick
        routes = RequestRoutes.build(plan.requests, wg.world_size)
        self.resources.awake = True
        await self.context.wait(
            list(
                (
                    replica._server_handle.prepare.remote(count, self.phase_token)
                    for replica, count in zip(self.resources.replicas, routes.counts, strict=True)
                )
            )
        )
        return PreparedPhase("fusion", self.client, routes, self, loaded_versions(plan, "fusion"), self.phase_token)

    async def release(self, prepared=None):
        await self.resources.sleep()
        if self.configured:
            await self.context.wait(self.context.actor.release_fusion())

    async def audit(self):
        return await self.context.wait(self.context.actor.fusion_metrics())
