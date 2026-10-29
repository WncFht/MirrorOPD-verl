# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Native fusion lifecycle without changing the global plan or TQ collector."""

import copy
import time

from omegaconf import open_dict

from verl.experimental.rollout_policy.plan import stable_seed
from verl.workers.rollout.replica import RolloutMode


class NativeVLLMExecutor:
    def __init__(self, runtime):
        self.runtime = runtime
        self.replica = None
        self.current_version = None
        self.awake = False

    async def prepare(self, plan):
        from verl.experimental.rollout_policy.runtime import LocalPhaseClient
        from verl.workers.rollout.vllm_fusion.compat import fusion_hf_overrides
        from verl.workers.rollout.vllm_fusion.replica import VLLMFusionReplica

        rt = self.runtime
        wg = rt.trainer.actor_rollout_wg
        execution = rt.policy.execution
        tick = time.monotonic()
        if self.replica is None:
            rollout = copy.deepcopy(rt.config.actor_rollout_ref.rollout)
            rollout.load_format = "auto"
            rollout.max_model_len = rollout.max_model_len or (rollout.prompt_length + rollout.response_length)
            rollout.max_num_seqs = execution.max_num_seqs
            rollout.max_num_batched_tokens = rollout.max_model_len
            rollout.enforce_eager = execution.enforce_eager
            rollout.enable_prefix_caching = False
            rollout.enable_chunked_prefill = False
            rollout.logprobs_mode = "processed_logprobs"
            rollout.gpu_memory_utilization = 0.9
            with open_dict(rollout.engine_kwargs.vllm):
                rollout.engine_kwargs.vllm.hf_overrides = fusion_hf_overrides(plan.teacher_binding.path)
                rollout.engine_kwargs.vllm.kv_cache_memory_bytes = execution.kv_cache_memory_bytes
            self.replica = VLLMFusionReplica(
                4000, rollout, rt.config.actor_rollout_ref.model, gpus_per_node=1, name_suffix="policy_fusion"
            )
            self.replica.rollout_mode = RolloutMode.COLOCATED
            self.replica.workers = wg.workers
            await rt.guarded(self.replica.launch_servers())
            self.awake = True
            server = self.replica._server_handle
            await rt.guarded(server.collective_rpc.remote("fusion_initialize", args=(rt.policy.student == "current",)))
            await rt.wait(
                wg.configure_fusion(
                    {
                        "backend": "vllm_fusion",
                        "server": server,
                        "replica_rank": 4000,
                        "current": rt.policy.student == "current",
                        "deadline_seconds": rt.policy.deadline_seconds,
                    }
                )
            )
            rt.replicas["fusion"] = [self.replica]
            rt.clients["fusion"] = LocalPhaseClient(rt.config, [server], fusion=True)
        elif not self.awake:
            await rt.guarded(self.replica.wake_up())
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
        await rt.guarded(
            self.replica._server_handle.prepare.remote(
                len(plan.requests), rt.phase_token, plan.resolved_control, plan.actor_version
            )
        )
        await rt.guarded(self.replica._server_handle.set_global_steps.remote(plan.actor_version[0]))
        return rt.clients["fusion"]

    async def release(self):
        if self.replica is not None and self.awake:
            await self.runtime.guarded(self.replica.sleep(), self.runtime.policy.drain_seconds)
            self.awake = False

    async def metrics(self):
        return await self.runtime.guarded(self.replica._server_handle.collective_rpc.remote("fusion_audit"))
