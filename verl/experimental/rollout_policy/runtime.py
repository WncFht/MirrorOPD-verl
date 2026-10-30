# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Shared-sync rollout lifecycle and one collector for the original global batch."""

import asyncio
import copy
import dataclasses
import json
import time
from pathlib import Path

import torch
import transfer_queue as tq
from omegaconf import OmegaConf, open_dict
from tensordict import NonTensorData, NonTensorStack

from verl.experimental.rollout_policy.collector import UnsafeDrainError, collect_plan
from verl.experimental.rollout_policy.plan import build_plan, stable_seed
from verl.utils.ray_utils import auto_await
from verl.workers.config import HFModelConfig
from verl.workers.rollout.llm_server import LLMServerClient
from verl.workers.rollout.replica import RolloutMode, get_rollout_replica_class


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
        priority = int(kwargs.get("priority", 0))
        output = await self.servers[priority % len(self.servers)].generate.remote(
            request_id=request_id,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
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


class RolloutRuntime:
    def __init__(self, trainer):
        self.trainer = trainer
        self.config = trainer.config
        self.policy = trainer.rollout_policy
        self.replicas = {}
        self.clients = {}
        self.fusion_version = None
        self.initialized_fusion = False
        self.metrics = {}
        self.active_refs = []
        self.bindings = getattr(trainer, "rollout_bindings", None)
        self.current_sampling_active = False

    async def guarded(self, awaitable, timeout=None):
        task = asyncio.ensure_future(awaitable)
        _, pending = await asyncio.wait([task], timeout=timeout or self.policy.deadline_seconds)
        if pending:
            raise UnsafeDrainError("Lifecycle deadline exceeded; GPU ownership is uncertain, stop this run")
        return task.result()

    async def wait(self, refs):
        refs = refs if isinstance(refs, list) else [refs]
        tasks = [asyncio.ensure_future(ref) for ref in refs]
        if not tasks:
            return []
        _, pending = await asyncio.wait(tasks, timeout=self.policy.deadline_seconds)
        if pending:
            # Cancelling a Ray awaiter does not prove the GPU writer stopped.
            raise UnsafeDrainError("Remote phase deadline exceeded; terminate run and its Ray actors")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        errors = [r for r in results if isinstance(r, BaseException)]
        if errors:
            self.active_refs = []
            raise errors[0]
        return results

    async def set_policy_sampling(self, replicas, enabled):
        await self.guarded(
            asyncio.gather(
                *(
                    replica._server_handle.collective_rpc.remote("configure_rollout_policy_sampler", args=(enabled,))
                    for replica in replicas
                )
            )
        )

    async def set_current_sampling(self, enabled):
        if not enabled and not self.current_sampling_active:
            return
        if enabled:
            self.current_sampling_active = True
        await self.set_policy_sampling(self.trainer.checkpoint_manager.replicas, enabled)
        self.current_sampling_active = enabled

    @auto_await
    async def sync_validation(self):
        await self.set_current_sampling(False)
        tick = time.monotonic()
        await self.wait(self.trainer.actor_rollout_wg.update_weights(global_steps=self.trainer.global_steps))
        if hasattr(self.trainer, "timing_raw"):
            self.trainer.timing_raw["validation_weight_sync"] = time.monotonic() - tick

    @auto_await
    async def sleep_validation(self):
        await self.guarded(asyncio.to_thread(self.trainer.checkpoint_manager.sleep_replicas))

    def teacher_path(self):
        teachers = self.config.distillation.teacher_models
        return next(iter(teachers.values())).model_path if len(teachers) == 1 else None

    async def frozen_client(self, source, path, max_requests):
        if source not in self.replicas:
            rollout = copy.deepcopy(self.config.actor_rollout_ref.rollout)
            rollout.load_format = "auto"
            rollout.max_model_len = rollout.max_model_len or (rollout.prompt_length + rollout.response_length)
            # The frozen model keeps a CPU weight backup across phases.
            model = HFModelConfig(path=path, trust_remote_code=False)
            rollout.max_num_seqs = min(rollout.max_num_seqs, max_requests)
            # Budget for the actual replica concurrency, rather than allocating
            # every free byte just because the learner's engine used a large ratio.
            hf = model.hf_config
            kv_heads = getattr(hf, "num_key_value_heads", hf.num_attention_heads)
            head_dim = getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads)
            kv_bytes = 2 * 2 * hf.num_hidden_layers * kv_heads * head_dim * rollout.max_model_len * rollout.max_num_seqs
            granularity = 256 * 1024**2
            with open_dict(rollout.engine_kwargs.vllm):
                if rollout.engine_kwargs.vllm.get("kv_cache_memory_bytes") is None:
                    rollout.engine_kwargs.vllm.kv_cache_memory_bytes = max(
                        granularity, ((kv_bytes + granularity - 1) // granularity) * granularity
                    )
            if model.tokenizer.get_vocab() != self.trainer.tokenizer.get_vocab():
                raise ValueError("Frozen rollout source has a different token-ID mapping")
            replica_cls = get_rollout_replica_class("vllm")
            replicas = []
            for rank, worker in enumerate(self.trainer.actor_rollout_wg.workers):
                # Frozen engines never share the default receiver's IPC identity.
                offset = {"left": 1000, "right": 2000, "model": 3000}[source]
                replica = replica_cls(
                    offset + rank,
                    rollout,
                    model,
                    gpus_per_node=self.config.trainer.n_gpus_per_node,
                    name_suffix=f"policy_{source}",
                )
                replica.rollout_mode = RolloutMode.COLOCATED
                replica.workers = [worker]
                replicas.append(replica)
            await self.guarded(asyncio.gather(*(r.launch_servers() for r in replicas)), self.policy.deadline_seconds)
            self.replicas[source] = replicas
            self.clients[source] = LocalPhaseClient(self.config, [r._server_handle for r in replicas])
        else:
            await self.guarded(
                asyncio.gather(*(r.wake_up() for r in self.replicas[source])), self.policy.deadline_seconds
            )
        await self.set_policy_sampling(self.replicas[source], True)
        return self.clients[source]

    async def fusion_client(self, plan):
        if self.policy.execution.backend == "vllm_fusion":
            from .execution import NativeVLLMExecutor

            if not hasattr(self, "fusion_executor"):
                self.fusion_executor = NativeVLLMExecutor(self)
            return await self.fusion_executor.prepare(plan)
        first, second, alpha = plan.fusion_components
        wg = self.trainer.actor_rollout_wg
        if not self.initialized_fusion:
            setup_tick = time.monotonic()
            settings = {
                "current": first.kind == "current",
                "student_path": first.path,
                "fusion_space": plan.fusion_space,
                "teacher_path": second.path,
                "stop_ids": list(plan.termination),
                "deadline_seconds": self.policy.deadline_seconds,
                "micro_batch_size": self.policy.micro_batch_size,
                "compact": True,
                "compact_power_of_two": True,
                "reference": OmegaConf.select(
                    self.config, "actor_rollout_ref.rollout.engine_kwargs.hf_fusion.reference", default=False
                ),
            }
            await self.wait(wg.configure_fusion(settings))
            rollout = copy.deepcopy(self.config.actor_rollout_ref.rollout)
            rollout.name = "hf_fusion"
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
                    self.config.actor_rollout_ref.model,
                    gpus_per_node=self.config.trainer.n_gpus_per_node,
                )
                for rank in range(wg.world_size)
            ]
            await self.guarded(asyncio.gather(*(r.init_hybrid(wg) for r in replicas)))
            self.replicas["fusion"] = replicas
            self.clients["fusion"] = LocalPhaseClient(self.config, [r._server_handle for r in replicas], fusion=True)
            self.initialized_fusion = True
            self.metrics["perf/timing/fusion_server_setup_seconds"] = time.monotonic() - setup_tick
        await self.guarded(asyncio.gather(*(r.sleep() for r in self.replicas["fusion"])), self.policy.drain_seconds)
        self.phase_token = str(stable_seed("fusion-phase", plan.collection_step, [r.uid for r in plan.requests]))
        if first.kind == "current" and self.fusion_version != plan.actor_version:
            tick = time.monotonic()
            await self.wait(wg.initialize_fusion_student())
            self.metrics["perf/timing/fusion_student_initialize_seconds"] = time.monotonic() - tick
            tick = time.monotonic()
            await self.wait(
                wg.update_weights(
                    global_steps=plan.collection_step, target="fusion_student", version=plan.actor_version
                )
            )
            self.metrics["perf/timing/fusion_current_refit_seconds"] = time.monotonic() - tick
            self.fusion_version = plan.actor_version
        tick = time.monotonic()
        await self.wait(wg.prepare_fusion(alpha, plan.actor_version, phase_token=self.phase_token))
        self.metrics["perf/timing/fusion_prepare_residency_seconds"] = time.monotonic() - tick
        counts = [0] * wg.world_size
        n = self.config.actor_rollout_ref.rollout.n
        for request in plan.requests:
            counts[(request.row * n + request.response_index) % wg.world_size] += 1
        await self.guarded(
            asyncio.gather(
                *(
                    replica._server_handle.prepare.remote(count, self.phase_token)
                    for replica, count in zip(self.replicas["fusion"], counts, strict=True)
                )
            )
        )
        return self.clients["fusion"]

    async def prepare(self, phase, plan):
        tick = time.monotonic()
        self.active_plan = plan
        if phase.decoder == "fusion":
            self.client = await self.fusion_client(plan)
        elif phase.decoder != "fusion" and plan.has_current(phase.decoder):
            # Explicit on-demand sync; default validation target stays independent.
            await self.wait(self.trainer.actor_rollout_wg.update_weights(global_steps=plan.actor_version[0]))
            await self.set_current_sampling(True)
            self.client = self.trainer.get_llm_client()
        else:
            # A restored actor can still own model/optimizer storage. Frozen
            # generation must acquire the shared device only after offload.
            await self.wait(self.trainer.actor_rollout_wg.park_fusion_actor())
            path = plan.binding(phase.decoder).path
            counts = [0] * self.trainer.actor_rollout_wg.world_size
            for request in phase.requests:
                counts[
                    (request.row * self.config.actor_rollout_ref.rollout.n + request.response_index) % len(counts)
                ] += 1
            self.client = await self.frozen_client(phase.decoder, path, max(counts))
        self.metrics[f"perf/timing/rollout_policy_{phase.decoder}_prepare_seconds"] = time.monotonic() - tick

    async def generate(self, phase, plan):
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
            prompt["priority"] = request.row * self.config.actor_rollout_ref.rollout.n + request.response_index
            params = dict(plan.sampling) | {"seed": request.seed}
            if phase.decoder == "fusion":
                params["_fusion_phase_token"] = self.phase_token
            metadata = {
                "mode": self.policy.mode,
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
        workers = self.trainer.agent_loop_manager.agent_loop_workers
        chunks = [sessions[i :: len(workers)] for i in range(len(workers))]
        self.active_refs = [
            worker.generate_sessions.remote(chunk, self.client, self.policy.deadline_seconds, self.policy.drain_seconds)
            for worker, chunk in zip(workers, chunks, strict=True)
            if chunk
        ]
        results = await self.wait(self.active_refs)
        self.active_refs = []
        if phase.decoder == "fusion":
            rank_metrics = (
                await self.fusion_executor.metrics()
                if hasattr(self, "fusion_executor")
                else await self.wait(self.trainer.actor_rollout_wg.fusion_metrics())
            )
            for rank, values in enumerate(rank_metrics):
                self.metrics[f"perf/memory/fusion_gpu_{rank}_allocated_peak_bytes"] = values["peak_allocated_bytes"]
                self.metrics[f"perf/memory/fusion_gpu_{rank}_reserved_peak_bytes"] = values["peak_reserved_bytes"]
                if "microbatches" in values:
                    self.metrics[f"perf/memory/fusion_gpu_{rank}_kv_peak_bytes"] = max(
                        (row["kv_peak_bytes"] for row in values["microbatches"]), default=0
                    )
            self.rank_metrics = rank_metrics
        self.metrics[f"perf/timing/rollout_policy_{phase.decoder}_generate_seconds"] = time.monotonic() - tick
        return [item for result in results for item in result]

    async def drain(self):
        # A cancelled Ray ObjectRef can settle before its actor coroutine has
        # finished cleanup. Obtain an independent acknowledgement from every
        # possible TQ writer before clearing keys or switching GPU ownership.
        workers = self.trainer.agent_loop_manager.agent_loop_workers
        try:
            await self.guarded(
                asyncio.gather(*(worker.drain_sessions.remote(self.policy.drain_seconds) for worker in workers)),
                self.policy.drain_seconds,
            )
        except BaseException as exc:
            raise UnsafeDrainError("Cannot confirm all session writers stopped; terminate run") from exc
        self.active_refs = []
        if hasattr(self, "fusion_executor"):
            await self.fusion_executor.release()
        elif "fusion" in self.replicas:
            await self.guarded(asyncio.gather(*(r.sleep() for r in self.replicas["fusion"])), self.policy.drain_seconds)

    async def release(self, phase):
        tick = time.monotonic()
        if phase.decoder == "fusion" and hasattr(self, "fusion_executor"):
            await self.fusion_executor.release()
        elif phase.decoder != "fusion" and self.active_plan.has_current(phase.decoder):
            await self.set_current_sampling(False)
            await self.guarded(asyncio.to_thread(self.trainer.checkpoint_manager.sleep_replicas))
        else:
            await self.guarded(
                asyncio.gather(*(r.sleep() for r in self.replicas[phase.decoder])), self.policy.deadline_seconds
            )
            if phase.decoder == "fusion":
                await self.wait(self.trainer.actor_rollout_wg.release_fusion())
        self.metrics[f"perf/timing/rollout_policy_{phase.decoder}_release_seconds"] = time.monotonic() - tick

    async def release_all(self):
        await self.set_current_sampling(False)
        await self.guarded(asyncio.to_thread(self.trainer.checkpoint_manager.sleep_replicas))
        for source, replicas in self.replicas.items():
            if source == "fusion" and hasattr(self, "fusion_executor"):
                await self.fusion_executor.release()
                continue
            await self.guarded(asyncio.gather(*(r.sleep() for r in replicas)), self.policy.drain_seconds)
        if self.initialized_fusion:
            await self.wait(self.trainer.actor_rollout_wg.release_fusion())

    @auto_await
    async def collect(self, batch):
        self.batch = batch
        self.metrics = {}
        self.rank_metrics = []
        rollout = self.config.actor_rollout_ref.rollout
        sampling = {
            "temperature": rollout.temperature,
            "top_p": rollout.top_p,
            "top_k": rollout.top_k,
            "max_tokens": rollout.response_length,
            "logprobs": rollout.calculate_log_probs or self.policy.mode == "token_mixture",
            "repetition_penalty": 1.0,
        }
        stop_ids = list(
            self.trainer.tokenizer.eos_token_id
            if isinstance(self.trainer.tokenizer.eos_token_id, list)
            else [self.trainer.tokenizer.eos_token_id]
        )
        stop_ids.extend(rollout.stop_token_ids or [])
        plan = build_plan(
            self.policy,
            uids=list(batch["uid"]),
            stable_ids=[str(x) for x in batch["index"]],
            n=rollout.n,
            collection_step=self.trainer.global_steps,
            optimizer_step=self.trainer._optimizer_progress().applied,
            seed=rollout.seed,
            student_path=self.config.actor_rollout_ref.model.path,
            teacher_path=self.teacher_path(),
            bindings=self.bindings,
            sampling=sampling,
            stop_ids=tuple(sorted(set(stop_ids))),
        )
        plan_dir = Path(self.config.trainer.default_local_dir) / "rollout_plans"
        plan_dir.mkdir(parents=True, exist_ok=True)
        plan_path = plan_dir / f"step_{plan.collection_step}.json"
        plan_path.write_text(json.dumps(dataclasses.asdict(plan), ensure_ascii=False, indent=2) + "\n")
        tick = time.monotonic()
        await collect_plan(plan, self, TQPublisher(self.policy.deadline_seconds))
        self.metrics["perf/timing/rollout_policy_batch_seconds"] = time.monotonic() - tick
        self.metrics[f"rollout/policy/mode_{self.policy.mode}"] = 1
        if self.policy.mode == "token_mixture":
            self.metrics["rollout/policy/right_weight"] = plan.resolved_control
        elif self.policy.mode == "trajectory_mixture":
            self.metrics["rollout/policy/right_target_fraction"] = plan.resolved_control
            self.metrics["rollout/policy/right_effective_fraction"] = plan.effective_right_fraction
        fields = await self.guarded(
            tq.async_kv_batch_get(
                keys=[f"{request.uid}_{request.response_index}_0" for request in plan.requests],
                partition_id="train",
                select_fields=["response_mask"],
            )
        )
        source_tokens, source_counts = {}, {}
        for request, mask in zip(plan.requests, fields["response_mask"].unbind(), strict=True):
            source_tokens[request.source] = source_tokens.get(request.source, 0) + int(mask.sum())
            source_counts[request.source] = source_counts.get(request.source, 0) + 1
        for source, tokens in source_tokens.items():
            self.metrics[f"rollout/policy/{source}_response_tokens"] = tokens
            self.metrics[f"rollout/policy/{source}_requests"] = source_counts[source]
        if self.policy.mode == "trajectory_mixture":
            self.metrics["rollout/policy/right_token_fraction"] = source_tokens.get("right", 0) / sum(
                source_tokens.values()
            )
        self.metrics["rollout/policy/requests"] = len(plan.requests)
        (plan_dir / f"step_{plan.collection_step}_runtime.json").write_text(
            json.dumps({"metrics": self.metrics, "ranks": getattr(self, "rank_metrics", [])}, indent=2) + "\n"
        )
        self.last_plan = plan
        return self.metrics
