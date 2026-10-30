# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Build immutable batches and delegate execution and publication to their owners."""

import dataclasses
import json
import time
from pathlib import Path

import transfer_queue as tq

from verl.utils.ray_utils import auto_await

from .collector import collect_plan
from .lifecycle import ExecutionContext
from .phases import BatchExecution, ExecutionPool
from .plan import build_plan
from .sessions import SessionDriver, TQPublisher


class RolloutRuntime:
    def __init__(self, trainer):
        self.trainer = trainer
        self.config = trainer.config
        self.policy = trainer.rollout_policy
        self.bindings = trainer.rollout_bindings
        self.context = ExecutionContext(
            self.config, self.policy, trainer.actor_rollout_wg, trainer.checkpoint_manager, trainer.tokenizer
        )
        self.pool = ExecutionPool(self.context)

    @auto_await
    async def sync_validation(self):
        await self.pool.single.validation_begin(self.trainer.global_steps)

    @auto_await
    async def sleep_validation(self):
        await self.pool.single.validation_end()

    @auto_await
    async def collect(self, batch):
        sessions = SessionDriver(self.context, self.trainer.agent_loop_manager.agent_loop_workers, batch)
        execution = BatchExecution(self.pool, sessions)
        metrics = sessions.metrics
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
            bindings=self.bindings,
            sampling=sampling,
            stop_ids=tuple(sorted(set(stop_ids))),
        )
        plan_dir = Path(self.config.trainer.default_local_dir) / "rollout_plans"
        plan_dir.mkdir(parents=True, exist_ok=True)
        plan_path = plan_dir / f"step_{plan.collection_step}.json"
        plan_path.write_text(json.dumps(dataclasses.asdict(plan), ensure_ascii=False, indent=2) + "\n")
        tick = time.monotonic()
        await collect_plan(plan, execution, TQPublisher(self.policy.deadline_seconds))
        metrics["perf/timing/rollout_policy_batch_seconds"] = time.monotonic() - tick
        metrics[f"rollout/policy/mode_{self.policy.mode}"] = 1
        if self.policy.mode == "token_mixture":
            metrics["rollout/policy/right_weight"] = plan.resolved_control
        elif self.policy.mode == "trajectory_mixture":
            metrics["rollout/policy/right_target_fraction"] = plan.resolved_control
            metrics["rollout/policy/right_effective_fraction"] = plan.effective_right_fraction
        fields = await self.context.guarded(
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
            metrics[f"rollout/policy/{source}_response_tokens"] = tokens
            metrics[f"rollout/policy/{source}_requests"] = source_counts[source]
        if self.policy.mode == "trajectory_mixture":
            metrics["rollout/policy/right_token_fraction"] = source_tokens.get("right", 0) / sum(source_tokens.values())
        metrics["rollout/policy/requests"] = len(plan.requests)
        (plan_dir / f"step_{plan.collection_step}_runtime.json").write_text(
            json.dumps({"metrics": metrics, "ranks": sessions.rank_metrics}, indent=2) + "\n"
        )
        self.last_plan = plan
        return metrics
