# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""One trainer-facing lifecycle; existing default transport remains delegated."""

import transfer_queue as tq

from verl.utils.debug import marked_timer
from verl.utils.ray_utils import auto_await

from .lifecycle import ExecutionContext


class RolloutService:
    def __init__(self, trainer):
        self.trainer = trainer
        self.metrics = {}

    def initialize(self):
        pass

    def after_update(self):
        pass

    def after_sample(self):
        pass

    def validation_begin(self):
        pass

    def validation_end(self):
        pass


class DefaultRolloutService(RolloutService):
    def __init__(self, trainer, submit):
        super().__init__(trainer)
        self.submit = submit
        self.common_sampling = trainer.common_rollout_sampling
        self.resident_version = None

    @auto_await
    async def configure_sampling(self):
        if self.common_sampling:
            trainer = self.trainer
            context = ExecutionContext(trainer.config, trainer.rollout_policy, trainer.actor_rollout_wg)
            await context.set_policy_sampling(trainer.checkpoint_manager.replicas, True)

    def initialize(self):
        self.ensure_current(self.trainer.global_steps)

    def ensure_current(self, version):
        if self.resident_version != version:
            self.trainer.checkpoint_manager.update_weights(version)
            self.configure_sampling()
            self.resident_version = version

    def park(self):
        self.trainer.checkpoint_manager.sleep_replicas()
        # Level-2 sleep discards weights; a matching version alone is insufficient.
        self.resident_version = None

    def after_update(self):
        if (
            self.trainer.config.trainer.get("evaluation")
            and self.trainer.global_steps >= self.trainer.total_training_steps
        ):
            return
        with marked_timer("update_weights", self.trainer.timing_raw, color="red"):
            self.ensure_current(self.trainer.global_steps)

    def after_sample(self):
        self.park()


class PolicyRolloutService(RolloutService):
    def __init__(self, trainer):
        super().__init__(trainer)
        from .runtime import RolloutRuntime

        self.runtime = RolloutRuntime(trainer)

    def validation_begin(self):
        self.runtime.sync_validation()

    def validation_end(self):
        self.runtime.sleep_validation()

    def submit(self, batch):
        trainer = self.trainer
        tags = [
            {"is_prompt": True, "status": "pending", "global_steps": trainer.global_steps} for _ in range(len(batch))
        ]
        tq.kv_batch_put(keys=list(batch["uid"]), partition_id="train", tags=tags)
        trainer._stamp_rollout_submission(batch)
        self.metrics = self.runtime.collect(batch)
        return len(batch)


def create_rollout_service(trainer, submit_default):
    return (
        DefaultRolloutService(trainer, submit_default)
        if trainer.rollout_policy.default_path
        else PolicyRolloutService(trainer)
    )
