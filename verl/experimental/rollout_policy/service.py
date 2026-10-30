# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""One trainer-facing lifecycle; existing default transport remains delegated."""

import transfer_queue as tq

from verl.utils.debug import marked_timer


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

    def initialize(self):
        self.trainer.checkpoint_manager.update_weights(self.trainer.global_steps)

    def after_update(self):
        with marked_timer("update_weights", self.trainer.timing_raw, color="red"):
            self.trainer.checkpoint_manager.update_weights(self.trainer.global_steps)

    def after_sample(self):
        self.trainer.checkpoint_manager.sleep_replicas()


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
