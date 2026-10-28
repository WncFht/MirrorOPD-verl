# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os

import transfer_queue as tq

from verl.experimental.rollout_policy.plan import RolloutPolicyConfig
from verl.trainer.ppo.v1.trainer_base import PPOTrainer, register_trainer
from verl.utils.debug import marked_timer

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@register_trainer("sync")
class PPOTrainerSync(PPOTrainer):
    """Synchronous PPO trainer
    1. Trainer and rollout are colocated
    2. Partial rollout is disabled
    """

    rollout_policy = RolloutPolicyConfig()

    def on_init_end(self):
        if not self.rollout_policy.default_path:
            from verl.experimental.rollout_policy.runtime import RolloutRuntime

            self.rollout_runtime = RolloutRuntime(self)
            return
        # update weights after loading checkpoint
        self.checkpoint_manager.update_weights(self.global_steps)

    def on_step_end(self):
        if not self.rollout_policy.default_path:
            return
        with marked_timer("update_weights", self.timing_raw, color="red"):
            # wake up all replicas to update weights
            self.checkpoint_manager.update_weights(self.global_steps)

    def on_sample_end(self):
        # sleep all replicas to discard weights and kv cache
        if self.rollout_policy.default_path:
            self.checkpoint_manager.sleep_replicas()
        if self.curr_step_profile:
            self._stop_rollout_profiling()

    def on_validate_begin(self):
        if not self.rollout_policy.default_path:
            self.rollout_runtime.sync_validation()

    def on_validate_end(self):
        if not self.rollout_policy.default_path:
            self.rollout_runtime.sleep_validation()

    def _submit_batch_to_rollout(self, batch):
        if self.rollout_policy.default_path:
            return super()._submit_batch_to_rollout(batch)
        tags = [{"is_prompt": True, "status": "pending", "global_steps": self.global_steps} for _ in range(len(batch))]
        tq.kv_batch_put(keys=list(batch["uid"]), partition_id="train", tags=tags)
        self._stamp_rollout_submission(batch)
        self._policy_metrics = self.rollout_runtime.collect(batch)
        return len(batch)

    def prepare_step(self):
        result = super().prepare_step()
        if not self.rollout_policy.default_path:
            result.update(self._policy_metrics)
        return result
