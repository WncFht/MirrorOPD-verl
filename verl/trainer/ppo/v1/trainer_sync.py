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

from verl.experimental.rollout_policy.plan import RolloutPolicyConfig
from verl.trainer.ppo.v1.trainer_base import PPOTrainer, register_trainer

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
        from verl.experimental.rollout_policy.service import create_rollout_service

        self.rollout_service = create_rollout_service(self, super()._submit_batch_to_rollout)
        self.rollout_service.initialize()

    def on_step_end(self):
        self.rollout_service.after_update()

    def on_sample_end(self):
        self.rollout_service.after_sample()
        if self.curr_step_profile:
            self._stop_rollout_profiling()

    def on_validate_begin(self):
        self.rollout_service.validation_begin()

    def on_validate_end(self):
        self.rollout_service.validation_end()

    def _submit_batch_to_rollout(self, batch):
        return self.rollout_service.submit(batch)

    def prepare_step(self):
        result = super().prepare_step()
        result.update(self.rollout_service.metrics)
        return result
