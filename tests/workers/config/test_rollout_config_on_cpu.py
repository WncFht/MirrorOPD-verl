# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config.rollout import RolloutConfig


def test_max_num_batched_tokens_raised_when_chunked_prefill_disabled():
    # vLLMHttpServer._validate_configs raises max_num_batched_tokens to max_model_len
    # when chunked prefill is off; that write must not hit the frozen-field guard.
    config = RolloutConfig(name="vllm", enable_chunked_prefill=False, max_num_batched_tokens=8192)
    config.max_model_len = 32768
    config.max_num_batched_tokens = config.max_model_len
    assert config.max_num_batched_tokens == 32768


@pytest.mark.parametrize("value", [None, [], [0], [151645, 151645]])
def test_additional_stop_ids_accept_optional_integer_list(value):
    assert RolloutConfig(name="vllm", stop_token_ids=value).stop_token_ids == value


@pytest.mark.parametrize("value", [151645, "151645", ["151645"], [True], [-1], [1.5], {1: 2}])
def test_additional_stop_ids_reject_invalid_types(value):
    with pytest.raises(ValueError, match="nonnegative integers"):
        RolloutConfig(name="vllm", stop_token_ids=value)


@pytest.mark.parametrize("backend", ["sglang", "hf", "trtllm"])
def test_additional_stop_ids_fail_on_unsupported_backends(backend):
    with pytest.raises(ValueError, match="requires rollout.name='vllm'"):
        RolloutConfig(name=backend, stop_token_ids=[151645])
    assert RolloutConfig(name=backend, stop_token_ids=[]).stop_token_ids == []


def test_stop_ids_survive_hydra_and_resolved_config_roundtrip():
    config_dir = Path(__file__).resolve().parents[3] / "verl/trainer/config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(
            config_name="ppo_trainer",
            overrides=[
                "actor_rollout_ref.rollout.name=vllm",
                "actor_rollout_ref.rollout.stop_token_ids=[151645]",
            ],
        )
    saved = OmegaConf.to_yaml(config.actor_rollout_ref.rollout, resolve=True)
    restored = omega_conf_to_dataclass(OmegaConf.create(saved))
    assert restored.stop_token_ids == [151645]
    teacher = config.distillation.teacher_models.teacher_model.inference
    assert teacher.get("stop_token_ids") is None
