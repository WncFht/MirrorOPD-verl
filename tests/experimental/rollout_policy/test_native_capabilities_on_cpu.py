# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Reject unaccepted native modes before loading checkpoints or creating GPU workers."""

import pytest
from omegaconf import OmegaConf

from verl.experimental.rollout_policy.validation import validate_policy


@pytest.mark.parametrize(
    "key,value",
    [
        ("quantization", "fp8"),
        ("quantization_config_file", "quant.json"),
        ("engine_kwargs.vllm.quantization", "fp8"),
        ("engine_kwargs.vllm.speculative_config", {"method": "ngram"}),
        ("engine_kwargs.vllm.enable_eplb", True),
        ("engine_kwargs.vllm.kv_cache_dtype", "fp8"),
    ],
)
def test_unaccepted_modes_fail_before_model_loading(key, value):
    config = OmegaConf.create(
        {
            "rollout_policy": {"mode": "token_mixture", "execution": {"backend": "vllm_fusion"}},
            "actor_rollout_ref": {"rollout": {"name": "vllm"}},
        }
    )
    OmegaConf.update(config, "actor_rollout_ref.rollout." + key, value, force_add=True)
    with pytest.raises(ValueError, match="Native fusion"):
        validate_policy(config)


def test_default_single_does_not_consume_fusion_capability_policy():
    config = OmegaConf.create(
        {
            "rollout_policy": {"execution": {"backend": "vllm_fusion"}},
            "actor_rollout_ref": {"rollout": {"name": "vllm", "quantization": "fp8"}},
        }
    )
    assert validate_policy(config).default_path
