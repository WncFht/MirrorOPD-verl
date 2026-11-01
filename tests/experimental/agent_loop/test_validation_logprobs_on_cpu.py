# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Validation keeps its sampling budget while omitting training-only probabilities."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import NonTensorData, TensorDict

import verl.experimental.agent_loop.agent_loop as loop_module
import verl.trainer.ppo.v1.agent_loop_tq as tq_module
from verl.protocol import DataProto


def rollout_config(calculate_log_probs):
    return OmegaConf.create(
        {
            "name": "vllm",
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "calculate_log_probs": calculate_log_probs,
            "topk_log_probs": 0,
            "agent": {"default_agent_loop": "single_turn_agent"},
            "val_kwargs": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": -1,
                "max_tokens": 31744,
                "repetition_detection": {"min_pattern_size": 16, "max_pattern_size": 128, "min_count": 8},
            },
        }
    )


@pytest.mark.parametrize("validate", [False, True])
@pytest.mark.parametrize("calculate_log_probs", [False, True])
def test_tq_sampling_keeps_training_logprobs_and_validation_budget(monkeypatch, validate, calculate_log_probs):
    async def run():
        config = OmegaConf.create({"actor_rollout_ref": {"rollout": rollout_config(calculate_log_probs)}})
        worker = SimpleNamespace(config=config, background_tasks=set(), _run_prompt=AsyncMock())
        monkeypatch.setattr(tq_module, "get_trajectory_info", AsyncMock(return_value=[{"validate": validate}]))
        batch = TensorDict(
            {"index": torch.tensor([0]), "global_steps": NonTensorData(1), "validate": NonTensorData(validate)},
            batch_size=[1],
        )
        cls = tq_module.AgentLoopWorkerTQ.__ray_metadata__.modified_class
        await cls.generate_sequences(worker, batch)
        await asyncio.gather(*list(worker.background_tasks))
        return worker._run_prompt.call_args.args[1]

    params = asyncio.run(run())
    assert params["logprobs"] == (calculate_log_probs and not validate)
    assert params["temperature"] == (0.7 if validate else 1.0)
    assert params["top_p"] == (0.9 if validate else 1.0)
    assert params.get("max_tokens") == (31744 if validate else None)
    assert ("repetition_detection" in params) == validate
    if validate:
        assert params["repetition_detection"] == {"min_pattern_size": 16, "max_pattern_size": 128, "min_count": 8}


@pytest.mark.parametrize("validate", [False, True])
@pytest.mark.parametrize("calculate_log_probs", [False, True])
def test_legacy_sampling_has_the_same_probability_contract(monkeypatch, validate, calculate_log_probs):
    worker = SimpleNamespace(
        rollout_config=rollout_config(calculate_log_probs),
        _run_agent_loop=AsyncMock(return_value="result"),
        _postprocess=lambda outputs, **kwargs: outputs,
    )
    monkeypatch.setattr(loop_module, "get_trajectory_info", AsyncMock(return_value=[{"validate": validate}]))
    monkeypatch.setattr(
        loop_module.RolloutTraceConfig, "get_instance", lambda: SimpleNamespace(max_samples_per_step_per_worker=None)
    )
    batch = DataProto(
        batch=TensorDict({"input_ids": torch.tensor([[1]])}, batch_size=[1]),
        non_tensor_batch={"index": np.array([0])},
        meta_info={"validate": validate, "global_steps": 1},
    )
    result = asyncio.run(loop_module.AgentLoopWorker.generate_sequences(worker, batch))
    assert result == ["result"]
    params = worker._run_agent_loop.call_args.args[0]
    assert params["logprobs"] == (calculate_log_probs and not validate)
    assert params["temperature"] == (0.7 if validate else 1.0)
    assert params["top_p"] == (0.9 if validate else 1.0)
    assert params.get("max_tokens") == (31744 if validate else None)
    assert ("repetition_detection" in params) == validate
    if validate:
        assert params["repetition_detection"] == {"min_pattern_size": 16, "max_pattern_size": 128, "min_count": 8}


@pytest.mark.parametrize("validate", [False, True])
def test_training_repetition_is_independent_on_tq_path(monkeypatch, validate):
    train_stop = {"min_pattern_size": 32, "max_pattern_size": 256, "min_count": 10}

    async def run():
        rollout = rollout_config(False)
        rollout.repetition_detection = train_stop
        rollout.val_kwargs.repetition_detection = None
        config = OmegaConf.create({"actor_rollout_ref": {"rollout": rollout}})
        worker = SimpleNamespace(config=config, background_tasks=set(), _run_prompt=AsyncMock())
        monkeypatch.setattr(tq_module, "get_trajectory_info", AsyncMock(return_value=[{"validate": validate}]))
        batch = TensorDict(
            {"index": torch.tensor([0]), "global_steps": NonTensorData(1), "validate": NonTensorData(validate)},
            batch_size=[1],
        )
        cls = tq_module.AgentLoopWorkerTQ.__ray_metadata__.modified_class
        await cls.generate_sequences(worker, batch)
        await asyncio.gather(*worker.background_tasks)
        return worker._run_prompt.call_args.args[1]

    params = asyncio.run(run())
    assert params.get("repetition_detection") == (None if validate else train_stop)


@pytest.mark.parametrize("validate", [False, True])
def test_training_repetition_is_independent_on_legacy_path(monkeypatch, validate):
    train_stop = {"min_pattern_size": 32, "max_pattern_size": 256, "min_count": 10}
    rollout = rollout_config(False)
    rollout.repetition_detection = train_stop
    rollout.val_kwargs.repetition_detection = None
    worker = SimpleNamespace(
        rollout_config=rollout,
        _run_agent_loop=AsyncMock(return_value="result"),
        _postprocess=lambda outputs, **kwargs: outputs,
    )
    monkeypatch.setattr(loop_module, "get_trajectory_info", AsyncMock(return_value=[{"validate": validate}]))
    monkeypatch.setattr(
        loop_module.RolloutTraceConfig, "get_instance", lambda: SimpleNamespace(max_samples_per_step_per_worker=None)
    )
    batch = DataProto(
        batch=TensorDict({"input_ids": torch.tensor([[1]])}, batch_size=[1]),
        non_tensor_batch={"index": np.array([0])},
        meta_info={"validate": validate, "global_steps": 1},
    )
    asyncio.run(loop_module.AgentLoopWorker.generate_sequences(worker, batch))
    assert worker._run_agent_loop.call_args.args[0].get("repetition_detection") == (None if validate else train_stop)


def test_training_repetition_rejects_unsupported_backend():
    from verl.workers.config.rollout import rollout_repetition_detection

    config = rollout_config(False)
    config.name = "hf"
    config.repetition_detection = {"min_pattern_size": 16, "max_pattern_size": 128, "min_count": 8}
    with pytest.raises(ValueError, match="rollout.repetition_detection requires rollout.name=vllm"):
        rollout_repetition_detection(config, validate=False)
