# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent validation budgets across requests, token outputs and batch storage."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import NonTensorData, TensorDict

import verl.trainer.ppo.v1.agent_loop_tq as tq_module
from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput, AgentLoopWorker
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.workers.config.rollout import SamplingConfig
from verl.workers.rollout.replica import TokenOutput


@pytest.mark.parametrize("value", [None, 1, 31744])
def test_sampling_budget_accepts_optional_positive_integer(value):
    assert SamplingConfig(max_tokens=value).max_tokens == value


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "128"])
def test_sampling_budget_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="max_tokens"):
        SamplingConfig(max_tokens=value)


@pytest.mark.parametrize("budget,generated,expected", [(None, 8, 3), (7, 8, 7), (2, 8, 2), (7, 2, 2)])
def test_single_turn_preserves_effective_response_budget(budget, generated, expected):
    loop = object.__new__(SingleTurnAgentLoop)
    loop.response_length = 3
    loop.rollout_config = SimpleNamespace(full_determinism=False)
    loop.process_multi_modal_info = AsyncMock(return_value={})
    loop._get_mm_processor_kwargs = lambda _: None
    loop._assert_mm_supported = lambda _: None
    loop.ct_build_initial_tokens = AsyncMock(return_value=[101, 102])
    captured = {}

    async def generate(**kwargs):
        # Exercise a consumer that mutates its input: the output budget must
        # already have been captured by SingleTurnAgentLoop.
        captured.update(kwargs["sampling_params"])
        kwargs["sampling_params"].pop("max_tokens", None)
        return TokenOutput(
            token_ids=list(range(generated)),
            log_probs=[-0.5] * generated,
            routed_experts=np.zeros((2 + generated, 1, 1), dtype=np.int16),
        )

    async def merge(prompt, response, mask, logprobs, assistant_logprobs):
        return SimpleNamespace(token_ids=prompt + response), [1] * len(response), assistant_logprobs

    loop.server_manager = SimpleNamespace(generate=generate)
    loop.ct_merge_assistant_token = merge
    params = {} if budget is None else {"max_tokens": budget}
    output = asyncio.run(loop.run(params, raw_prompt=[{"role": "user", "content": "hi"}]))
    assert len(output.response_ids) == len(output.response_mask) == len(output.response_logprobs) == expected
    assert output.routed_experts.shape[0] == 2 + expected
    assert captured.get("max_tokens") == budget
    assert loop.response_length == 3


@pytest.mark.parametrize("validate,budget,expected", [(False, 7, None), (True, 7, 7), (True, None, None)])
def test_v1_sends_validation_override_only(monkeypatch, validate, budget, expected):
    async def run():
        cfg = OmegaConf.create(
            {
                "actor_rollout_ref": {
                    "rollout": {
                        "temperature": 1.0,
                        "top_p": 1.0,
                        "top_k": -1,
                        "calculate_log_probs": True,
                        "topk_log_probs": 0,
                        "agent": {"default_agent_loop": "single_turn_agent"},
                        "val_kwargs": {"temperature": 0.7, "top_p": 0.9, "top_k": -1, "max_tokens": budget},
                    }
                }
            }
        )
        worker = SimpleNamespace(config=cfg, background_tasks=set(), _run_prompt=AsyncMock())
        monkeypatch.setattr(tq_module, "get_trajectory_info", AsyncMock(return_value=[{"validate": validate}]))
        batch = TensorDict(
            {"index": torch.tensor([0]), "global_steps": NonTensorData(1), "validate": NonTensorData(validate)},
            batch_size=[1],
        )
        cls = tq_module.AgentLoopWorkerTQ.__ray_metadata__.modified_class
        await cls.generate_sequences(worker, batch)
        await asyncio.gather(*list(worker.background_tasks))
        params = worker._run_prompt.call_args.args[1]
        assert params.get("max_tokens") == expected
        assert params["temperature"] == (0.7 if validate else 1.0)

    asyncio.run(run())


def test_custom_agent_rejects_unimplemented_validation_budget():
    worker = object.__new__(AgentLoopWorker)
    with pytest.raises(ValueError, match="single_turn_agent"):
        asyncio.run(worker._run_agent_loop({"max_tokens": 7}, {"validate": True}, agent_name="tool_agent"))


class _Tokenizer:
    pad_token_id = 0
    padding_side = "right"

    def pad(self, data, *, max_length, return_attention_mask, **kwargs):
        values = data["input_ids"]
        pad = [0] * (max_length - len(values))
        mask = [1] * len(values)
        if self.padding_side == "left":
            values, mask = pad + values, pad + mask
        else:
            values, mask = values + pad, mask + pad
        result = {"input_ids": torch.tensor([values])}
        if return_attention_mask:
            result["attention_mask"] = torch.tensor([mask])
        return result


@pytest.mark.parametrize("validate,response_count,width", [(False, 2, 3), (True, 6, 7)])
def test_legacy_padding_uses_validation_width(validate, response_count, width):
    class Worker:
        _pad_token_ids = AgentLoopWorker._pad_token_ids

        def __init__(self):
            self.tokenizer = _Tokenizer()
            self.rollout_config = OmegaConf.create(
                {"prompt_length": 4, "response_length": 3, "val_kwargs": {"max_tokens": 7}}
            )
            self._compute_score = AsyncMock()
            self._compute_teacher_logprobs = AsyncMock()

        def _compute_multi_modal_inputs(self, output, input_ids):
            return {}

        def _get_mm_processor_kwargs(self, *args):
            return None

        def _compute_position_ids(self, input_ids, *args):
            return torch.arange(input_ids.shape[-1]).unsqueeze(0)

    output = AgentLoopOutput(
        prompt_ids=[101, 102],
        response_ids=[7] * response_count,
        response_mask=[1] * response_count,
        response_logprobs=[-0.5] * response_count,
        metrics=AgentLoopMetrics(),
    )
    internal = asyncio.run(
        AgentLoopWorker._agent_loop_postprocess(
            Worker(), output, validate, raw_prompt=[{"role": "user", "content": "hi"}]
        )
    )
    assert internal.response_ids.shape == internal.response_mask.shape == internal.response_logprobs.shape == (1, width)
    assert internal.input_ids.shape == (1, 4 + width)
    assert internal.response_mask.sum().item() == response_count


def test_v1_stores_long_validation_unpadded(monkeypatch):
    captured = {}

    async def put(**kwargs):
        captured.update(kwargs)

    worker = SimpleNamespace(
        _compute_score=AsyncMock(),
        _compute_teacher_logprobs=AsyncMock(),
        _compute_multi_modal_inputs=lambda output, ids: {},
        _compute_position_ids=lambda ids, mask, mm: torch.arange(ids.shape[-1]).unsqueeze(0),
    )
    monkeypatch.setattr(tq_module.tq, "async_kv_batch_put", put)
    output = AgentLoopOutput(
        prompt_ids=[101, 102], response_ids=[7] * 7, response_mask=[1] * 7, metrics=AgentLoopMetrics(), num_turns=2
    )
    cls = tq_module.AgentLoopWorkerTQ.__ray_metadata__.modified_class
    asyncio.run(cls._agent_loop_postprocess(worker, output, True, uid="u0", session_id=0, global_steps=1))
    assert captured["partition_id"] == "val"
    assert captured["fields"]["responses"][0].numel() == 7
    assert captured["tags"][0]["response_len"] == 7
