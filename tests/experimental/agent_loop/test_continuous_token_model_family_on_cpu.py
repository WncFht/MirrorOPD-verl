# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Exercise chat-protocol selection through the actual agent-loop constructor."""

import asyncio
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl.experimental.agent_loop.agent_loop import DictConfigWrap
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.utils.tokenizer.continuous_token import DeepSeekContinuousTokenBuilder, QwenContinuousTokenBuilder


@pytest.mark.parametrize("family", [None, "auto", "deepseek"])
def test_agent_loop_selects_chat_protocol_independently_of_qwen2_architecture(family):
    # The distill tokenizer deliberately has no ChatML end token.
    tokens = {"<｜end▁of▁sentence｜>" if family == "deepseek" else "<|im_end|>": 1}
    tokenizer = SimpleNamespace(
        unk_token_id=None,
        convert_tokens_to_ids=tokens.get,
        encode=lambda text, **kwargs: [2],
    )
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {"rollout": {"prompt_length": 32, "response_length": 32}},
            "data": {} if family is None else {"continuous_token": {"model_family": family}},
        }
    )

    async def construct():
        return SingleTurnAgentLoop(
            trainer_config=DictConfigWrap(config),
            server_manager=None,
            tokenizer=tokenizer,
            processor=None,
            dataset_cls=None,
            data_config=DictConfigWrap(config.data),
            hf_model_type="qwen2",
        )

    agent = asyncio.run(construct())
    expected = DeepSeekContinuousTokenBuilder if family == "deepseek" else QwenContinuousTokenBuilder
    assert isinstance(agent.continuous_token_builder, expected)
