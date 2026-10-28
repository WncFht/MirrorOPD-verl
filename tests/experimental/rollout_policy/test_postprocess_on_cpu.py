# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
from verl.trainer.ppo.v1 import agent_loop_tq


@pytest.mark.parametrize(
    "source,binding,alpha,version,has_current",
    [
        ("student", "current", 0.5, 0, True),
        ("teacher", "current", 0.5, None, False),
        ("fusion", "initial", 0.5, None, False),
        ("fusion", "current", 1.0, None, False),
    ],
)
@pytest.mark.parametrize("temperature", [0, 1])
def test_real_output_schema_and_source_versions(monkeypatch, source, binding, alpha, version, has_current, temperature):
    output = AgentLoopOutput(
        prompt_ids=[3, 4],
        response_ids=[5, 0],
        response_mask=[1, 1],
        response_logprobs=[-0.3, -0.7],
        num_turns=2,
        metrics={},
        extra_fields={"min_global_steps": version, "max_global_steps": version, "finish_reason": "eos"},
    )
    writer = AsyncMock()
    monkeypatch.setattr(agent_loop_tq.tq, "async_kv_batch_put", writer)
    worker = SimpleNamespace(
        _compute_score=AsyncMock(),
        _compute_teacher_logprobs=AsyncMock(),
        _compute_multi_modal_inputs=lambda *args: {},
        _compute_position_ids=lambda ids, *args: torch.arange(ids.shape[1]).unsqueeze(0),
    )
    method = agent_loop_tq.AgentLoopWorkerTQ.__ray_metadata__.modified_class._agent_loop_postprocess
    asyncio.run(
        method(
            worker,
            output,
            False,
            uid="u",
            session_id=0,
            global_steps=1,
            __rollout_policy__={
                "source": source,
                "student": binding,
                "teacher_weight": alpha,
                "actor_version": (0, 0),
                "temperature": temperature,
            },
        )
    )
    tag = writer.call_args.kwargs["tags"][0]
    assert tag["rollout_has_current_component"] == has_current
    assert tag["min_global_steps"] == version
    assert writer.call_args.kwargs["keys"] == ["u_0_0"]
    assert output.response_logprobs == ([0.0, 0.0] if temperature == 0 else [-0.3, -0.7])


@pytest.mark.parametrize("reason,mask,version", [("aborted", [1], 0), ("length", [0], 0), ("length", [1], 1)])
def test_incomplete_or_wrong_version_never_publishes(monkeypatch, reason, mask, version):
    output = AgentLoopOutput(
        prompt_ids=[3],
        response_ids=[5],
        response_mask=mask,
        num_turns=2,
        metrics={},
        extra_fields={"min_global_steps": version, "max_global_steps": version, "finish_reason": reason},
    )
    writer = AsyncMock()
    monkeypatch.setattr(agent_loop_tq.tq, "async_kv_batch_put", writer)
    worker = SimpleNamespace(
        _compute_score=AsyncMock(),
        _compute_teacher_logprobs=AsyncMock(),
        _compute_multi_modal_inputs=lambda *args: {},
        _compute_position_ids=lambda ids, *args: torch.arange(ids.shape[1]).unsqueeze(0),
    )
    method = agent_loop_tq.AgentLoopWorkerTQ.__ray_metadata__.modified_class._agent_loop_postprocess
    with pytest.raises(RuntimeError):
        asyncio.run(
            method(
                worker,
                output,
                False,
                uid="u",
                session_id=0,
                global_steps=1,
                __rollout_policy__={
                    "source": "student",
                    "student": "current",
                    "teacher_weight": 0.5,
                    "actor_version": (0, 0),
                },
            )
        )
    writer.assert_not_called()
