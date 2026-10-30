# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Fusion returns actual selected-token probabilities independent of learner logging."""

import asyncio
from types import SimpleNamespace

import pytest

from verl.workers.rollout.replica import TokenOutput
from verl.workers.rollout.vllm_fusion.replica import FusionServer
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer


def server():
    instance = FusionServer.__new__(FusionServer)
    instance.fusion_accepting = True
    instance.fusion_phase = "phase"
    instance.fusion_seen = set()
    instance.fusion_expected = 1
    instance.fusion_loaded_version = (0, 0)
    instance.fusion_alpha = 0.5
    instance.fusion_teacher_path = "teacher"
    instance.fusion_space = "logit"
    instance.config = SimpleNamespace(response_length=2, max_model_len=16)
    return instance


@pytest.mark.parametrize("fusion_space", ["logit", "probability"])
def test_full_vocab_no_logprob_option_still_returns_fusion_behavior(monkeypatch, fusion_space):
    async def generate(self, request_id, prompt_ids, sampling_params, **kwargs):
        assert sampling_params["logprobs"] is True
        return TokenOutput(token_ids=[4, 5], log_probs=[-0.3, -0.4], stop_reason="completed")

    monkeypatch.setattr(vLLMHttpServer, "generate", generate)
    params = {"_fusion_phase_token": "phase", "logprobs": False, "temperature": 1.0}
    instance = server()
    instance.fusion_space = fusion_space
    result = asyncio.run(instance.generate("id", [1, 2], params))
    assert result.log_probs == [-0.3, -0.4]
    assert params["logprobs"] is False
    assert params["_fusion_phase_token"] == "phase"
    assert result.extra_fields["actor_version"] == (0, 0)
    assert result.extra_fields["fusion_space"] == fusion_space


@pytest.mark.parametrize("spaces", [["logit"], ["probability", "logit"], ["probability", "probability"]])
def test_probability_phase_never_opens_against_a_geometric_engine(spaces):
    instance = server()
    resumed = []

    async def idle():
        return None

    async def resume():
        resumed.append(True)

    async def rpc(name, args):
        assert name == "fusion_prepare"
        return [{"version": (0, 0), "teacher_path": "teacher", "fusion_space": value} for value in spaces]

    instance.wait_for_requests_to_drain = instance.clear_kv_cache = idle
    instance.resume_generation = resume
    instance.engine = SimpleNamespace(collective_rpc=rpc)
    if set(spaces) == {"probability"}:
        asyncio.run(instance.prepare(1, "new-phase", 0.5, (0, 0), "probability"))
        assert instance.fusion_accepting and instance.fusion_space == "probability" and resumed
    else:
        with pytest.raises(RuntimeError, match="space"):
            asyncio.run(instance.prepare(1, "new-phase", 0.5, (0, 0), "probability"))
        assert not instance.fusion_accepting and not resumed


@pytest.mark.parametrize("log_probs", [None, [-0.3], [-0.3, float("nan")]])
@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_incomplete_behavior_payload_cannot_be_a_success(monkeypatch, log_probs, temperature):
    async def generate(self, **kwargs):
        return TokenOutput(token_ids=[4, 5], log_probs=log_probs, stop_reason="completed")

    monkeypatch.setattr(vLLMHttpServer, "generate", generate)
    with pytest.raises(RuntimeError, match="behavior probabilities"):
        asyncio.run(server().generate("id", [1], {"_fusion_phase_token": "phase", "temperature": temperature}))
