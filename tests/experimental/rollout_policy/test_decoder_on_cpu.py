# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from verl.experimental.rollout_policy.hf_decoder import DecodeRequest, HFFusionDecoder
from verl.experimental.rollout_policy.sampling import SamplingConfig, fusion_log_probs


def models():
    torch.manual_seed(7)

    def make(layers):
        return Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=17,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=layers,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                eos_token_id=0,
                pad_token_id=0,
            )
        ).eval()

    return make(1), make(2)


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_cached_matches_full_prefix_with_unequal_lengths(compact, alpha):
    student, teacher = models()
    requests = [DecodeRequest("a", (0, 1, 2), 1), DecodeRequest("b", (3,), 2)]
    config = SamplingConfig(max_tokens=5, stop_ids=(0,))
    result = HFFusionDecoder(student, teacher, pad_id=0, compact=compact).generate(requests, alpha, config)
    for request, response in zip(requests, result, strict=True):
        prefix = list(request.prompt_ids)
        for token, logprob in zip(response.token_ids, response.log_probs, strict=True):
            with torch.no_grad():
                ids = torch.tensor([prefix])
                s = student(ids).logits[:, -1]
                t = teacher(ids).logits[:, -1]
                oracle = fusion_log_probs(s, t, alpha, config)[0, token]
            assert logprob == pytest.approx(oracle.item(), abs=2e-6)
            prefix.append(token)
        assert len(response.log_probs) == len(response.token_ids)


def test_first_token_eos_and_cap_one():
    student, teacher = models()
    request = DecodeRequest("a", (0, 1), 1)
    decoder = HFFusionDecoder(student, teacher, pad_id=0)
    for stop, cap in [((0,), 5), ((), 1)]:
        config = SamplingConfig(max_tokens=cap, stop_ids=stop, legal_ids=(0,))
        output = decoder.generate([request], 0.5, config)[0]
        assert output.token_ids == [0]
        assert output.log_probs == [0.0]
        assert output.finish_reason == ("eos" if stop else "length")


@pytest.mark.parametrize("power_of_two", [False, True])
def test_microbatch_streams_and_compaction_preserve_requests(power_of_two):
    student, teacher = models()
    requests = [DecodeRequest(str(i), tuple(range(1, i + 2)), i) for i in range(4)]
    config = SamplingConfig(max_tokens=12, stop_ids=(0, 1, 2, 3, 4, 5, 6))
    fixed = HFFusionDecoder(student, teacher, pad_id=0).generate(requests, 0.5, config)
    compact = HFFusionDecoder(student, teacher, pad_id=0, compact=True, compact_power_of_two=power_of_two).generate(
        requests, 0.5, config
    )
    single = [HFFusionDecoder(student, teacher, pad_id=0).generate([r], 0.5, config)[0] for r in requests]
    assert [r.token_ids for r in fixed] == [r.token_ids for r in compact] == [r.token_ids for r in single]
