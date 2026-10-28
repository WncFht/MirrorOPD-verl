# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
from types import MethodType

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Flash KV/CUDA graph validation")


def test_graph_compaction_preserves_prefixes_probabilities_and_request_order():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from verl.experimental.rollout_policy.hf_decoder import DecodeRequest, HFFusionDecoder
    from verl.experimental.rollout_policy.precision import configure_precision
    from verl.experimental.rollout_policy.sampling import SamplingConfig

    torch.manual_seed(17)
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        pad_token_id=1,
        eos_token_id=1,
        _attn_implementation="flash_attention_2",
    )
    models = [Qwen3ForCausalLM(config).eval().bfloat16().cuda() for _ in range(2)]
    for model in models:
        configure_precision(model, "stable")
    requests = [DecodeRequest(str(i), tuple([3] * (i + 1) + [1, 4]), i) for i in range(4)]
    sampling = SamplingConfig(max_tokens=8, stop_ids=(1,), legal_ids=tuple(range(32)))

    def forced(self, log_probs, generators, step):
        values = [1 if g.initial_seed() < 3 and step == g.initial_seed() else 2 for g in generators]
        tokens = torch.tensor(values, device=log_probs.device)
        return tokens, log_probs.gather(-1, tokens[:, None]).squeeze(-1)

    outputs = []
    for graph, compact, append in [(False, 0, False), (False, 1, True), (True, 1, True), (True, 2, True)]:
        decoder = HFFusionDecoder(
            *models,
            pad_id=1,
            compact=compact,
            compact_power_of_two=compact == 2,
            preallocate_kv=append,
            flash_kv=append,
            flash_kv_splits=1,
            cuda_graph=graph,
        )
        decoder._select_tokens = MethodType(forced, decoder)
        outputs.append(decoder.generate(requests, 0.5, sampling))
        assert not decoder.graphs
    for result in outputs:
        assert [row.request_id for row in result] == [str(i) for i in range(4)]
        assert [row.token_ids for row in result] == [[1], [2, 1], [2, 2, 1], [2] * 8]
        assert [row.finish_reason for row in result] == ["eos", "eos", "eos", "length"]
        for actual, reference in zip(result, outputs[0], strict=True):
            torch.testing.assert_close(
                torch.tensor(actual.log_probs), torch.tensor(reference.log_probs), rtol=0, atol=0.02
            )
