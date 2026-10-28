# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from verl.experimental.rollout_policy.hf_decoder import DecodeRequest, HFFusionDecoder
from verl.experimental.rollout_policy.kv_cache import AppendLayer
from verl.experimental.rollout_policy.sampling import SamplingConfig


def test_append_storage_reorder_reset_and_capacity():
    cache = AppendLayer(8)
    first = torch.arange(3 * 2 * 4 * 2).reshape(3, 2, 4, 2).float()
    keys, values = cache.update(first, first + 1)
    torch.testing.assert_close(keys, first)
    torch.testing.assert_close(values, first + 1)
    address = cache.key_buffer.data_ptr()
    cache.update(first[:, :, :1] + 100, first[:, :, :1] + 101)
    assert cache.key_buffer.data_ptr() == address
    expected = torch.cat([first, first[:, :, :1] + 100], dim=2)[[2, 0]]
    cache.batch_select_indices(torch.tensor([2, 0]))
    torch.testing.assert_close(cache.keys, expected)
    cache.crop(-1)
    assert cache.get_seq_length() == 4
    with pytest.raises(ValueError, match="capacity"):
        cache.update(expected, expected)
    cache.reset()
    assert cache.get_seq_length() == 0
    cache.update(first[:2, :, :1], first[:2, :, :1])
    torch.testing.assert_close(cache.keys, first[:2, :, :1])


@pytest.mark.parametrize("compact", [False, True])
def test_append_and_dynamic_dual_cache_have_same_trajectories(compact):
    torch.manual_seed(7)

    def model(layers):
        return Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=17,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=layers,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
            )
        ).eval()

    student, teacher = model(1), model(2)
    requests = [DecodeRequest(str(i), tuple(range(1, i + 2)), i) for i in range(4)]
    config = SamplingConfig(max_tokens=16, stop_ids=(0, 1, 2, 3, 4, 5, 6))
    reference = HFFusionDecoder(student, teacher, pad_id=0, compact=compact).generate(requests, 0.5, config)
    candidate = HFFusionDecoder(student, teacher, pad_id=0, compact=compact, preallocate_kv=True).generate(
        requests, 0.5, config
    )
    assert [row.token_ids for row in candidate] == [row.token_ids for row in reference]
    for actual, expected in zip(candidate, reference, strict=True):
        torch.testing.assert_close(torch.tensor(actual.log_probs), torch.tensor(expected.log_probs), rtol=0, atol=1e-6)


def test_graph_position_rewrites_one_slot_without_advancing_warmups():
    cache = AppendLayer(8)
    prefix = torch.arange(8).reshape(1, 1, 4, 2).float()
    cache.update(prefix, prefix)
    cache.graph_position = torch.tensor([4])
    cache.length = 5
    token = torch.tensor([[[[100.0, 101.0]]]])
    for _ in range(3):
        cache.update(token, token)
    assert cache.get_seq_length() == 5
    torch.testing.assert_close(cache.keys[:, :, :5], torch.cat([prefix, token], dim=2))
    cache.reset()
    assert cache.graph_position is None
    cache.update(token, token)
    assert cache.get_seq_length() == 1
