# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Storage dtype conversion must preserve initialized nonparameter values."""

import pytest
import torch

from verl.utils.torch_dtypes import cast_module_parameters


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float64])
def test_parameter_cast_preserves_buffer_values_aliases_and_persistence(dtype):
    module = torch.nn.Linear(3, 2)
    module.weight.grad = torch.ones_like(module.weight)
    original = torch.tensor([0.10001, 0.30003], dtype=torch.float32)
    module.register_buffer("frequency", original, persistent=False)
    module.register_buffer("counter", torch.tensor(7))
    module.register_buffer("empty", None, persistent=False)
    module.child = torch.nn.Module()
    module.child.register_buffer("shared_frequency", original)
    assert cast_module_parameters(module, dtype) is module
    assert module.weight.dtype == module.bias.dtype == module.weight.grad.dtype == dtype
    assert module.frequency is module.child.shared_frequency is original
    assert torch.equal(module.frequency, torch.tensor([0.10001, 0.30003]))
    assert module.counter.dtype == torch.int64 and module.counter.item() == 7
    assert module.empty is None and "frequency" not in module.state_dict()
    assert "child.shared_frequency" in module.state_dict()


def test_parameter_cast_supports_meta_parameters_and_materialized_buffers():
    module = torch.nn.Linear(3, 2, device="meta")
    module.register_buffer("frequency", torch.tensor([0.10001]))
    original = module.frequency
    cast_module_parameters(module, torch.bfloat16)
    assert module.weight.is_meta and module.weight.dtype == torch.bfloat16
    assert module.frequency is original and module.frequency.dtype == torch.float32


def test_qwen_rope_is_preserved_but_naive_whole_module_cast_loses_values():
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

    config = Qwen2Config(hidden_size=64, num_attention_heads=4)
    module = Qwen2RotaryEmbedding(config)
    expected = module.inv_freq.clone()
    cast_module_parameters(module, torch.bfloat16)
    assert torch.equal(module.inv_freq, expected)
    assert torch.equal(module.original_inv_freq, expected)
    module.bfloat16().float()
    assert not torch.equal(module.inv_freq, expected)
