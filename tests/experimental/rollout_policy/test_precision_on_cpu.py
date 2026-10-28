# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
import copy

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from verl.experimental.rollout_policy.precision import configure_precision


@pytest.mark.parametrize("mode", ["head", "residual", "linear", "intermediate", "stable"])
def test_precision_preserves_parameter_storage_names_and_ties(mode):
    model = (
        Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=17,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                tie_word_embeddings=True,
            )
        )
        .eval()
        .bfloat16()
    )
    before = {name: (value.data_ptr(), value.clone()) for name, value in model.state_dict().items()}
    configure_precision(model, mode)
    for name, value in model.state_dict().items():
        assert value.data_ptr() == before[name][0]
        torch.testing.assert_close(value, before[name][1], rtol=0, atol=0)
    with torch.inference_mode():
        result = model(torch.tensor([[1, 2, 3], [3, 4, 5]]))
        assert result.logits.dtype == torch.float32
        assert torch.isfinite(result.logits).all()
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    with pytest.raises(ValueError, match="already"):
        configure_precision(model, mode)


def test_fixed_moe_mode_does_not_change_dense_model():
    model = (
        Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=17,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
            )
        )
        .eval()
        .bfloat16()
    )
    other = copy.deepcopy(model)
    configure_precision(model, "stable")
    configure_precision(other, "stable_moe")
    with torch.inference_mode():
        ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        torch.testing.assert_close(model(ids).logits, other(ids).logits, rtol=0, atol=0)
