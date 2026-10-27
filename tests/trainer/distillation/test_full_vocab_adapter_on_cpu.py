# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Adapter gradients, including the shared embedding/output parameter."""

import copy

import pytest
import torch
from transformers import AutoModelForCausalLM, Qwen2Config, Qwen3Config, Qwen3MoeConfig

from verl.models.transformers.full_vocab import install_full_vocab_adapter
from verl.trainer.distillation.full_vocab import full_reverse_kl
from verl.trainer.distillation.hidden import FrozenTeacherHead


@pytest.mark.parametrize(
    "config_cls,tied", [(Qwen2Config, False), (Qwen3Config, False), (Qwen3Config, True), (Qwen3MoeConfig, False)]
)
def test_adapter_matches_dense_model_value_and_all_gradients(config_cls, tied, tmp_path):
    torch.manual_seed(7)
    config = config_cls(
        vocab_size=19,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        tie_word_embeddings=tied,
        attention_dropout=0.0,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
    )
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    reference = AutoModelForCausalLM.from_config(config).double().eval()
    adapted = copy.deepcopy(reference)
    adapted.save_pretrained(tmp_path)
    head = FrozenTeacherHead(str(tmp_path))
    torch.testing.assert_close(head.cpu_weight, adapted.lm_head.weight.detach())
    install_full_vocab_adapter(adapted)
    ids = torch.tensor([[1, 2, 3, 4]])
    ht, wt = torch.randn(4, 8, dtype=torch.float64), torch.randn(19, 8, dtype=torch.float64)
    lp = reference(ids).logits.squeeze(0).log_softmax(-1)
    lq = (ht @ wt.T).log_softmax(-1)
    expected = (lp.exp() * (lp - lq)).sum()

    def output(hidden, lm_head):
        return lm_head(hidden.squeeze(0), full_vocab_callback=lambda h, w: full_reverse_kl(h, w, ht, wt))

    actual = adapted(ids, full_vocab_callback=output)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    expected.backward()
    actual.backward()
    for (name, param), (other_name, other_param) in zip(
        reference.named_parameters(), adapted.named_parameters(), strict=True
    ):
        assert name == other_name
        torch.testing.assert_close(param.grad, other_param.grad, atol=1e-9, rtol=1e-9)
    if tied:
        assert adapted.lm_head.weight is adapted.model.embed_tokens.weight
    torch.testing.assert_close(adapted(ids).logits, reference(ids).logits)
