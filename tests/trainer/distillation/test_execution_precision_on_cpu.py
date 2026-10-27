# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
import copy

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen3Config, Qwen3ForCausalLM

from verl.models.transformers.precision import install_precision_adapter, packed_fp32_attention, register_fp32_attention
from verl.workers.config import FSDPEngineConfig


@pytest.mark.parametrize(
    "profile,dtype", [("bf16", "bfloat16"), ("fp16", "float16"), ("fp32", "float32"), ("bf16_fp32", "bfloat16")]
)
def test_profile_resolves_once_and_rejects_conflicting_cast(profile, dtype):
    config = FSDPEngineConfig(strategy="fsdp2", forward_precision=profile)
    assert config.mixed_precision == {"param_dtype": dtype, "reduce_dtype": "float32"}
    with pytest.raises(ValueError, match="conflicts"):
        FSDPEngineConfig(
            strategy="fsdp2",
            forward_precision=profile,
            mixed_precision={"param_dtype": "float32" if dtype != "float32" else "bfloat16"},
        )


@pytest.mark.parametrize("config_type,model_type", [(Qwen2Config, Qwen2ForCausalLM), (Qwen3Config, Qwen3ForCausalLM)])
def test_instance_adapter_keeps_teacher_parameters_and_names(config_type, model_type):
    cfg = config_type(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        tie_word_embeddings=True,
    )
    model = model_type(cfg).bfloat16()
    teacher = copy.deepcopy(model)
    names = dict(model.named_parameters(remove_duplicate=False))
    install_precision_adapter(model, "bf16_fp32")
    assert all(dict(model.named_parameters(remove_duplicate=False))[k] is v for k, v in names.items())
    assert model.lm_head.weight is model.model.embed_tokens.weight
    assert model.model.embed_tokens(torch.tensor([[1, 2]])).dtype == torch.float32
    assert teacher.model.embed_tokens(torch.tensor([[1, 2]])).dtype == torch.bfloat16
    linear = model.model.layers[0].mlp.gate_proj
    assert linear(torch.ones(1, 2, 32)).dtype == torch.float32
    assert model.lm_head(torch.ones(1, 2, 32)).dtype == torch.float32
    torch.testing.assert_close(model.state_dict(), teacher.state_dict(), atol=0, rtol=0)


def test_packed_fp32_matches_independent_sequences_and_gradients():
    torch.manual_seed(4)
    q = torch.randn(1, 4, 7, 8, requires_grad=True)
    k = torch.randn(1, 2, 7, 8, requires_grad=True)
    v = torch.randn(1, 2, 7, 8, requires_grad=True)
    pos = torch.tensor([[0, 1, 2, 0, 1, 2, 3]])
    out, _ = packed_fp32_attention(None, q, k, v, None, position_ids=pos)
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel(SDPBackend.MATH):
        expected = torch.cat(
            [
                torch.nn.functional.scaled_dot_product_attention(
                    q[:, :, a:b], k[:, :, a:b], v[:, :, a:b], is_causal=True, enable_gqa=True
                )
                for a, b in [(0, 3), (3, 7)]
            ],
            dim=2,
        ).transpose(1, 2)
    torch.testing.assert_close(out, expected, atol=3e-7, rtol=2e-5)
    grad = torch.randn_like(out)
    actual = torch.autograd.grad(out, (q, k, v), grad, retain_graph=True)
    expected_grad = torch.autograd.grad(expected, (q, k, v), grad)
    for a, b in zip(actual, expected_grad, strict=True):
        torch.testing.assert_close(a, b, atol=3e-7, rtol=2e-5)


def test_registered_reference_preserves_packed_model_boundaries():
    register_fp32_attention()
    cfg = Qwen2Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    cfg._attn_implementation = "verl_fp32"
    model = Qwen2ForCausalLM(cfg).eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    pos = torch.tensor([[0, 1, 2, 0, 1, 2, 3]])
    packed = model(ids, position_ids=pos).logits
    separate = torch.cat([model(ids[:, a:b], position_ids=pos[:, a:b]).logits for a, b in [(0, 3), (3, 7)]], dim=1)
    torch.testing.assert_close(packed, separate, rtol=2e-5, atol=1e-6)


@pytest.mark.parametrize(
    "change,message", [({"use_liger": True}, "native modules"), ({"lora_rank": 4}, "native modules")]
)
def test_profiles_reject_unvalidated_module_combinations(change, message):
    from types import SimpleNamespace

    from verl.models.transformers.precision import prepare_precision_config

    model = SimpleNamespace(
        hf_config=Qwen2Config(), use_liger=False, use_fused_kernels=False, lora_rank=0, use_remove_padding=True
    )
    for key, value in change.items():
        setattr(model, key, value)
    with pytest.raises(ValueError, match=message):
        prepare_precision_config(model, FSDPEngineConfig(strategy="fsdp2", forward_precision="bf16_fp32"))


def test_mixed_profile_rejects_attention_that_changes_its_compute_boundary():
    from types import SimpleNamespace

    from verl.models.transformers.precision import prepare_precision_config

    config = Qwen2Config()
    config._attn_implementation = "sdpa"
    model = SimpleNamespace(
        hf_config=config, use_liger=False, use_fused_kernels=False, lora_rank=0, use_remove_padding=True
    )
    with pytest.raises(ValueError, match="FlashAttention2"):
        prepare_precision_config(model, FSDPEngineConfig(strategy="fsdp2", forward_precision="bf16_fp32"))
