# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Equivalent-token relabeling across model input, teacher generation and head scoring."""

import asyncio
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoModelForCausalLM, LlamaConfig, PreTrainedTokenizerFast

from verl.trainer.distillation.compatibility import ModelPairContract, preflight_pair
from verl.trainer.distillation.token_mapping import TokenIdMapping


def tokenizer(permuted=False):
    vocab = {"[UNK]": 2, "a": 0, "b": 1} if permuted else {"[UNK]": 0, "a": 1, "b": 2}
    return PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="[UNK]")), unk_token="[UNK]")


def test_input_and_output_mapping_are_explicit_and_bijective():
    a, b = tokenizer(), tokenizer(True)
    mapping = TokenIdMapping.build(a, b)
    assert mapping.tokens([0, 1, 2]) == [2, 0, 1]
    assert mapping.tokens([2, 0, 1], reverse=True) == [0, 1, 2]
    torch.testing.assert_close(mapping.tensor(torch.tensor([[1, 2, 0]])), torch.tensor([[0, 1, 2]]))
    config = LlamaConfig(vocab_size=3, max_position_embeddings=64)
    support = dict(mode="common_conditioned", student_ids=[0, 1, 2], teacher_ids=[2, 0, 1])
    with pytest.raises(ValueError, match="explicitly enable"):
        ModelPairContract.build(config, config, a, b, distribution=support)
    pair = ModelPairContract.build(config, config, a, b, distribution=support, allow_permutation=True)
    assert pair.input_mapping_sha256 == mapping.mapping_id
    with pytest.raises(ValueError, match="token meanings"):
        ModelPairContract.build(
            config,
            config,
            a,
            b,
            distribution=dict(mode="common_conditioned", common_vocab_size=3),
            allow_permutation=True,
        )


def test_generation_adapter_maps_tokens_and_stops_without_retokenizing(monkeypatch):
    from verl.trainer.evaluation.generation import CheckedClient, LocalPhaseClient
    from verl.workers.rollout.replica import TokenOutput

    async def generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):
        assert prompt_ids == [0, 1] and sampling_params["stop_token_ids"] == [1]
        return TokenOutput(token_ids=[1, 0], log_probs=[-0.2, -0.3])

    monkeypatch.setattr(LocalPhaseClient, "generate", generate)
    client = object.__new__(CheckedClient)
    client.max_model_len = 32
    client.prompt_limit = 16
    client.token_mapping = TokenIdMapping.build(tokenizer(), tokenizer(True))
    sampling = {"max_tokens": 4, "stop_token_ids": [2]}
    output = asyncio.run(client.generate("test", prompt_ids=[1, 2], sampling_params=sampling))
    assert output.token_ids == [2, 1] and output.log_probs == [-0.2, -0.3]
    assert sampling["stop_token_ids"] == [2]


def test_real_checkpoint_preflight_accepts_nullable_stop_ids(tmp_path):
    for role, width in [("student", 7), ("teacher", 5)]:
        config = LlamaConfig(
            vocab_size=width,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=128,
        )
        AutoModelForCausalLM.from_config(config).save_pretrained(tmp_path / role)
        tokenizer().save_pretrained(tmp_path / role)
    cfg = OmegaConf.create(
        {
            "distillation": {
                "enabled": True,
                "teacher_models": {"teacher_model": {"model_path": str(tmp_path / "teacher")}},
                "distillation_loss": {
                    "loss_mode": "mirror",
                    "mirror": {"objective": "mdpo", "estimator": "full_vocab", "eta": 1},
                    "full_vocab": {"distribution": {"mode": "common_conditioned", "common_vocab_size": 5}},
                },
            },
            "actor_rollout_ref": {"model": {"path": str(tmp_path / "student")}, "rollout": {"stop_token_ids": None}},
        }
    )
    assert preflight_pair(cfg).student_width == 7


def test_dynamic_date_must_be_explicit_and_consistent(monkeypatch):
    import verl.trainer.distillation.compatibility as compatibility

    monkeypatch.setattr(
        compatibility,
        "inspect_pair",
        lambda *a, **kw: SimpleNamespace(dynamic_date=True, student_context=131072, teacher_context=131072),
    )
    cfg = OmegaConf.create(
        {
            "distillation": {
                "enabled": True,
                "teacher_models": {"t": {"model_path": "t"}},
                "distillation_loss": {
                    "loss_mode": "mirror",
                    "mirror": {"objective": "mdpo", "estimator": "full_vocab", "eta": 1},
                    "full_vocab": {},
                },
            },
            "actor_rollout_ref": {"model": {"path": "s"}, "rollout": {}},
            "data": {"apply_chat_template_kwargs": {}},
            "trainer": {"evaluation": {"protocols": {"public": {"template_json": '{"date_string":"01 Nov 2026"}'}}}},
        }
    )
    with pytest.raises(ValueError, match="date_string"):
        preflight_pair(cfg)
    cfg.data.apply_chat_template_kwargs.date_string = "02 Nov 2026"
    with pytest.raises(ValueError, match="same date"):
        preflight_pair(cfg)
    cfg.data.apply_chat_template_kwargs.date_string = "01 Nov 2026"
    assert preflight_pair(cfg).dynamic_date


def test_global_engine_stops_are_mapped_before_server_creation():
    from verl.trainer.evaluation.generation_contract import prepared_generation_config

    cfg = OmegaConf.create(
        {"name": "vllm", "engine_kwargs": {"vllm": {}}, "logprobs_mode": "raw_logprobs", "stop_token_ids": [2]}
    )
    source, target = SimpleNamespace(tokenizer=tokenizer()), SimpleNamespace(tokenizer=tokenizer(True))
    prepared = prepared_generation_config(cfg, target, source)
    assert prepared.stop_token_ids == [1]
    assert cfg.stop_token_ids == [2]
