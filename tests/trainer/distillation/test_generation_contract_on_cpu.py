# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Catch bootstrap/current sampling-contract drift before allocating an engine."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl.experimental.rollout_policy.validation import policy_engine_config
from verl.trainer.evaluation.artifacts import identity
from verl.trainer.evaluation.generation_contract import (
    SUPPORT_PROCESSOR,
    engine_contract,
    input_identity,
    sample_identity,
)


class Tokenizer:
    chat_template = "template"
    bos_token_id, eos_token_id, pad_token_id = 0, 1, 1

    def __init__(self, normalizer="identity"):
        self.backend_tokenizer = SimpleNamespace(
            to_str=lambda: json.dumps({"normalizer": normalizer, "model": {"vocab": self.get_vocab()}})
        )

    def get_vocab(self):
        return {"a": 0, "b": 1}


def model(tmp_path):
    return SimpleNamespace(
        path=str(tmp_path),
        tokenizer=Tokenizer(),
        hf_config=SimpleNamespace(vocab_size=3, eos_token_id=1, model_type="qwen2"),
    )


def rollout():
    return OmegaConf.create(
        dict(
            name="vllm",
            dtype="bfloat16",
            engine_kwargs={"vllm": {}},
            logprobs_mode="raw_logprobs",
            stop_token_ids=[],
            ignore_eos=False,
        )
    )


def test_engine_support_is_required_and_preparation_is_idempotent(tmp_path):
    raw, m = rollout(), model(tmp_path)
    with pytest.raises(ValueError, match="before server creation"):
        engine_contract(raw, m, m)
    prepared = policy_engine_config(raw)
    twice = policy_engine_config(prepared)
    assert prepared.engine_kwargs.vllm.logits_processors == [SUPPORT_PROCESSOR]
    assert "logits_processors" not in raw.engine_kwargs.vllm
    assert engine_contract(prepared, m, m) == engine_contract(twice, m, m)
    first = identity(engine_contract(prepared, m, m))
    prepared.ignore_eos = True
    assert identity(engine_contract(prepared, m, m)) != first


def test_input_definition_and_semantic_sample_identity_are_bound(tmp_path):
    m = model(tmp_path)
    before = input_identity(m)
    m.tokenizer = Tokenizer(normalizer="different")
    assert input_identity(m) != before  # Same token->ID map is insufficient.
    assert sample_identity("m", 0, "d", "q", 0, "p", "masked") != sample_identity("m", 0, "d", "q", 0, "p", "raw")


def test_bootstrap_passes_prepared_config_to_actual_replica_constructor(tmp_path, monkeypatch):
    from verl.trainer.evaluation import bootstrap

    cfg = OmegaConf.create(
        {
            "trainer": {
                "n_gpus_per_node": 4,
                "evaluation": {
                    "output": str(tmp_path / "out"),
                    "bootstrap": str(tmp_path / "bank"),
                    "source_file": "source",
                    "datasets": [],
                    "protocols": {"source": {"max_new_tokens": 7168}, "public": {}},
                },
            },
            "actor_rollout_ref": {"model": {"path": str(tmp_path)}, "rollout": rollout()},
            "distillation": {"teacher_models": {"teacher_model": {"model_path": str(tmp_path)}}},
        }
    )
    monkeypatch.setattr(bootstrap, "omega_conf_to_dataclass", lambda *a, **k: model(tmp_path))
    monkeypatch.setattr(bootstrap, "model_identity", lambda path: "model")

    class ConstructorReached(Exception):
        pass

    def replica(rank, resolved, teacher, **kwargs):
        assert resolved.engine_kwargs.vllm.logits_processors == [SUPPORT_PROCESSOR]
        assert resolved.logprobs_mode == "processed_logprobs"
        assert resolved.max_model_len == 40960
        raise ConstructorReached

    monkeypatch.setattr(bootstrap, "get_rollout_replica_class", lambda name: replica)
    with pytest.raises(ConstructorReached):
        asyncio.run(bootstrap.generate_bootstrap(cfg, "initial"))
