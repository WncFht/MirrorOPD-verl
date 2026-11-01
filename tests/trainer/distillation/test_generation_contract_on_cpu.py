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
            agent={"default_agent_loop": "single_turn_agent"},
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
                "nnodes": 1,
                "v1": {"trainer_mode": "sync"},
                "evaluation": {
                    "_target_": "verl.trainer.evaluation.coordinator.EvaluationCoordinator",
                    "output": str(tmp_path / "out"),
                    "bootstrap": str(tmp_path / "bank"),
                    "source_file": "source",
                    "datasets": [],
                    "protocols": {"source": {"max_new_tokens": 7168}, "public": {}},
                },
            },
            "actor_rollout_ref": {
                "model": {"path": str(tmp_path)},
                "rollout": rollout(),
                "hybrid_engine": True,
                "actor": {"strategy": "fsdp2"},
            },
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


def test_independent_current_evaluation_does_not_change_training_policy(tmp_path, monkeypatch):
    from verl.experimental.rollout_policy import validation
    from verl.trainer.evaluation.generation_contract import require_current_generation

    monkeypatch.setattr(validation, "uses_common_single_sampling", lambda config: True)
    config = OmegaConf.create({"rollout_policy": {"mode": "single", "model": {"kind": "teacher"}}})
    before = OmegaConf.to_container(config)
    require_current_generation(config, independent_engine=True)
    assert OmegaConf.to_container(config) == before
    with pytest.raises(ValueError, match="current single-model"):
        require_current_generation(config)
