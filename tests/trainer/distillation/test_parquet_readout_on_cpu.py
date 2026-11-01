# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Run the real callback through native-shaped TQ and worker boundaries on CPU."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf
from transfer_queue import KVBatchMeta

from verl.trainer.evaluation import source_history as readout
from verl.trainer.evaluation.artifacts import ArtifactStore
from verl.trainer.ppo.readout_callback import build_readout_callback


class BatchDict(dict):
    def __len__(self):
        return len(next(iter(self.values())))


class MemoryQueue:
    def __init__(self):
        self.rows = {}

    def put(self, keys, partition_id, fields=None, **kwargs):
        if fields is None:
            return
        for i, key in enumerate(keys):
            self.rows.setdefault((partition_id, key), {}).update({name: values[i] for name, values in fields.items()})

    def get(self, keys, partition_id, select_fields):
        result = BatchDict()
        for field in select_fields:
            values = [self.rows[(partition_id, key)][field] for key in keys]
            if isinstance(values[0], torch.Tensor):
                result[field] = torch.nested.as_nested_tensor(values, layout=torch.jagged)
            else:
                result[field] = values
        return result

    def clear(self, keys, partition_id):
        for key in keys:
            self.rows.pop((partition_id, key), None)

    def rollout(self, outer, sources, partition):
        keys = [f"{partition}-{i}" for i in sources]
        for key, source in zip(keys, sources, strict=True):
            self.rows[(partition, key)] = dict(
                prompts=torch.tensor([11, source]),
                responses=torch.tensor([source + 1, 21, 22]),
                response_mask=torch.tensor([1, 1, int(source % 2 == 0)]),
                extra_info={"index": source},
                metrics={"finish_reason": "stop"},
            )
        return KVBatchMeta(
            keys=keys,
            partition_id=partition,
            fields=None,
            tags=[dict(min_global_steps=max(0, outer - 1), max_global_steps=max(0, outer - 1)) for _ in sources],
        )


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    source = tmp_path / "source.parquet"
    source.write_bytes(b"fixed-source-identity")  # The fake native dataset below owns decoding.
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"fixture-weights")
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "hidden_size": 4,
                "architectures": ["Qwen3ForCausalLM"],
                "eos_token_id": 1,
                "pad_token_id": 1,
            }
        )
    )
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"word": 0, "eos": 1})), eos_token="eos", pad_token="eos"
    )
    tokenizer.chat_template = "fixed"
    tokenizer.save_pretrained(model)
    config = OmegaConf.create(
        {
            "trainer": {
                "v1": {"trainer_mode": "sync"},
                "nnodes": 1,
                "resume_mode": "disable",
                "val_before_train": True,
                "evaluation": {
                    "_target_": "verl.trainer.evaluation.source_history.SourceHistoryReadout",
                    "output": str(tmp_path / "readout"),
                    "source_file": str(source),
                    "source_count": 2,
                    "history_size": 2,
                    "source_every": 1,
                    "history_every": 1,
                    "score": {"audit_state": True},
                },
            },
            "distillation": {
                "execution_mode": "shared_sync",
                "distillation_loss": {
                    "full_vocab": {"distribution": {}, "input_id_mapping": "identity", "termination_token_ids": []}
                },
                "teacher_models": {"teacher_model": {"model_path": str(model)}},
            },
            "data": {"val_files": [str(source)], "max_response_length": 8},
            "actor_rollout_ref": {
                "hybrid_engine": True,
                "actor": {"strategy": "fsdp2"},
                "model": {"path": str(model)},
                "rollout": {
                    "response_length": 8,
                    "val_kwargs": {"n": 1, "max_tokens": 8},
                    "name": "vllm",
                    "dtype": "bfloat16",
                    "engine_kwargs": {"vllm": {}},
                    "agent": {"default_agent_loop": "single_turn_agent"},
                },
            },
        }
    )
    from verl.experimental.rollout_policy.validation import policy_engine_config

    config.actor_rollout_ref.rollout = policy_engine_config(config.actor_rollout_ref.rollout)
    queue = MemoryQueue()
    monkeypatch.setattr(readout.tq, "kv_batch_get", queue.get)
    monkeypatch.setattr(readout.tq, "kv_batch_put", queue.put)
    monkeypatch.setattr(readout.tq, "kv_clear", queue.clear)
    bank = queue.rollout(0, [0, 1], "val")
    callback = build_readout_callback(config)
    trainer = SimpleNamespace(
        config=config,
        actor_optimizer_step=0,
        global_steps=0,
        total_training_steps=2,
        val_dataset=[{"prompt": "question-a"}, {"prompt": "question-b"}],
        checkpoint_manager=SimpleNamespace(sleep_replicas=MagicMock(), update_weights=MagicMock()),
        agent_loop_manager=SimpleNamespace(generate_sequences=MagicMock()),
        replay_buffer=SimpleNamespace(sample=lambda **kwargs: (bank, {})),
        _stamp_rollout_submission=MagicMock(),
        _balance_batch=lambda meta, metrics, **kwargs: meta,
        _compute_shared_teacher_log_probs=MagicMock(),
    )

    def score(meta):
        for key in meta.keys:
            stored = queue.rows[(meta.partition_id, key)]
            count = int(stored["response_mask"].sum())
            risk = float(stored["responses"][0]) / (callback.pending_outer + 1)
            stored.update(rkl_sum=torch.tensor([risk * count]), rkl_count=torch.tensor([count]))

    trainer.actor_rollout_wg = SimpleNamespace(
        begin_rkl_readout=MagicMock(),
        score_full_vocab=score,
        end_rkl_readout=lambda: [{"unchanged": True, "state_digest": f"state-{trainer.actor_optimizer_step}"}],
    )
    from verl.experimental.rollout_policy.service import DefaultRolloutService

    trainer.common_rollout_sampling = True
    trainer.rollout_service = DefaultRolloutService(trainer, None)
    trainer.rollout_service.resident_version = 0
    trainer.rollout_service.configure_sampling = MagicMock()
    trainer._validate = lambda: callback.validate(trainer)
    return callback, trainer, queue


def test_initial_and_two_outer_readouts_publish_parquet_without_token_history(runtime):
    callback, trainer, queue = runtime
    metrics = callback.on_readout(trainer, 0)
    assert metrics["eval/rkl/source_heldout/question_mean"] == 1.5
    assert trainer.checkpoint_manager.update_weights.call_count == 1
    for outer in (1, 2):
        trainer.actor_optimizer_step = outer * 32
        metrics = callback.on_before_weight_sync(
            trainer, queue.rollout(outer, [10 * outer, 10 * outer + 1], "train"), outer
        )
        event = ArtifactStore(callback.root / "scores" / f"outer-{outer:03d}")
        assert event.manifest(complete=True)["metadata"]["optimizer_steps"] == outer * 32
        assert len(event.read_rows("rkl.parquet")) == 4
        assert "eval/rkl/history/equal_batch_question_mean" in metrics
        assert all("response_ids" not in row for rows in callback.batches.values() for row in rows)
        refs = callback.store.read_rows(f"history/outer-{outer:03d}.parquet")
        assert all("response_ids" not in row for row in refs)
        assert len(callback.store.resolve(refs)) == 2
    assert metrics["eval/rkl/source_heldout/reduction_percent"] == pytest.approx(100 * 2 / 3)
    assert callback.scored == {0, 1, 2}
    assert all("rkl-readout" not in partition for partition, _ in queue.rows)
    assert not list((callback.root / "trajectories").glob("*.json"))
    assert not list((callback.root / "history").glob("*.json"))
    # Repeated dispatch does not publish a second event or score again.
    assert callback.validate(trainer) == {}
    assert callback.on_readout(trainer, 2) == {}
    assert callback.store.manifest(complete=True)["metadata"]["completed_outer"] == 2


@pytest.mark.parametrize("failure", ["score", "audit", "restore"])
def test_failure_never_publishes_a_complete_score_event(runtime, failure):
    callback, trainer, queue = runtime
    if failure == "score":
        trainer.actor_rollout_wg.score_full_vocab = MagicMock(side_effect=RuntimeError("score failed"))
    elif failure == "audit":
        trainer.actor_rollout_wg.end_rkl_readout = lambda: [{"unchanged": False, "state_digest": "bad"}]
    else:
        trainer.checkpoint_manager.update_weights.side_effect = RuntimeError("restore failed")
    with pytest.raises(RuntimeError):
        callback.on_readout(trainer, 0)
    assert not (callback.root / "scores/outer-000/manifest.json").exists()
    assert not callback.scored
    if failure == "score":
        # An unresolved RPC may still write remotely: fail-stop retains its partition.
        assert any("rkl-readout" in partition for partition, _ in queue.rows)
        trainer.checkpoint_manager.update_weights.assert_not_called()
    else:
        assert all("rkl-readout" not in partition for partition, _ in queue.rows)


def test_shared_source_bank_checks_template_and_does_not_regenerate(runtime, tmp_path):
    callback, trainer, queue = runtime
    callback.on_readout(trainer, 0)
    config = OmegaConf.create(OmegaConf.to_container(trainer.config))
    config.trainer.evaluation.output = str(tmp_path / "new-run")
    config.trainer.evaluation.initial_bank = str(callback.bank_path)
    reused = build_readout_callback(config)
    assert reused.bank == callback.bank
    import json

    tokenizer_path = tmp_path / "model/tokenizer_config.json"
    tokenizer_config = json.loads(tokenizer_path.read_text())
    tokenizer_config["chat_template"] = "different"
    tokenizer_path.write_text(json.dumps(tokenizer_config))
    config.trainer.evaluation.output = str(tmp_path / "mismatch")
    with pytest.raises(ValueError, match="protocol differs"):
        build_readout_callback(config)


def test_independent_history_schedule_and_unrequested_audit(runtime):
    callback, trainer, queue = runtime
    from dataclasses import replace

    callback.score_budget = replace(callback.score_budget, audit_state=False)
    trainer.actor_rollout_wg.end_rkl_readout = lambda: [{"unchanged": True, "state_digest": None}]
    callback.schedules["source_heldout"] = readout.Schedule(at=(0,), every=2)
    callback.on_readout(trainer, 0)
    trainer.actor_optimizer_step = 32
    metrics = callback.on_before_weight_sync(trainer, queue.rollout(1, [10, 11], "train"), 1)
    assert "eval/rkl/source_heldout/question_mean" not in metrics
    assert "eval/rkl/history/equal_batch_question_mean" in metrics
    event = ArtifactStore(callback.root / "scores/outer-001")
    assert event.manifest(complete=True)["metadata"]["audit"]["status"] == "not_requested"


def test_manual_validation_does_not_bypass_task_frequency(runtime):
    callback, trainer, queue = runtime
    callback.schedules = {"source_heldout": readout.Schedule(at=(0,), every=2), "history": readout.Schedule(every=2)}
    callback.on_readout(trainer, 0)
    trainer.actor_optimizer_step = 32
    assert callback.on_before_weight_sync(trainer, queue.rollout(1, [10, 11], "train"), 1) == {}
    sleeps = trainer.checkpoint_manager.sleep_replicas.call_count
    assert trainer._validate() == {}
    assert trainer.checkpoint_manager.sleep_replicas.call_count == sleeps
    assert not (callback.root / "scores/outer-001").exists()


def test_acceptance_auditor_reads_new_artifacts_without_changing_legacy_mean(runtime, tmp_path):
    from examples.mirroropd.benchmarks.profile.audit_fixed_rkl import audit, mean_rkl

    callback, trainer, queue = runtime
    callback.on_readout(trainer, 0)
    for outer in (1, 2):
        trainer.actor_optimizer_step = outer * 32
        callback.on_before_weight_sync(trainer, queue.rollout(outer, [10 * outer, 10 * outer + 1], "train"), outer)
    trainer.config.trainer.total_training_steps = 2
    trainer.config.data.train_batch_size = 2
    trainer.config.actor_rollout_ref.actor = {"ppo_epochs": 32}
    (tmp_path / "resolved_config.yaml").write_text(OmegaConf.to_yaml(trainer.config))
    updates = [
        dict(
            data={
                "step/optimizer": i,
                "optim/actor/update_applied": 1,
                "optim/actor/loss": 0.5,
                "optim/actor/grad_norm_preclip": 1.0,
            }
        )
        for i in range(1, 65)
    ]
    updates.extend([dict(data={"perf/timing/outer_step_seconds": 1.0}) for _ in range(2)])
    (tmp_path / "metrics.jsonl").write_text("\n".join(json.dumps(x) for x in updates))
    summary = audit(tmp_path)
    assert summary["aggregation"] == "question_mean"
    assert summary["source_rkl_reduction_percent"] == pytest.approx(100 * 2 / 3)
    assert summary["unique_training_questions"] == 4
    old = [dict(rkl_sum=1, tokens=1), dict(rkl_sum=300, tokens=100)]
    assert mean_rkl(old) == 301 / 101
    assert mean_rkl(old, question_mean=True) == 2


def test_source_bank_reuse_rejects_changed_weights(runtime, tmp_path):
    callback, trainer, _ = runtime
    callback.on_readout(trainer, 0)
    config = OmegaConf.create(OmegaConf.to_container(trainer.config))
    config.trainer.evaluation.output = str(tmp_path / "changed-weights")
    config.trainer.evaluation.initial_bank = str(callback.bank_path)
    (tmp_path / "model/model.safetensors").write_bytes(b"different-weights")
    with pytest.raises(ValueError, match="model, input"):
        build_readout_callback(config)
    assert not (tmp_path / "changed-weights").exists()


def test_source_bank_identity_is_portable_across_model_locations(runtime, tmp_path):
    import shutil

    callback, trainer, _ = runtime
    callback.on_readout(trainer, 0)
    relocated = tmp_path / "relocated-model"
    shutil.copytree(tmp_path / "model", relocated)
    config = OmegaConf.create(OmegaConf.to_container(trainer.config))
    config.actor_rollout_ref.model.path = str(relocated)
    config.trainer.evaluation.output = str(tmp_path / "relocated-run")
    config.trainer.evaluation.initial_bank = str(callback.bank_path)
    reused = build_readout_callback(config)
    assert reused.bank == callback.bank


def test_source_bank_reuse_rejects_changed_effective_stopping_contract(runtime, tmp_path):
    callback, trainer, _ = runtime
    callback.on_readout(trainer, 0)
    config = OmegaConf.create(OmegaConf.to_container(trainer.config))
    config.trainer.evaluation.output = str(tmp_path / "changed-stop")
    config.trainer.evaluation.initial_bank = str(callback.bank_path)
    config.actor_rollout_ref.rollout.stop_token_ids = [0]  # A new stop, distinct from native EOS=1.
    with pytest.raises(ValueError, match="model, input"):
        build_readout_callback(config)
    assert not (tmp_path / "changed-stop").exists()


def test_source_bank_rejects_noncurrent_training_generator_before_creating_output(runtime, tmp_path):
    _, trainer, _ = runtime
    config = OmegaConf.create(OmegaConf.to_container(trainer.config))
    config.trainer.evaluation.output = str(tmp_path / "wrong-generator")
    config.rollout_policy = {"mode": "single", "model": {"kind": "teacher"}}
    with pytest.raises(ValueError, match="current single-model"):
        build_readout_callback(config)
    assert not (tmp_path / "wrong-generator").exists()


def test_source_readout_restores_the_sampler_through_the_rollout_owner(runtime):
    callback, trainer, _ = runtime
    callback.on_readout(trainer, 0)
    trainer.rollout_service.configure_sampling.assert_called_once()
    assert trainer.rollout_service.resident_version == 0


def test_source_readout_rejects_a_second_scheduler(runtime, tmp_path):
    _, trainer, _ = runtime
    config = OmegaConf.create(OmegaConf.to_container(trainer.config))
    config.trainer.test_freq = 10
    config.trainer.evaluation.output = str(tmp_path / "duplicate-scheduler")
    with pytest.raises(ValueError, match="owns evaluation schedules"):
        build_readout_callback(config)
    assert not (tmp_path / "duplicate-scheduler").exists()
