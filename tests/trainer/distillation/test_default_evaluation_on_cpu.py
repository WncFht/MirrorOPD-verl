# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Native default validation preserves session reduction, rewards and both dumps."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl.trainer.evaluation import default as module
from verl.trainer.ppo.v1.trainer_base import PPOTrainer


@pytest.mark.parametrize("colocated_reward", [False, True])
def test_default_protocol_keeps_final_sessions_metrics_and_replay_records(monkeypatch, tmp_path, colocated_reward):
    events = []
    submitted = {}
    monkeypatch.setattr(module.tq, "kv_batch_put", lambda **k: submitted.update(k))
    monkeypatch.setattr(module.tq, "kv_clear", lambda **k: events.append("clear"))

    def sample(**kwargs):
        uid = submitted["keys"][0]
        return SimpleNamespace(keys=[f"{uid}_0_0", f"{uid}_0_1", f"{uid}_1_0"], partition_id="val"), {}

    def get(*, keys, select_fields, **kwargs):
        if select_fields == ["prompts", "responses"]:
            return {
                "prompts": torch.nested.nested_tensor([torch.tensor([1])] * 3, layout=torch.jagged),
                "responses": torch.nested.nested_tensor(
                    [torch.tensor([2]), torch.tensor([3, 4]), torch.tensor([5])], layout=torch.jagged
                ),
            }
        uid = submitted["keys"][0]
        return dict(
            uid=np.array([uid, uid]),
            rm_scores=torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
            num_turns=np.array([2, 1]),
            reward_model=np.array([{"ground_truth": "answer"}] * 2),
            data_source=np.array(["MATH-500"] * 2),
            extra_fields=np.array(
                [
                    {"reward_extra_info": {"acc": 1.0}, "finish_reason": "stop", "tool_call_counts": 1},
                    {"reward_extra_info": {"acc": 0.0}, "finish_reason": "length", "tool_call_counts": 0},
                ]
            ),
        )

    monkeypatch.setattr(module.tq, "kv_batch_get", get)
    trainer = SimpleNamespace(
        readout_callback=None,
        global_steps=3,
        config=OmegaConf.create(
            {
                "trainer": {"validation_data_dir": str(tmp_path), "eval_metrics_resamples": 8},
                "data": {"seed": 42},
                "actor_rollout_ref": {"rollout": {"val_kwargs": {"n": 2}}},
            }
        ),
        val_dataloader=[
            {
                "raw_prompt": np.array(["question"]),
                "data_source": np.array(["MATH-500"]),
                "reward_model": np.array([{"ground_truth": "answer", "metric_fields": ["acc"]}]),
            }
        ],
        tokenizer=SimpleNamespace(pad_token_id=0, decode=lambda ids, **k: ",".join(str(i) for i in ids.tolist() if i)),
        reward_loop_manager=SimpleNamespace(reward_loop_worker_handles=None if colocated_reward else [object()]),
        checkpoint_manager=SimpleNamespace(
            sleep_replicas=lambda: events.append("sleep"), update_weights=lambda: events.append("restore")
        ),
        agent_loop_manager=SimpleNamespace(generate_sequences=lambda batch: events.append("generate")),
        replay_buffer=SimpleNamespace(sample=sample),
        _stamp_rollout_submission=Mock(),
        _maybe_log_val_generations=Mock(),
        _dump_generations=Mock(),
        _collect_pipeline_metrics=lambda batch: {"timing_s/rollout_batch": 2.0, "rollout/preemptions": 0},
    )
    trainer._val_metrics_update = lambda *a, **k: PPOTrainer._val_metrics_update(trainer, *a, **k)
    trainer._compute_reward_colocate = lambda batch: (events.append("reward"), batch)[1]
    metrics = PPOTrainer._validate(trainer)
    assert metrics["eval/math500/acc/mean"] == 0.5
    assert metrics["validation/timing_s/rollout_batch"] == 2.0
    assert events == (
        ["generate", "sleep", "reward", "restore", "clear"] if colocated_reward else ["generate", "clear"]
    )
    logged = trainer._maybe_log_val_generations.call_args.kwargs
    assert logged["outputs"] == ["3,4", "5"] and logged["scores"] == [1.0, 0.0]
    all_rows, records = [call.kwargs for call in trainer._dump_generations.call_args_list]
    assert all_rows["outputs"] == ["2", "3,4", "5"]
    assert records["outputs"] == ["3,4", "5"]
    assert records["reward_extra_infos_dict"]["status"] == ["ok", "ok"]
    assert records["reward_extra_infos_dict"]["sample_index"] == [0, 1]
    assert records["reward_extra_infos_dict"]["response_length"] == [2, 1]


def test_validation_owner_dispatch_does_not_require_a_dataset_for_custom_protocol():
    callback = SimpleNamespace(handles_validation=True, validate=Mock(return_value={"custom": 1.0}))
    trainer = SimpleNamespace(readout_callback=callback)
    assert PPOTrainer._validate(trainer) == {"custom": 1.0}
    callback.validate.assert_called_once_with(trainer)


def test_non_owning_readout_keeps_default_dataset_requirement():
    trainer = SimpleNamespace(readout_callback=SimpleNamespace(handles_validation=False), val_dataloader=None)
    with pytest.raises(ValueError, match="nonempty data.val_files"):
        PPOTrainer._validate(trainer)
