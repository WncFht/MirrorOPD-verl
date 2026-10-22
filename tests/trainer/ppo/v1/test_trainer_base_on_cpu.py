# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer, ReplayBufferAsync
from verl.trainer.ppo.v1.trainer_base import PPOTrainer, apply_greedy_sampling_params
from verl.utils import tensordict_utils as tu


def test_greedy_reward_baseline_does_not_request_score_centering_heads():
    params = {"top_p": 0.9, "top_k": 64, "temperature": 1.0, "topk_log_probs": 128, "logprobs": True}
    apply_greedy_sampling_params(params)
    assert params == {"top_p": 1.0, "top_k": -1, "temperature": 0, "logprobs": True}


class _StubTrainer(PPOTrainer):
    def on_step_end(self):
        pass

    def on_sample_end(self):
        pass


class _CustomSampler:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _trainer_with_filter_groups(filter_groups: dict, trainer_mode: str = "sync") -> _StubTrainer:
    trainer = _StubTrainer.__new__(_StubTrainer)
    trainer.trainer_mode = trainer_mode
    trainer.config = OmegaConf.create(
        {
            "algorithm": {"filter_groups": filter_groups},
            "data": {"train_batch_size": 64, "gen_batch_size": 8},
            "reward": {"reward_model": {"enable": False, "enable_resource_pool": False}},
            "trainer": {
                "v1": {
                    trainer_mode: {},
                    "sampler": {
                        "custom_sampler": None,
                        "max_off_policy_threshold": 1,
                        "max_off_policy_strategy": "drop",
                        "sampler_kwargs": {},
                    },
                }
            },
        }
    )
    return trainer


def test_builtin_sampler_class_follows_trainer_mode():
    sync_sampler = _trainer_with_filter_groups({"enable": False}, trainer_mode="sync")._build_replay_buffer()
    async_samplers = [
        _trainer_with_filter_groups({"enable": True, "metric": "acc"}, trainer_mode=mode)._build_replay_buffer()
        for mode in ("colocate_async", "separate_async")
    ]

    assert type(sync_sampler) is ReplayBuffer
    assert all(type(sampler) is ReplayBufferAsync for sampler in async_samplers)
    assert all(sampler.filter_groups_metric == "acc" for sampler in async_samplers)
    assert all(sampler.train_batch_size is None for sampler in async_samplers)
    assert all(sampler.gen_batch_size is None for sampler in async_samplers)


def test_custom_sampler_skips_builtin_filter_groups_validation():
    trainer = _trainer_with_filter_groups({"enable": True, "metric": "acc"})
    trainer.config.trainer.v1.sampler.custom_sampler = {"path": "custom.py", "name": "CustomSampler"}

    with (
        patch("verl.trainer.ppo.v1.trainer_base.load_extern_type", return_value=_CustomSampler),
        patch.object(trainer, "_resolve_filter_groups_metric") as resolve_filter_groups_metric,
    ):
        sampler = trainer._build_replay_buffer()

    resolve_filter_groups_metric.assert_not_called()
    assert isinstance(sampler, _CustomSampler)
    assert "filter_groups_metric" not in sampler.kwargs
    assert "train_batch_size" not in sampler.kwargs
    assert "gen_batch_size" not in sampler.kwargs
    assert "max_inflight_gen_batches" not in sampler.kwargs
    assert "sync_refill_failed_groups" not in sampler.kwargs


def test_builtin_filter_groups_uses_default_inflight_limit():
    trainer = _trainer_with_filter_groups({"enable": True, "metric": "acc"})

    sampler = trainer._build_replay_buffer()

    assert sampler.filter_groups_metric == "acc"
    assert sampler.train_batch_size == 64
    assert sampler.gen_batch_size == 1
    assert sampler.max_inflight_gen_batches == 1


def test_builtin_filter_groups_forwards_configured_inflight_limit():
    trainer = _trainer_with_filter_groups({"enable": True, "metric": "acc", "max_inflight_gen_batches": 3})

    sampler = trainer._build_replay_buffer()

    assert sampler.max_inflight_gen_batches == 3


def test_builtin_sync_failure_refill_forces_single_prompt_generation():
    trainer = _trainer_with_filter_groups({"enable": False})
    trainer.config.trainer.v1.sampler.sync_refill_failed_groups = True

    sampler = trainer._build_replay_buffer()

    assert sampler.sync_refill_failed_groups is True
    assert sampler.gen_batch_size == 1


def test_sync_failure_refill_overrides_dataloader_generation_batch_size():
    trainer = _trainer_with_filter_groups({"enable": False})
    trainer.config.trainer.v1.sampler.sync_refill_failed_groups = True
    trainer.config.data.update(
        {
            "train_files": [],
            "val_files": [],
            "train_max_samples": -1,
            "val_max_samples": -1,
            "dataloader_num_workers": 0,
            "val_batch_size": 1,
            "validation_shuffle": False,
        }
    )
    trainer.config.trainer.total_epochs = 1
    trainer.config.trainer.total_training_steps = None
    trainer.parameter_sync_step = 1
    trainer.tokenizer = None
    trainer.processor = None

    with (
        patch("verl.trainer.ppo.v1.trainer_base.create_rl_dataset", side_effect=[[{}, {}], [{}]]),
        patch("verl.trainer.ppo.v1.trainer_base.create_rl_sampler", return_value=None),
        patch("verl.trainer.ppo.v1.trainer_base.StatefulDataLoader") as dataloader,
        patch("verl.trainer.ppo.v1.trainer_base.logger.warning") as warning,
    ):
        trainer._init_dataloader()

    assert trainer.config.data.gen_batch_size == 1
    assert dataloader.call_args_list[0].kwargs["batch_size"] == 1
    warning.assert_any_call("data.gen_batch_size=8 is overridden to 1.")


def test_builtin_filter_groups_warns_when_total_generation_limit_is_configured():
    trainer = _trainer_with_filter_groups({"enable": True, "metric": "acc", "max_num_gen_batches": 10})

    with patch("verl.trainer.ppo.v1.trainer_base.logger.warning") as warning:
        trainer._build_replay_buffer()

    warning.assert_called_once_with(
        "algorithm.filter_groups.max_num_gen_batches=%s is ignored by the built-in V1 ReplayBuffer; "
        "use max_inflight_gen_batches to bound concurrent Sync DAPO generation.",
        10,
    )


def _training_only_trainer():
    trainer = _trainer_with_filter_groups({"enable": False})
    trainer.config.data.update(
        {"train_files": ["train.parquet"], "val_files": [], "dataloader_num_workers": 0, "val_batch_size": None}
    )
    trainer.config.trainer.update(
        {"total_epochs": 1, "total_training_steps": None, "val_before_train": False, "val_only": False, "test_freq": -1}
    )
    trainer.parameter_sync_step = 1
    trainer.tokenizer = trainer.processor = None
    return trainer


def test_training_only_dataloader_does_not_load_an_empty_validation_dataset():
    trainer = _training_only_trainer()
    with (
        patch("verl.trainer.ppo.v1.trainer_base.create_rl_dataset", return_value=[{}] * 64) as dataset,
        patch("verl.trainer.ppo.v1.trainer_base.create_rl_sampler", return_value=None),
        patch("verl.trainer.ppo.v1.trainer_base.StatefulDataLoader") as loader,
    ):
        trainer._init_dataloader()
    assert dataset.call_count == loader.call_count == 1
    assert trainer.val_dataset == []
    assert trainer.val_dataloader is None
    assert trainer.total_training_steps == 1
    with pytest.raises(ValueError, match="nonempty data.val_files"):
        trainer._validate()


@pytest.mark.parametrize("option,value", [("test_freq", 20), ("val_before_train", True), ("val_only", True)])
def test_empty_validation_files_reject_requested_validation(option, value):
    trainer = _training_only_trainer()
    trainer.config.trainer[option] = value
    with patch("verl.trainer.ppo.v1.trainer_base.create_rl_dataset", return_value=[{}] * 64):
        with pytest.raises(ValueError, match="nonempty data.val_files"):
            trainer._init_dataloader()


def test_validation_only_logs_complete_validation_time_without_training():
    trainer = _StubTrainer.__new__(_StubTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "project_name": "test",
                "experiment_name": "validation",
                "logger": ["console"],
                "val_before_train": True,
                "val_only": True,
            }
        }
    )
    trainer.global_steps = 0
    trainer._validate = MagicMock(return_value={"eval/math/acc_mean": 0.5})
    trainer._shutdown_dump_executor = MagicMock()
    trainer.step = MagicMock(side_effect=AssertionError("Validation-only must not update parameters"))
    with (
        patch("verl.trainer.ppo.v1.trainer_base.SkipManager.init"),
        patch("verl.trainer.ppo.v1.trainer_base.Tracking") as tracking,
        patch("verl.trainer.ppo.v1.trainer_base.ValidationGenerationsLogger"),
        patch("verl.trainer.ppo.v1.trainer_base.DapoFilteredRewardTableLogger"),
    ):
        trainer.fit(agent_loop_manager=None)
    logged = tracking.return_value.log.call_args.kwargs
    assert logged["step"] == 0
    assert logged["data"]["eval/math/acc_mean"] == 0.5
    assert logged["data"]["perf/timing/validation_seconds"] >= 0
    trainer.step.assert_not_called()
    trainer._shutdown_dump_executor.assert_called_once()


@pytest.mark.parametrize("score_centering", [True, False])
def test_update_actor_passes_score_centering_flag(score_centering):
    trainer = _StubTrainer.__new__(_StubTrainer)
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {
                    "ppo_mini_batch_size": 2,
                    "calculate_entropy": False,
                    "entropy_coeff": 0.0,
                    "ppo_epochs": 1,
                    "data_loader_seed": 1,
                    "shuffle": False,
                    "policy_loss": {"rollout_correction": {"score_centering": score_centering}},
                },
                "rollout": {"n": 4, "temperature": 1.0},
            }
        }
    )
    trainer.actor_rollout_wg = MagicMock()
    trainer.actor_rollout_wg.update_actor.return_value = {"metrics": {"mfu": 0.0}}
    batch = SimpleNamespace(extra_info={})

    trainer._update_actor(batch, {})

    assert batch.extra_info["score_centering"] is score_centering


def _progress_trainer():
    trainer = _StubTrainer.__new__(_StubTrainer)
    trainer.actor_optimizer_step_attempts = trainer.actor_optimizer_step = 0
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {
                    "ppo_mini_batch_size": 64,
                    "calculate_entropy": False,
                    "entropy_coeff": 0.0,
                    "ppo_epochs": 4,
                    "data_loader_seed": 1,
                    "shuffle": False,
                    "policy_loss": {},
                },
                "rollout": {"n": 1, "temperature": 1.0},
            }
        }
    )
    trainer.actor_rollout_wg = MagicMock()
    return trainer


def test_actor_rpc_accumulates_once_and_finalizes_token_weighted_diagnostics():
    trainer = _progress_trainer()
    trainer.actor_rollout_wg.update_actor.return_value = tu.get_tensordict(
        {},
        {
            "optimizer_step_attempts": 4,
            "optimizer_steps": 3,
            "metrics": {
                "mfu": [0.0],
                "distillation/token_observations": [2.0, 5.0],
                "distillation/ppo_clip_count": [0.5, 1.0],
            },
        },
    )
    metrics = {}
    for _ in range(2):
        trainer._update_actor(SimpleNamespace(extra_info={}), metrics)
    assert trainer.actor_optimizer_step_attempts == 8 and trainer.actor_optimizer_step == 6
    assert metrics["correction/behavior_ppo/batch_active_fraction"] == 1.5 / 7
    assert "distill/token_observations" not in metrics
    assert not any(key.startswith("training/actor_optimizer") for key in metrics)
    assert trainer._actor_optimizer_metrics() == {
        "training/actor_optimizer_step_attempts": 8,
        "training/actor_optimizer_step": 6,
        "training/actor_optimizer_skipped_updates": 2,
    }
    trainer.actor_rollout_wg.update_actor.side_effect = RuntimeError("RPC failed")
    with pytest.raises(RuntimeError, match="RPC failed"):
        trainer._update_actor(SimpleNamespace(extra_info={}), {})
    assert trainer.actor_optimizer_step_attempts == 8 and trainer.actor_optimizer_step == 6


@pytest.mark.parametrize("attempts,steps", [(True, 1), (4.0, 4), (-1, 0), (2, 3)])
def test_invalid_rpc_counts_do_not_partially_accumulate(attempts, steps):
    trainer = _progress_trainer()
    with pytest.raises(ValueError):
        trainer._accumulate_actor_optimizer_steps(
            tu.get_tensordict({}, {"optimizer_step_attempts": attempts, "optimizer_steps": steps})
        )
    assert trainer.actor_optimizer_step_attempts == trainer.actor_optimizer_step == 0


def test_unknown_execution_does_not_hide_known_attempts():
    trainer = _progress_trainer()
    trainer._accumulate_actor_optimizer_steps(
        tu.get_tensordict({}, {"optimizer_step_attempts": 4, "optimizer_steps": None})
    )
    trainer._accumulate_actor_optimizer_steps(
        tu.get_tensordict({}, {"optimizer_step_attempts": 2, "optimizer_steps": 2})
    )
    assert trainer._actor_optimizer_metrics() == {"training/actor_optimizer_step_attempts": 6}


def test_fit_logs_optimizer_progress_after_step_aggregation_and_initial_validation():
    trainer = _progress_trainer()
    trainer.config.trainer = {
        "project_name": "test",
        "experiment_name": "progress",
        "logger": ["console"],
        "val_before_train": True,
        "total_epochs": 1,
        "save_freq": -1,
        "test_freq": -1,
    }
    trainer.config.global_profiler = {"steps": None}
    trainer.global_steps = 0
    trainer.steps_per_epoch = trainer.total_training_steps = 2
    trainer._validate = MagicMock(return_value={"val/acc": 0.5})
    for name in ("_shutdown_dump_executor", "_reissue_inflight_prompts", "_start_profiling", "_stop_profiling"):
        setattr(trainer, name, MagicMock())
    batch = SimpleNamespace(keys=[], partition_id="train")

    def step(metrics, timing):
        trainer._accumulate_actor_optimizer_steps(
            tu.get_tensordict({}, {"optimizer_step_attempts": 4, "optimizer_steps": 4})
        )
        return batch

    trainer.step = step
    trainer._compute_metrics = MagicMock(
        side_effect=lambda batch, metrics, timing, **kw: metrics.update({"training/actor_optimizer_step": -100})
    )
    with (
        patch("verl.trainer.ppo.v1.trainer_base.SkipManager"),
        patch("verl.trainer.ppo.v1.trainer_base.Tracking") as tracking,
        patch("verl.trainer.ppo.v1.trainer_base.ValidationGenerationsLogger"),
        patch("verl.trainer.ppo.v1.trainer_base.DapoFilteredRewardTableLogger"),
        patch("verl.trainer.ppo.v1.trainer_base.tq.kv_clear"),
    ):
        trainer.fit(None)
    logged = [call.kwargs for call in tracking.return_value.log.call_args_list]
    assert [entry["step"] for entry in logged] == [0, 1, 2]
    assert [entry["data"]["step/optimizer"] for entry in logged] == [0, 4, 8]


@pytest.mark.parametrize("interval", [-1, True, 1.5, "8"])
def test_metrics_quantile_interval_is_validated_before_logging(interval):
    trainer = _StubTrainer.__new__(_StubTrainer)
    trainer.config = OmegaConf.create({"trainer": {"metrics_quantile_interval": interval}})
    with (
        patch("verl.trainer.ppo.v1.trainer_base.SkipManager.init"),
        pytest.raises(ValueError, match="nonnegative integer"),
    ):
        trainer.fit(None)


def test_critic_records_have_independent_real_counters_and_checkpoint(tmp_path):
    trainer = _progress_trainer()
    trainer.use_critic = True
    trainer.global_steps = 5
    trainer.critic_optimizer_step_attempts = trainer.critic_optimizer_step = 0
    trainer.config.critic = {"ppo_mini_batch_size": 2, "ppo_epochs": 1, "data_loader_seed": 42, "shuffle": False}
    trainer.logger = MagicMock()
    result = tu.get_tensordict(
        {},
        {
            "metrics": {},
            "optimizer_step_attempts": 2,
            "optimizer_steps": 1,
            "update_records": [
                {"metrics": {"optim/critic/loss": 1.0}, "applied": True},
                {"metrics": {"optim/critic/loss": 2.0}, "applied": False},
            ],
        },
    )
    trainer.critic_wg = SimpleNamespace(train_mini_batch=lambda batch: SimpleNamespace(get=lambda: result))
    trainer._update_critic(SimpleNamespace(extra_info={}), {})
    rows = [c.kwargs["data"] for c in trainer.logger.log.call_args_list]
    assert [r["step/critic_update"] for r in rows] == [1, 2]
    assert [r["step/critic_optimizer"] for r in rows] == [1, 1]
    assert trainer.actor_optimizer_step == 0
    trainer._save_actor_optimizer_progress(tmp_path)
    trainer.critic_optimizer_step = 999
    trainer._load_actor_optimizer_progress(tmp_path)
    assert trainer.critic_optimizer_step == 1 and trainer.critic_optimizer_step_attempts == 2
