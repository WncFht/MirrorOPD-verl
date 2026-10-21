# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Real CPU AdamW updates through engine/worker aggregation and trainer logging."""

from contextlib import nullcontext
from functools import partial
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl.trainer.distillation.losses import distillation_ppo_loss
from verl.trainer.ppo.v1.trainer_base import PPOTrainer
from verl.utils import tensordict_utils as tu
from verl.utils.py_functional import append_to_dict
from verl.workers import engine_workers
from verl.workers.config import DistillationLossConfig
from verl.workers.engine.base import BaseEngine
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine
from verl.workers.engine_workers import TrainingWorker


class Trainer(PPOTrainer):
    def on_step_end(self):
        pass

    def on_sample_end(self):
        pass


@pytest.mark.parametrize("interval,expected_quantiles", [(0, []), (1, [1, 2, 3, 4]), (2, [1, 2, 4])])
def test_actual_updates_microbatches_scheduler_and_log_coordinates(monkeypatch, interval, expected_quantiles):
    monkeypatch.setattr(
        engine_workers,
        "get_torch_device",
        lambda: SimpleNamespace(max_memory_allocated=lambda: 0, max_memory_reserved=lambda: 0),
    )
    torch.manual_seed(7)
    model = torch.nn.Linear(3, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    engine = SimpleNamespace(
        module=model,
        optimizer=optimizer,
        optimizer_config=SimpleNamespace(clip_grad=1.0),
        scaler=None,
        _qat_enabled=False,
        train_mode=lambda **kwargs: nullcontext(),
        get_data_parallel_rank=lambda: 0,
        get_data_parallel_size=lambda: 1,
        get_data_parallel_group=lambda: None,
        is_mp_src_rank_with_outputs=lambda: True,
        optimizer_zero_grad=optimizer.zero_grad,
    )
    executed = []

    def optimizer_step():
        if len(executed) == 1:
            for parameter in model.parameters():
                parameter.grad.fill_(float("inf"))
        norm = FSDPEngine.optimizer_step(engine)
        executed.append(engine._optimizer_step_applied)
        return norm

    engine.optimizer_step = optimizer_step
    scheduler = MagicMock(return_value=0.005)
    engine.lr_scheduler_step = scheduler
    actor_cfg = SimpleNamespace(loss_agg_mode="token-mean", global_batch_info={}, loss_scale_factor=None)
    loss_cfg = DistillationLossConfig(
        loss_mode="k1",
        policy_loss_mode="token_is",
        advantage_source="current",
        use_task_rewards=False,
        use_policy_gradient=True,
        loss_max_clamp=None,
        log_prob_min_clamp=None,
    )
    loss_fn = partial(distillation_ppo_loss, actor_cfg, SimpleNamespace(distillation_loss=loss_cfg))

    def forward_backward(data, loss_function, forward_only=False):
        assert not forward_only
        losses, metrics = [], {}
        for idx in [slice(0, 1), slice(1, None)]:
            features = data["features"][idx]
            size = features.shape[0]
            lp = model(features).log_softmax(-1)[:, 0]
            packed_lp = torch.stack((lp, torch.zeros_like(lp)), -1).reshape(-1)
            teacher = torch.stack((torch.full_like(lp, -0.4), torch.zeros_like(lp)), -1).reshape(-1, 1)
            mini = TensorDict(
                {
                    "prompts": torch.nested.nested_tensor(
                        [torch.ones(1, dtype=torch.long) for _ in range(size)], layout=torch.jagged
                    ),
                    "responses": torch.nested.nested_tensor(
                        [torch.ones(1, dtype=torch.long) for _ in range(size)], layout=torch.jagged
                    ),
                    "response_mask": torch.ones(size, 1, dtype=torch.bool),
                    "teacher_logprobs": teacher,
                    "rollout_log_probs": data["behavior"][idx],
                    "old_log_probs": data["behavior"][idx],
                },
                batch_size=[],
            )
            tu.assign_non_tensor(
                mini,
                dp_size=1,
                batch_num_tokens=data.shape[0],
                global_batch_size=data.shape[0],
                max_response_len=1,
                collect_metrics=True,
                metrics_quantile=tu.get(data, "metrics_quantile"),
            )
            loss, observations = loss_function(model_output={"log_probs": packed_lp}, data=mini)
            loss.backward()
            losses.append(loss.item())
            append_to_dict(metrics, observations)
        return {"loss": losses, "metrics": metrics, "model_output": {}}

    engine.forward_backward_batch = forward_backward
    engine.train_batch = lambda data, loss_function: BaseEngine.train_batch(engine, data, loss_function)
    worker = SimpleNamespace(
        engine=engine,
        loss_fn=loss_fn,
        profiler=SimpleNamespace(rank=0, check_enable=lambda: False, step=MagicMock()),
        device_name="cpu",
        flops_counter=None,
        model_config={},
        engine_config=SimpleNamespace(
            strategy="cpu",
            forward_only=False,
            use_dynamic_bsz=False,
            max_token_len_per_gpu=128,
            micro_batch_size_per_gpu=1,
            use_fused_kernels=False,
        ),
    )
    worker.train_batch = MethodType(TrainingWorker.train_batch, worker)
    worker._postprocess_output = MethodType(TrainingWorker._postprocess_output, worker)
    features = torch.randn(8, 3)
    data = TensorDict(
        {"features": features, "behavior": model(features).log_softmax(-1)[:, :1].detach()}, batch_size=[8]
    )

    def update_actor(batch):
        tu.assign_non_tensor(data, **batch.extra_info)
        return TrainingWorker.train_mini_batch(worker, data)

    trainer = Trainer.__new__(Trainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {"metrics_quantile_interval": interval},
            "actor_rollout_ref": {
                "actor": {
                    "strategy": "fsdp2",
                    "ppo_mini_batch_size": 4,
                    "ppo_epochs": 2,
                    "calculate_entropy": False,
                    "entropy_coeff": 0.0,
                    "data_loader_seed": 1,
                    "shuffle": False,
                    "policy_loss": {},
                },
                "rollout": {"n": 1, "temperature": 1.0},
            },
        }
    )
    trainer.actor_optimizer_step_attempts = trainer.actor_optimizer_step = 0
    trainer.global_steps = 1
    trainer.logger = MagicMock()
    trainer.actor_rollout_wg = SimpleNamespace(update_actor=update_actor)
    trainer._update_actor(SimpleNamespace(extra_info={}), {})
    rows = [c.kwargs["data"] for c in trainer.logger.log.call_args_list]
    assert len(rows) == 4
    assert [r["step/update"] for r in rows] == [1, 2, 3, 4]
    assert [r["step/optimizer"] for r in rows] == [1, 1, 2, 3]
    assert executed == [True, False, True, True]
    assert scheduler.call_count == 1
    assert all(r["optim/actor/lr"] == 0.01 for r in rows)
    assert [i + 1 for i, r in enumerate(rows) if "policy/current_behavior/ratio_p99" in r] == expected_quantiles
    assert rows[0]["policy/current_behavior/log_ratio_m2"] < 1e-12
    assert rows[-1]["policy/current_behavior/log_ratio_m2"] > 0
    assert all("advantage/distillation/abs_mean" in r for r in rows)
    assert trainer.actor_optimizer_step_attempts == 4 and trainer.actor_optimizer_step == 3
