# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Shared teacher contracts: configuration, phase ordering, TQ alignment and failure isolation."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tensordict import TensorDict
from transfer_queue import KVBatchMeta

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput, AgentLoopWorker
from verl.trainer.distillation.losses import distillation_loss
from verl.trainer.ppo.utils import Role
from verl.trainer.ppo.v1 import trainer_base
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass, validate_config
from verl.workers.config import DistillationConfig, FSDPEngineConfig
from verl.workers.engine.fsdp import transformer_impl
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine
from verl.workers.utils.padding import no_padding_2_padding


@pytest.fixture
def config():
    root = Path(__file__).resolve().parents[4]
    with initialize_config_dir(config_dir=str(root / "verl/trainer/config"), version_base=None):
        cfg = compose(
            config_name="ppo_trainer",
            overrides=[
                "trainer.use_v1=True",
                "trainer.v1.trainer_mode=sync",
                "trainer.nnodes=1",
                "trainer.n_gpus_per_node=8",
                "actor_rollout_ref.actor.strategy=fsdp2",
                "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
                "actor_rollout_ref.actor.fsdp_config.param_offload=True",
                "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
                "actor_rollout_ref.rollout.name=vllm",
                "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
                "distillation.enabled=True",
                "distillation.execution_mode=shared_sync",
                "distillation.nnodes=0",
                "distillation.n_gpus_per_node=0",
                "distillation.teacher_models.teacher_model.model_path=/teacher",
                "distillation.distillation_loss.loss_mode=k1",
                "distillation.distillation_loss.use_policy_gradient=True",
                "distillation.distillation_loss.use_task_rewards=False",
                "actor_rollout_ref.model.use_remove_padding=True",
                "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
            ],
        )
    return cfg


def _trainer(config):
    trainer = PPOTrainerSync.__new__(PPOTrainerSync)
    trainer.config = config
    trainer.distillation_config = omega_conf_to_dataclass(config.distillation)
    return trainer


def _nested(rows, dtype=torch.int64):
    return torch.nested.nested_tensor([torch.tensor(row, dtype=dtype) for row in rows], layout=torch.jagged)


def test_shared_config_preserves_inference_budget_and_legacy_defaults(config):
    parsed = omega_conf_to_dataclass(config.distillation)
    assert parsed.teacher_models["default"].num_replicas == 0
    assert parsed.teacher_models["default"].inference.response_length == config.data.max_response_length
    assert DistillationConfig().execution_mode == "standalone"
    assert FSDPEngineConfig().forward_only_cpu_offload
    validate_config(config, use_reference_policy=False, use_critic=False)


@pytest.mark.parametrize(
    "path,value",
    [
        ("trainer.v1.trainer_mode", "colocate_async"),
        ("trainer.nnodes", 2),
        ("trainer.v1.sync.parameter_sync_step", 2),
        ("actor_rollout_ref.rollout.free_cache_engine", False),
        ("actor_rollout_ref.rollout.tensor_model_parallel_size", 2),
        ("actor_rollout_ref.model.use_fused_kernels", True),
        ("actor_rollout_ref.actor.fsdp_config.param_offload", False),
        ("actor_rollout_ref.actor.fsdp_config.optimizer_offload", False),
        ("actor_rollout_ref.model.lora_rank", 8),
        ("algorithm.filter_groups.enable", True),
        ("skip.rollout_tq.enable", True),
        ("distillation.n_gpus_per_node", 4),
        ("distillation.shared_teacher.forward_only_cpu_offload", True),
        ("distillation.shared_teacher.reshard_after_forward", False),
        ("distillation.shared_teacher.infer_max_token_len_per_gpu", 1),
        ("distillation.distillation_loss.use_task_rewards", True),
    ],
)
def test_incompatible_shared_configuration_fails_before_workers(config, path, value):
    OmegaConf.update(config, path, value, force_add=True)
    with pytest.raises(Exception, match="shared_sync|shared teacher|Shared"):
        validate_config(config, use_reference_policy=False, use_critic=False)


def test_shared_pool_has_no_dedicated_teacher_allocation(config, monkeypatch):
    trainer = _trainer(config)
    pool_mgr = MagicMock()
    monkeypatch.setattr(trainer_base, "ResourcePoolManager", pool_mgr)
    trainer._init_resource_pool_mgr()
    assert pool_mgr.call_args.kwargs["resource_pool_spec"] == {"global_pool": [8]}
    assert trainer.mapping[Role.TeacherModel] == trainer.mapping[Role.ActorRollout] == "global_pool"
    trainer.teacher_model_manager = None
    assert trainer.get_teacher_client() is None


@pytest.mark.parametrize("teacher_fails", [False, True])
def test_shared_initialization_offloads_models_before_starting_student_servers(config, monkeypatch, teacher_fails):
    trainer = _trainer(config)
    trainer.use_teacher_policy = True
    trainer.use_reference_policy = trainer.use_critic = False
    for name in ("_init_tokenizer", "_init_dataloader", "_init_dump_executor"):
        setattr(trainer, name, MagicMock())
    events = []
    worker_config = object()
    trainer._make_shared_teacher_config = MagicMock(return_value=worker_config)
    pool = object()
    manager = SimpleNamespace(
        create_resource_pool=lambda: None,
        resource_pool_dict={"global_pool": pool},
        get_resource_pool=lambda role: pool,
    )
    monkeypatch.setattr(trainer_base, "ResourcePoolManager", lambda **kwargs: manager)
    actor = SimpleNamespace(init_model=lambda: events.append("actor_init"))

    def reset_teacher():
        events.append("teacher_init")
        if teacher_fails:
            raise RuntimeError("teacher initialization failed")

    teacher = SimpleNamespace(reset=reset_teacher)
    group = SimpleNamespace(spawn=lambda **kwargs: {str(Role.ActorRollout): actor, str(Role.TeacherModel): teacher})
    colocate = MagicMock(return_value=object())
    monkeypatch.setattr(trainer_base, "create_colocated_worker_cls", colocate)
    monkeypatch.setattr(trainer_base, "RayWorkerGroup", lambda **kwargs: group)
    monkeypatch.setattr(trainer_base, "RewardLoopManager", lambda **kwargs: object())
    standalone_manager = MagicMock()
    monkeypatch.setattr(trainer_base, "MultiTeacherModelManager", standalone_manager)
    servers = SimpleNamespace(get_replicas=lambda: [])
    monkeypatch.setattr(
        trainer_base.LLMServerManager, "create", lambda **kwargs: (events.append("student_servers"), servers)[1]
    )
    checkpoint = SimpleNamespace(sleep_replicas=lambda: events.append("student_sleep"))
    monkeypatch.setattr(trainer_base, "CheckpointEngineManager", lambda **kwargs: checkpoint)
    trainer._load_checkpoint = lambda: events.append("load_checkpoint")
    if teacher_fails:
        with pytest.raises(RuntimeError, match="teacher initialization"):
            trainer._setup()
        assert events == ["actor_init", "teacher_init"]
    else:
        trainer._setup()
        assert events == ["actor_init", "teacher_init", "student_servers", "student_sleep", "load_checkpoint"]
        assert trainer.shared_teacher_wg is teacher
    registrations = colocate.call_args.kwargs["class_dict"]
    assert set(registrations) == {str(Role.ActorRollout), str(Role.TeacherModel)}
    assert registrations[str(Role.TeacherModel)].kwargs["config"] is worker_config
    standalone_manager.assert_not_called()


def test_teacher_model_config_is_fresh_and_uses_teacher_path(config, monkeypatch):
    trainer = _trainer(config)
    tokenizer = SimpleNamespace(get_vocab=lambda: {"a": 0, "b": 1}, bos_token_id=0, eos_token_id=1, pad_token_id=1)
    trainer.tokenizer = tokenizer
    model = SimpleNamespace(tokenizer=tokenizer, hf_config=SimpleNamespace())
    factory = MagicMock(return_value=model)
    monkeypatch.setattr(trainer_base, "HFModelConfig", factory)
    worker = trainer._make_shared_teacher_config()
    assert factory.call_args.kwargs["path"] == "/teacher"
    assert not factory.call_args.kwargs["use_fused_kernels"]
    assert not factory.call_args.kwargs["enable_gradient_checkpointing"]
    assert worker.engine_config.fsdp_size == 8
    assert trainer.distillation_config.shared_teacher.fsdp_size == -1
    assert worker.optimizer_config is None
    tokenizer.get_vocab = lambda: {"wrong": 0}
    trainer.tokenizer = SimpleNamespace(get_vocab=lambda: {"a": 0, "b": 1})
    with pytest.raises(ValueError, match="vocabularies"):
        trainer._make_shared_teacher_config()


@pytest.mark.parametrize(
    "attributes",
    [
        {"vision_config": {}},
        {"audio_config": {}},
        {"is_encoder_decoder": True},
        {"num_experts": 8},
    ],
)
def test_shared_model_rejects_non_dense_text_architectures(attributes):
    model = SimpleNamespace(hf_config=SimpleNamespace(**attributes))
    with pytest.raises(ValueError, match="dense text"):
        PPOTrainerSync._validate_shared_model(model)


def _score_fixture(config, monkeypatch, fail=None):
    trainer = _trainer(config)
    batch = KVBatchMeta(
        keys=["u2_0_0", "u1_0_0"],
        tags=[{}, {}],
        partition_id="train",
        extra_info={"temperature": 0.7, "calculate_entropy": True},
    )
    rows = [[4, 5, 6, 7], [8, 9, 10]]
    data = TensorDict(
        {
            "input_ids": _nested(rows),
            "log_probs": _nested([[-1.0, -2.0, -3.0, 999.0], [-4.0, -5.0, 999.0]], torch.float32),
        },
        batch_size=[2],
    )
    events, submitted = [], []

    def move(device, **kwargs):
        events.append(device)
        assert kwargs == {"model": True, "optimizer": False, "grad": False}
        if fail == device:
            raise RuntimeError(f"{device} failure")

    def infer(meta):
        events.append("infer")
        submitted.append(meta)
        if fail == "infer":
            raise RuntimeError("infer failure")
        return SimpleNamespace(futures=["future"])

    def wait(futures):
        events.append("wait")
        assert futures == ["future"]
        if fail == "wait":
            raise RuntimeError("wait failure")
        return []

    trainer.shared_teacher_wg = SimpleNamespace(to=move, infer_batch=infer)
    monkeypatch.setattr(trainer_base.ray, "get", wait)
    monkeypatch.setattr(trainer_base.tq, "kv_batch_get", lambda **kwargs: data)
    published = []
    monkeypatch.setattr(trainer_base.tq, "kv_batch_put", lambda **kwargs: published.append(kwargs))
    return trainer, batch, data, events, submitted, published


def test_scoring_waits_offloads_and_preserves_keys_metadata_and_causal_alignment(config, monkeypatch):
    trainer, batch, data, events, submitted, published = _score_fixture(config, monkeypatch)
    timings = {}
    assert trainer._compute_shared_teacher_log_probs(batch, timings) is batch
    assert events == ["device", "infer", "wait", "cpu"]
    assert set(timings) == {"teacher_forward", "teacher_transfer"}
    assert batch.extra_info == {"temperature": 0.7, "calculate_entropy": True}
    assert submitted[0].extra_info["temperature"] == 1
    assert submitted[0].fields == ["input_ids", "position_ids", "loss_mask"]
    assert published[0]["keys"] == batch.keys
    fields = published[0]["fields"]
    assert [r.flatten().tolist() for r in fields["teacher_ids"].unbind()] == [[5, 6, 7, 0], [9, 10, 0]]
    assert [r.flatten().tolist() for r in fields["teacher_logprobs"].unbind()] == [
        [-1.0, -2.0, -3.0, 0.0],
        [-4.0, -5.0, 0.0],
    ]
    assert data["log_probs"].unbind()[0][-1] == 999  # no in-place change to fetched storage

    # Exercise the production response slicing and loss with unequal prompt/response lengths.
    data.update(fields)
    data["prompts"] = _nested([[4, 5], [8]])
    data["responses"] = _nested([[6, 7], [9, 10]])
    data["response_mask"] = _nested([[1, 1], [1, 0]])  # masked final token
    current = torch.tensor([-0.9, -2.1, -3.2, 0.0, -4.2, -5.1, 0.0], requires_grad=True)
    output = {"log_probs": torch.nested.nested_tensor_from_jagged(current, offsets=data["input_ids"].offsets())}
    data["old_log_probs"] = no_padding_2_padding(output["log_probs"], data).detach()
    tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=3, global_batch_size=2)
    actor = SimpleNamespace(loss_agg_mode="token-mean", global_batch_info={}, loss_scale_factor=None)
    loss, _ = distillation_loss(actor, trainer.distillation_config, output, data)
    loss.backward()
    # At ratio=1, gradient = -(teacher_logp-current_logp) / number of unmasked tokens.
    torch.testing.assert_close(current.grad, torch.tensor([0.0, -0.1 / 3, -0.2 / 3, 0.0, -0.2 / 3, 0.0, 0.0]))
    assert torch.isfinite(loss)


@pytest.mark.parametrize("failure", ["device", "infer", "wait", "cpu"])
def test_failed_scoring_never_publishes_teacher_fields(config, monkeypatch, failure):
    trainer, batch, _, events, _, published = _score_fixture(config, monkeypatch, fail=failure)
    with pytest.raises(RuntimeError, match=failure):
        trainer._compute_shared_teacher_log_probs(batch, {})
    assert events[-1] == "cpu"
    assert not published


def test_nonfinite_scores_rejected_after_teacher_released(config, monkeypatch):
    trainer, batch, data, events, _, published = _score_fixture(config, monkeypatch)
    data["log_probs"].values()[0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        trainer._compute_shared_teacher_log_probs(batch, {})
    assert events[-1] == "cpu"
    assert not published


@pytest.mark.parametrize("mode", ["standalone", "shared_sync", "full_reverse_kl"])
def test_step_order_reuses_update_and_only_inserts_shared_scoring(config, mode):
    trainer = _trainer(config)
    config.distillation.execution_mode = "shared_sync" if mode == "full_reverse_kl" else mode
    if mode == "full_reverse_kl":
        config.distillation.distillation_loss.loss_mode = mode
    trainer.global_steps = 1
    trainer.use_reference_policy = trainer.use_critic = False
    trainer.reward_loop_manager = SimpleNamespace(reward_loop_worker_handles=[object()])
    events = []
    batch = KVBatchMeta(keys=["u_0_0"], tags=[{}], partition_id="train")
    trainer.replay_buffer = SimpleNamespace(sample=lambda **kwargs: (batch, {}))
    trainer.on_sample_begin = lambda: None
    trainer.on_sample_end = lambda: events.append("sleep")
    for name in ("_balance_batch", "_compute_old_log_prob", "_compute_advantage", "_update_actor"):
        setattr(trainer, name, lambda batch, name=name, **kwargs: (events.append(name), batch)[1])
    trainer._compute_shared_teacher_log_probs = lambda batch, timing: (events.append("teacher"), batch)[1]
    trainer._step_once({}, {}, 1)
    assert events == [
        "sleep",
        "_balance_batch",
        *(["teacher"] if mode != "standalone" else []),
        *(["_compute_old_log_prob", "_compute_advantage"] if mode != "full_reverse_kl" else []),
        "_update_actor",
    ]


def test_shared_agent_skips_teacher_rpc_without_disabling_distillation():
    worker = SimpleNamespace(distillation_enabled=True, teacher_server_manager=None)
    output = AgentLoopOutput(prompt_ids=[1], response_ids=[2], response_mask=[1], metrics=AgentLoopMetrics())
    asyncio.run(AgentLoopWorker._compute_teacher_logprobs(worker, output, [1], [2], False))
    assert worker.distillation_enabled
    assert "teacher_logprobs" not in output.extra_fields


@pytest.mark.parametrize(
    "forward_only,cpu_policy,expected",
    [
        (True, False, ["load", "sync", "offload", "sync"]),
        (True, True, []),
        (False, False, ["load", "offload"]),
    ],
)
def test_phase_transfer_completion_and_legacy_behavior(monkeypatch, forward_only, cpu_policy, expected):
    engine = FSDPEngine.__new__(FSDPEngine)
    engine.engine_config = FSDPEngineConfig(strategy="fsdp2", forward_only=forward_only)
    engine._uses_fsdp2_cpu_offload_policy = cpu_policy
    engine.module = object()
    events = []
    monkeypatch.setattr(transformer_impl, "get_device_name", lambda: "cuda")
    monkeypatch.setattr(
        transformer_impl, "get_torch_device", lambda: SimpleNamespace(synchronize=lambda: events.append("sync"))
    )
    monkeypatch.setattr(transformer_impl, "load_fsdp_model_to_gpu", lambda model: events.append("load"))
    monkeypatch.setattr(transformer_impl, "offload_fsdp_model_to_cpu", lambda model: events.append("offload"))
    engine.to("cuda", model=True, optimizer=False, grad=False)
    engine.to("cpu", model=True, optimizer=False, grad=False)
    assert events == expected
