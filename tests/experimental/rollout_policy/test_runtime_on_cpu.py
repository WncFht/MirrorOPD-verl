# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
import asyncio
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl.experimental.rollout_policy.lifecycle import ExecutionContext, PreparedPhase, RequestRoutes
from verl.experimental.rollout_policy.models import resolve_bindings
from verl.experimental.rollout_policy.plan import RolloutPolicyConfig, build_plan
from verl.experimental.rollout_policy.sessions import SessionDriver
from verl.utils import tensordict_utils as tu
from verl.workers.rollout.vllm_rollout.policy_execution import SingleVLLMExecutor


def test_immutable_sampling_plan_reaches_every_real_session_metadata(tmp_path):
    captured = []

    async def generate(sessions, client, deadline, drain):
        captured.extend(sessions)
        return [(session["request"].key, session["request"].source) for session in sessions]

    worker = SimpleNamespace(generate_sessions=SimpleNamespace(remote=generate))
    policy = RolloutPolicyConfig.parse({"mode": "trajectory_mixture"})
    config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"n": 3}}})
    context = ExecutionContext(config, policy, None)
    runtime = SessionDriver(context, [worker, worker], tu.get_tensordict({"uid": ["a", "b"], "index": [11, 12]}))
    plan = build_plan(
        policy,
        uids=["a", "b"],
        stable_ids=[11, 12],
        n=3,
        collection_step=1,
        optimizer_step=0,
        seed=42,
        bindings=resolve_bindings(policy, str(tmp_path), str(tmp_path)),
        sampling={"temperature": 0.0, "top_p": 1.0, "top_k": -1},
    )

    async def run():
        for phase in plan.phases:
            prepared = PreparedPhase(phase.source, None, RequestRoutes.build(phase.requests, 2), None, ())
            results = await runtime.generate(phase, plan, prepared)
            assert set(results) == {(request.key, request.source) for request in phase.requests}

    asyncio.run(run())
    assert len(captured) == 6
    for session in captured:
        assert session["sampling"]["seed"] == session["request"].seed
        assert session["metadata"]["temperature"] == 0.0
        assert session["metadata"]["actor_version"] == (0, 0)
        assert session["metadata"]["source"] == session["request"].source


def test_preparation_and_release_follow_resolved_binding_not_slot(tmp_path):
    from unittest.mock import AsyncMock

    policy = RolloutPolicyConfig.parse(
        {"mode": "token_mixture", "left": {"kind": "initial"}, "right": {"kind": "current"}, "right_weight": 1}
    )
    sync = AsyncMock()
    sleep = lambda: None
    trainer = SimpleNamespace(
        config=OmegaConf.create({"actor_rollout_ref": {"rollout": {"n": 1}}}),
        rollout_policy=policy,
        actor_rollout_wg=SimpleNamespace(update_weights=sync, world_size=1),
        get_llm_client=lambda: "current-client",
        checkpoint_manager=SimpleNamespace(sleep_replicas=sleep, replicas=[]),
    )
    executor = SingleVLLMExecutor(
        ExecutionContext(trainer.config, policy, trainer.actor_rollout_wg, trainer.checkpoint_manager)
    )
    executor.frozen_client = AsyncMock(side_effect=AssertionError("endpoint allocated fixed model"))
    plan = build_plan(
        policy,
        uids=["u"],
        stable_ids=[0],
        n=1,
        collection_step=3,
        optimizer_step=2,
        seed=1,
        bindings=resolve_bindings(policy, str(tmp_path), None),
    )

    async def run():
        prepared = await executor.prepare(plan.phases[0], plan)
        assert prepared.resource_key is None
        await prepared.release()

    asyncio.run(run())
    sync.assert_awaited_once_with(global_steps=2)
    executor.frozen_client.assert_not_called()


def test_fixed_a_endpoint_does_not_prepare_learner_or_other_source(tmp_path):
    from unittest.mock import AsyncMock

    extra = tmp_path / "A"
    extra.mkdir()
    policy = RolloutPolicyConfig.parse(
        {
            "mode": "token_mixture",
            "left": {"kind": "checkpoint", "ref": str(extra)},
            "right": {"kind": "current"},
            "right_weight": 0,
        }
    )
    trainer = SimpleNamespace(
        config=OmegaConf.create({"actor_rollout_ref": {"rollout": {"n": 1}}}),
        rollout_policy=policy,
        actor_rollout_wg=SimpleNamespace(world_size=1, park_fusion_actor=AsyncMock()),
    )
    executor = SingleVLLMExecutor(ExecutionContext(trainer.config, policy, trainer.actor_rollout_wg))
    executor.frozen_client = AsyncMock(return_value="A-client")
    plan = build_plan(
        policy,
        uids=["u"],
        stable_ids=[0],
        n=1,
        collection_step=1,
        optimizer_step=0,
        seed=1,
        bindings=resolve_bindings(policy, str(tmp_path), None),
    )
    prepared = asyncio.run(executor.prepare(plan.phases[0], plan))
    executor.frozen_client.assert_awaited_once_with(plan.binding("left"), 1)
    trainer.actor_rollout_wg.park_fusion_actor.assert_awaited_once()
    assert prepared.client == "A-client"


@pytest.mark.parametrize("head_dim,kv_heads", [(None, None), (128, 8)])
@pytest.mark.parametrize("explicit_kv", [None, 123456789])
def test_frozen_startup_preserves_declared_memory_ratio_and_resolves_context(
    monkeypatch, tmp_path, head_dim, kv_heads, explicit_kv
):
    from unittest.mock import AsyncMock

    from verl.workers.rollout.vllm_rollout import policy_execution as module

    tokenizer = SimpleNamespace(get_vocab=lambda: {"x": 0})
    model = SimpleNamespace(
        tokenizer=tokenizer,
        hf_config=SimpleNamespace(
            num_attention_heads=32,
            num_key_value_heads=kv_heads,
            hidden_size=2560,
            num_hidden_layers=36,
            head_dim=head_dim,
        ),
    )
    monkeypatch.setattr(module, "HFModelConfig", lambda **kwargs: model)
    captured = []

    class Replica:
        def __init__(self, rank, rollout, model, **kwargs):
            captured.append(rollout)
            self.launch_servers = AsyncMock()
            self._server_handle = SimpleNamespace(collective_rpc=SimpleNamespace(remote=AsyncMock()))

    monkeypatch.setattr(module, "get_rollout_replica_class", lambda name: Replica)
    config = OmegaConf.create(
        {
            "trainer": {"n_gpus_per_node": 2},
            "actor_rollout_ref": {
                "rollout": {
                    "gpu_memory_utilization": 0.2,
                    "tensor_model_parallel_size": 2,
                    "data_parallel_size": 1,
                    "pipeline_model_parallel_size": 1,
                    "expert_parallel_size": 2,
                    "max_model_len": None,
                    "prompt_length": 4096,
                    "response_length": 7168,
                    "max_num_seqs": 8,
                    "engine_kwargs": {"vllm": {"kv_cache_memory_bytes": explicit_kv}},
                }
            },
        }
    )
    executor = SingleVLLMExecutor(
        ExecutionContext(
            config, RolloutPolicyConfig(), SimpleNamespace(workers=[object(), object()]), tokenizer=tokenizer
        )
    )
    from verl.experimental.rollout_policy.plan import ModelBinding

    asyncio.run(executor.frozen_client(ModelBinding("checkpoint", str(tmp_path)), 2))
    assert captured[0].gpu_memory_utilization == 0.2
    assert captured[0].max_model_len == 11264
    assert captured[0].max_num_seqs == 2
    assert len(captured) == 2
    for rollout in captured:
        assert rollout.tensor_model_parallel_size == rollout.data_parallel_size == 1
        assert rollout.pipeline_model_parallel_size == rollout.expert_parallel_size == 1
    assert config.actor_rollout_ref.rollout.tensor_model_parallel_size == 2
    assert config.actor_rollout_ref.rollout.expert_parallel_size == 2
    # 36-layer model: explicit 128 differs from hidden_size / heads = 80.
    kv_per_token = 147456 if head_dim else 368640
    quantum = 256 * 1024**2
    expected_kv = ((kv_per_token * 11264 * 2 + quantum - 1) // quantum) * quantum
    assert captured[0].engine_kwargs.vllm.kv_cache_memory_bytes == (explicit_kv or expected_kv)


def test_generation_timing_excludes_post_generation_audit(monkeypatch, tmp_path):
    from verl.experimental.rollout_policy import sessions as module

    clock = [0.0]
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock[0]))

    async def generate(sessions, client, deadline, drain):
        clock[0] += 2.0
        return ["generated"]

    async def audit():
        clock[0] += 5.0
        return [{"peak_allocated_bytes": 1, "peak_reserved_bytes": 2}]

    teacher = tmp_path / "teacher"
    teacher.mkdir()
    policy = RolloutPolicyConfig.parse({"mode": "token_mixture", "right_weight": 0.5})
    context = ExecutionContext(None, policy, None)
    worker = SimpleNamespace(generate_sessions=SimpleNamespace(remote=generate))
    runtime = SessionDriver(context, [worker], tu.get_tensordict({"uid": ["a"]}))
    plan = build_plan(
        policy,
        uids=["a"],
        stable_ids=[0],
        n=1,
        collection_step=1,
        optimizer_step=0,
        seed=42,
        bindings=resolve_bindings(policy, str(tmp_path), str(teacher)),
        sampling={"temperature": 1.0, "top_p": 1.0, "top_k": -1},
    )
    phase = plan.phases[0]
    prepared = PreparedPhase(
        phase.source, None, RequestRoutes.build(phase.requests, 1), SimpleNamespace(audit=audit), (), token="phase"
    )
    assert asyncio.run(runtime.generate(phase, plan, prepared)) == ["generated"]
    assert runtime.metrics[f"perf/timing/rollout_policy_{phase.source}_generate_seconds"] == 2.0
    assert runtime.metrics[f"perf/timing/rollout_policy_{phase.source}_audit_seconds"] == 5.0


def test_policy_sampler_control_uses_named_rpc(monkeypatch):
    from unittest.mock import AsyncMock

    rpc = AsyncMock(return_value={"policy_sampling": True})
    replica = SimpleNamespace(_server_handle=SimpleNamespace(collective_rpc=SimpleNamespace(remote=rpc)))
    runtime = ExecutionContext(None, RolloutPolicyConfig(), None)
    asyncio.run(runtime.set_policy_sampling([replica], True))
    rpc.assert_awaited_once_with("configure_rollout_policy_sampler", args=(True,))
