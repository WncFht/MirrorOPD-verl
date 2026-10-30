# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
import asyncio
from types import SimpleNamespace

from omegaconf import OmegaConf

from verl.experimental.rollout_policy.plan import RolloutPolicyConfig, build_plan
from verl.experimental.rollout_policy.runtime import RolloutRuntime
from verl.utils import tensordict_utils as tu


def test_immutable_sampling_plan_reaches_every_real_session_metadata(tmp_path):
    captured = []

    async def generate(sessions, client, deadline, drain):
        captured.extend(sessions)
        return [(session["request"].key, session["request"].source) for session in sessions]

    worker = SimpleNamespace(generate_sessions=SimpleNamespace(remote=generate))
    policy = RolloutPolicyConfig.parse({"mode": "trajectory_mixture"})
    config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"n": 3}}})
    runtime = RolloutRuntime(
        SimpleNamespace(
            config=config,
            rollout_policy=policy,
            agent_loop_manager=SimpleNamespace(agent_loop_workers=[worker, worker]),
        )
    )
    runtime.batch = tu.get_tensordict({"uid": ["a", "b"], "index": [11, 12]})
    runtime.client = None
    plan = build_plan(
        policy,
        uids=["a", "b"],
        stable_ids=[11, 12],
        n=3,
        collection_step=1,
        optimizer_step=0,
        seed=42,
        student_path=str(tmp_path),
        teacher_path=str(tmp_path),
        sampling={"temperature": 0.0, "top_p": 1.0, "top_k": -1},
    )

    async def run():
        for phase in plan.phases:
            results = await runtime.generate(phase, plan)
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
        actor_rollout_wg=SimpleNamespace(update_weights=sync),
        get_llm_client=lambda: "current-client",
        checkpoint_manager=SimpleNamespace(sleep_replicas=sleep),
    )
    runtime = RolloutRuntime(trainer)
    runtime.fusion_client = AsyncMock(side_effect=AssertionError("endpoint allocated fusion"))
    runtime.frozen_client = AsyncMock(side_effect=AssertionError("endpoint allocated fixed model"))
    plan = build_plan(
        policy, uids=["u"], stable_ids=[0], n=1, collection_step=3, optimizer_step=2, seed=1, student_path=str(tmp_path)
    )

    async def run():
        await runtime.prepare(plan.phases[0], plan)
        assert runtime.client == "current-client"
        await runtime.release(plan.phases[0])

    asyncio.run(run())
    sync.assert_awaited_once_with(global_steps=2)
    runtime.fusion_client.assert_not_called()
    runtime.frozen_client.assert_not_called()


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
        actor_rollout_wg=SimpleNamespace(world_size=1),
    )
    runtime = RolloutRuntime(trainer)
    runtime.frozen_client = AsyncMock(return_value="A-client")
    runtime.fusion_client = AsyncMock(side_effect=AssertionError("endpoint allocated fusion"))
    plan = build_plan(
        policy, uids=["u"], stable_ids=[0], n=1, collection_step=1, optimizer_step=0, seed=1, student_path=str(tmp_path)
    )
    asyncio.run(runtime.prepare(plan.phases[0], plan))
    runtime.frozen_client.assert_awaited_once_with("left", str(extra), 1)
    assert runtime.client == "A-client"
    runtime.fusion_client.assert_not_called()
