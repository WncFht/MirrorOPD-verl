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
