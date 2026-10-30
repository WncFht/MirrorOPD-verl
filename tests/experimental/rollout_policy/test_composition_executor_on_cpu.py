# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Exercise slot mapping at the actual native executor/RPC boundary."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from omegaconf import OmegaConf

from verl.experimental.rollout_policy.execution import NativeVLLMExecutor
from verl.experimental.rollout_policy.plan import RolloutPolicyConfig, build_plan
from verl.experimental.rollout_policy.runtime import RolloutRuntime


@pytest.mark.parametrize("current_side", [None, "left", "right"])
def test_native_loading_and_refit_target_are_independent_of_learner_initialization(monkeypatch, tmp_path, current_side):
    from verl.workers import config as model_config
    from verl.workers.rollout.vllm_fusion import compat, replica

    paths = {}
    for name in ("learner", "A", "B"):
        path = tmp_path / name
        path.mkdir()
        paths[name] = str(path)
    references = {"left": {"kind": "checkpoint", "ref": paths["A"]}, "right": {"kind": "checkpoint", "ref": paths["B"]}}
    if current_side:
        references[current_side] = {"kind": "current"}
    policy = RolloutPolicyConfig.parse({"mode": "token_mixture", "right_weight": 0.25, **references})
    plan = build_plan(
        policy,
        uids=["u"],
        stable_ids=[0],
        n=1,
        collection_step=3,
        optimizer_step=2,
        seed=42,
        student_path=paths["learner"],
    )
    server = SimpleNamespace(
        collective_rpc=SimpleNamespace(remote=AsyncMock()),
        prepare=SimpleNamespace(remote=AsyncMock()),
        set_global_steps=SimpleNamespace(remote=AsyncMock()),
    )
    replicas = []

    class Replica:
        def __init__(self, rank, rollout, model, **kwargs):
            self.model = model
            self.rollout = rollout
            self._server_handle = server
            self.launch_servers = AsyncMock()
            self.sleep = AsyncMock()
            self.wake_up = AsyncMock()
            replicas.append(self)

    monkeypatch.setattr(replica, "VLLMFusionReplica", Replica)
    monkeypatch.setattr(model_config, "HFModelConfig", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(compat, "fusion_hf_overrides", lambda path, space: {"second_path": path})
    wg = SimpleNamespace(
        world_size=1,
        workers=[object()],
        park_fusion_actor=AsyncMock(),
        configure_fusion=AsyncMock(),
        update_weights=AsyncMock(),
    )
    config = OmegaConf.create(
        {
            "trainer": {"n_gpus_per_node": 1},
            "actor_rollout_ref": {
                "model": {"path": paths["learner"]},
                "rollout": {"n": 1, "max_model_len": 64, "engine_kwargs": {"vllm": {}}},
            },
        }
    )
    runtime = RolloutRuntime(SimpleNamespace(config=config, rollout_policy=policy, actor_rollout_wg=wg))
    executor = NativeVLLMExecutor(runtime)

    async def run():
        await executor.prepare(plan)
        await executor.release()
        await executor.prepare(plan)
        await executor.release()

    asyncio.run(run())
    first, second, weight = plan.fusion_components
    assert len(replicas) == 1
    assert replicas[0].model.path == first.path
    assert replicas[0].rollout.engine_kwargs.vllm.hf_overrides.second_path == second.path
    assert server.prepare.remote.call_args.args[2] == weight
    assert replicas[0].wake_up.await_count == 1
    assert replicas[0].sleep.await_count == 2
    if current_side:
        wg.update_weights.assert_awaited_once_with(global_steps=3, target="fusion_student", version=(2, 2))
        assert first.path == paths["learner"]
    else:
        wg.update_weights.assert_not_called()
        assert (first.path, second.path) == (paths["A"], paths["B"])
