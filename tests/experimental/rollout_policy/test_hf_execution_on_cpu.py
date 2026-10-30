# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Fixed HF pairs acquire the device without exporting learner weights."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from omegaconf import OmegaConf

from verl.experimental.rollout_policy.lifecycle import ExecutionContext
from verl.experimental.rollout_policy.plan import ModelBinding, RolloutPolicyConfig, build_plan
from verl.workers.rollout.hf_fusion import execution


def test_fixed_pair_parks_actor_before_reference_preparation(monkeypatch):
    events = []

    async def park():
        events.append("park")

    async def configure(settings):
        events.append("configure")
        assert settings["current"] is False

    server = SimpleNamespace(prepare=SimpleNamespace(remote=AsyncMock()))

    class Replica:
        def __init__(self, rank, rollout, model, **kwargs):
            assert rollout.val_kwargs.repetition_detection is None
            self._server_handle = server
            self.init_hybrid = AsyncMock()
            self.sleep = AsyncMock()

    monkeypatch.setattr(execution, "get_rollout_replica_class", lambda name: Replica)
    actor = SimpleNamespace(
        world_size=1,
        fused_worker_used=False,
        park_fusion_actor=park,
        configure_fusion=configure,
        update_weights=AsyncMock(),
        prepare_fusion=AsyncMock(),
        release_fusion=AsyncMock(),
    )
    config = OmegaConf.create(
        {
            "trainer": {"n_gpus_per_node": 1},
            "actor_rollout_ref": {
                "model": {"path": "/learner"},
                "rollout": {
                    "engine_kwargs": {},
                    "val_kwargs": {
                        "repetition_detection": {"min_pattern_size": 2, "max_pattern_size": 8, "min_count": 3}
                    },
                },
            },
        }
    )
    policy = RolloutPolicyConfig.parse({"mode": "token_mixture", "execution": {"backend": "hf_fusion"}})
    plan = build_plan(
        policy,
        bindings={"left": ModelBinding("checkpoint", "/a", "a"), "right": ModelBinding("checkpoint", "/b", "b")},
        uids=["u"],
        stable_ids=[0],
        n=1,
        collection_step=1,
        optimizer_step=0,
        seed=1,
    )
    backend = execution.HFReferenceExecutor(ExecutionContext(config, policy, actor))

    async def run():
        phase = await backend.prepare(plan)
        await phase.release()

    asyncio.run(run())
    assert events == ["park", "configure"]
    assert config.actor_rollout_ref.rollout.val_kwargs.repetition_detection.min_count == 3
    actor.update_weights.assert_not_called()
    actor.release_fusion.assert_awaited_once()
