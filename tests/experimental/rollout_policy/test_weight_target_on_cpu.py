# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Exercise the real actor RPC body with explicit receiver commit and rank errors."""

import asyncio
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch.distributed as dist

from verl.workers.engine_workers import ActorRolloutRefWorker
from verl.workers.rollout.weight_target import create_fusion_worker


@pytest.mark.parametrize("rank_error", [None, "another rank rejected weights"])
def test_only_all_rank_success_commits_the_named_target(monkeypatch, rank_error):
    target = SimpleNamespace(receive=AsyncMock(return_value=None), commit=AsyncMock())
    engine = SimpleNamespace(
        get_per_tensor_param=Mock(return_value=(iter([]), None)), is_param_offload_enabled=True, to=Mock()
    )
    worker = SimpleNamespace(
        config=SimpleNamespace(rollout=SimpleNamespace(checkpoint_engine=SimpleNamespace(backend="naive"))),
        actor=SimpleNamespace(engine=engine),
        rollout_weight_targets={"policy_current": target},
        layered_summon=False,
    )
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        dist, "all_gather_object", lambda values, local: values.__setitem__(slice(None), [local, rank_error])
    )
    barrier = Mock()
    monkeypatch.setattr(dist, "barrier", barrier)
    call = inspect.unwrap(ActorRolloutRefWorker.update_weights)
    if rank_error:
        with pytest.raises(RuntimeError, match="at least one rank"):
            asyncio.run(call(worker, target="policy_current", version=(2, 32)))
        target.commit.assert_not_called()
        barrier.assert_not_called()
    else:
        asyncio.run(call(worker, target="policy_current", version=(2, 32)))
        target.commit.assert_awaited_once_with((2, 32))
        barrier.assert_called_once()
    engine.to.assert_called_once_with("cpu", model=True, optimizer=False, grad=False)


def test_fixed_native_pair_has_no_actor_receiver():
    generator, receiver = create_fusion_worker(None, None, {"backend": "vllm_fusion", "current": False})
    assert generator is receiver is None


def test_missing_receiver_fails_before_collective_export():
    worker = SimpleNamespace(rollout_weight_targets={})
    with pytest.raises(ValueError, match="immutable"):
        asyncio.run(
            inspect.unwrap(ActorRolloutRefWorker.update_weights)(worker, target="policy_current", version=(1, 1))
        )
