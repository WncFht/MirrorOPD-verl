# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Partial setup, explicit routing and ownership survive backend failures."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from verl.experimental.rollout_policy.lifecycle import ExecutionContext, OwnedReplicas, RequestRoutes
from verl.experimental.rollout_policy.plan import RolloutPolicyConfig, RolloutRequest
from verl.experimental.rollout_policy.sessions import LocalPhaseClient


def test_admission_counts_and_dispatch_share_routes_not_priority():
    requests = [RolloutRequest(0, "s", "u", i, i, "fusion", i) for i in range(3)]
    routes = RequestRoutes.build(requests, 2)
    assert routes.counts == (2, 1)
    servers = [
        SimpleNamespace(generate=SimpleNamespace(remote=AsyncMock(return_value=SimpleNamespace(extra_fields={}))))
        for _ in range(2)
    ]
    client = LocalPhaseClient(None, servers)
    params = {"_rollout_replica": routes.replica(requests[1])}
    asyncio.run(client.generate("r", prompt_ids=[1], sampling_params=params, priority=100))
    servers[0].generate.remote.assert_not_called()
    servers[1].generate.remote.assert_awaited_once()
    assert "_rollout_replica" not in servers[1].generate.remote.call_args.kwargs["sampling_params"]
    assert params == {"_rollout_replica": 1}


def test_partial_launch_closes_only_owned_servers_after_every_launch_settles(monkeypatch):
    import ray

    events = []
    monkeypatch.setattr(ray, "kill", lambda server, no_restart: events.append(("kill", server)))
    ctx = ExecutionContext(None, RolloutPolicyConfig(), None)
    replicas = [SimpleNamespace(servers=["first"]), SimpleNamespace(servers=["second"])]
    group = OwnedReplicas(ctx, replicas)

    async def launch(replica):
        if replica.servers == ["first"]:
            await asyncio.sleep(0.01)
            events.append(("ready", "first"))
        else:
            raise RuntimeError("startup failed")

    with pytest.raises(RuntimeError, match="startup failed"):
        asyncio.run(group.launch(launch))
    assert events == [("ready", "first"), ("kill", "first"), ("kill", "second")]
    assert not group.replicas and not group.ready
