# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
import asyncio
from types import SimpleNamespace

import pytest

from verl.workers.config.rollout import RolloutConfig
from verl.workers.rollout.base import get_rollout_class
from verl.workers.rollout.hf_fusion.replica import FusionBatchServer
from verl.workers.rollout.replica import get_rollout_replica_class


def test_registered_backend_constructs_with_shared_stop_contract():
    config = RolloutConfig(
        name="hf_fusion",
        tensor_model_parallel_size=1,
        stop_token_ids=[0, 1],
        engine_kwargs={"hf_fusion": {"micro_batch_size": 2}},
    )
    replica = get_rollout_replica_class("hf_fusion")(0, config, SimpleNamespace(path="local"), gpus_per_node=4)
    assert replica.world_size == 1
    assert get_rollout_class("hf_fusion").__name__ == "ServerAdapter"


@pytest.mark.parametrize("fail", [False, True])
def test_microbatch_partial_flush_and_failure_fanout(fail):
    batches = []

    async def remote(rows):
        await asyncio.sleep(0.001)
        batches.append([row["request_id"] for row in rows])
        if fail and len(batches) == 2:
            raise ValueError("injected decode failure")
        return [SimpleNamespace(extra_fields={"fusion_request_id": row["request_id"]}) for row in rows]

    server = FusionBatchServer.__ray_metadata__.modified_class(
        SimpleNamespace(fusion_generate_batch=SimpleNamespace(remote=remote)), 2
    )

    async def run():
        await server.prepare()
        calls = [server.generate(str(i), list(range(6 - i)), {"seed": i}) for i in range(5)]
        results = await asyncio.gather(*calls, return_exceptions=True)
        await server.drain()
        if fail:
            assert sum(isinstance(value, Exception) for value in results) == 3
            with pytest.raises(RuntimeError, match="injected"):
                await server.prepare()
        else:
            assert [value.extra_fields["fusion_request_id"] for value in results] == [str(i) for i in range(5)]
            assert [len(batch) for batch in batches] == [2, 2, 1]
        assert server.task is None and not server.queue

    asyncio.run(run())


def test_planned_admission_waits_for_slow_tokenization_and_flushes_tail():
    batches = []

    async def remote(rows):
        batches.append([row["request_id"] for row in rows])
        return [SimpleNamespace(extra_fields={"fusion_request_id": row["request_id"]}) for row in rows]

    server = FusionBatchServer.__ray_metadata__.modified_class(
        SimpleNamespace(fusion_generate_batch=SimpleNamespace(remote=remote)), 2
    )

    async def run():
        await server.prepare(3)
        first = asyncio.create_task(server.generate("a", [1], {}))
        await asyncio.sleep(0.03)
        assert not batches
        others = [server.generate("b", [1], {}), server.generate("c", [1], {})]
        results = await asyncio.gather(first, *others)
        assert [value.extra_fields["fusion_request_id"] for value in results] == ["a", "b", "c"]
        await server.drain()
        assert [len(batch) for batch in batches] == [2, 1]

    asyncio.run(run())


def test_failed_peer_releases_planned_admission_waiters():
    async def remote(rows):
        raise AssertionError("The incomplete microbatch must not execute")

    server = FusionBatchServer.__ray_metadata__.modified_class(
        SimpleNamespace(fusion_generate_batch=SimpleNamespace(remote=remote)), 2
    )

    async def run():
        await server.prepare(2)
        task = asyncio.create_task(server.generate("a", [1], {}))
        await asyncio.sleep(0.02)
        await server.abort_pending("peer tokenization failed")
        with pytest.raises(RuntimeError, match="peer tokenization failed"):
            await task
        await server.drain()
        assert server.task is None and not server.queue

    asyncio.run(run())


def test_phase_token_rejects_late_and_excess_requests():
    async def remote(rows):
        return [SimpleNamespace(extra_fields={"fusion_request_id": row["request_id"]}) for row in rows]

    server = FusionBatchServer.__ray_metadata__.modified_class(
        SimpleNamespace(fusion_generate_batch=SimpleNamespace(remote=remote)), 2
    )

    async def run():
        await server.prepare(1, "new")
        with pytest.raises(RuntimeError, match="Stale"):
            await server.generate("late", [1], {"_fusion_phase_token": "old"})
        result = await server.generate("ok", [1], {"_fusion_phase_token": "new"})
        assert result.extra_fields["fusion_request_id"] == "ok"
        with pytest.raises(RuntimeError, match="excess"):
            await server.generate("extra", [1], {"_fusion_phase_token": "new"})
        await server.drain()

    asyncio.run(run())


def test_misordered_worker_outputs_fail_the_entire_microbatch():
    async def remote(rows):
        return [SimpleNamespace(extra_fields={"fusion_request_id": row["request_id"]}) for row in reversed(rows)]

    server = FusionBatchServer.__ray_metadata__.modified_class(
        SimpleNamespace(fusion_generate_batch=SimpleNamespace(remote=remote)), 2
    )

    async def run():
        await server.prepare(2)
        results = await asyncio.gather(
            server.generate("a", [1], {}), server.generate("b", [1], {}), return_exceptions=True
        )
        assert all(isinstance(value, RuntimeError) and "identities" in str(value) for value in results)
        await server.drain()

    asyncio.run(run())
