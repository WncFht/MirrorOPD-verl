# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Admission, publication and verification must progress independently and drain safely."""

import asyncio
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import aclosing
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl.trainer.evaluation import generation
from verl.trainer.evaluation.accuracy import AccuracyCollector
from verl.trainer.evaluation.artifacts import GENERATION, ArtifactStore, identity
from verl.trainer.evaluation.config import DatasetSpec, GenerationProtocol


def test_slow_consumer_allows_bounded_refill_and_close_drains():
    async def run():
        started, aborted, settled = [], [], []
        release = asyncio.Event()

        async def worker(i):
            started.append(i)
            if i > 8:
                await release.wait()
            settled.append(i)
            return i

        async def abort():
            aborted.append(True)
            release.set()

        async with aclosing(
            generation.bounded_results(((i,) for i in range(100)), worker, concurrency=2, timeout=2, abort=abort)
        ) as stream:
            await anext(stream)
            await asyncio.sleep(0.05)
            # Admission continues while the consumer does no work, then reaches a fixed bound.
            assert 2 < len(started) <= 7
            count = len(started)
            await asyncio.sleep(0.05)
            assert len(started) == count
        assert aborted and sorted(started) == sorted(settled)

    asyncio.run(run())


def test_empty_and_failed_producers_never_leave_consumer_waiting():
    async def run():
        async def worker(i):
            raise ValueError("failed request")

        async def abort():
            pass

        assert [x async for x in generation.bounded_results([], worker, concurrency=2, timeout=1, abort=abort)] == []
        with pytest.raises(ValueError, match="failed request"):
            async for _ in generation.bounded_results([(0,)], worker, concurrency=2, timeout=1, abort=abort):
                pass

    asyncio.run(asyncio.wait_for(run(), 2))


@pytest.mark.parametrize("cancel_publication", [False, True])
def test_cross_dataset_admission_preserves_protocol_seed_and_coverage(tmp_path, monkeypatch, cancel_publication):
    async def run():
        entered_second = asyncio.Event()
        calls = []

        class Loop:
            def __init__(self, **kwargs):
                pass

            async def ct_build_initial_tokens(self, messages):
                return [1]

            async def run(self, params, *, raw_prompt, priority):
                name = raw_prompt[0]["content"]
                if name == "slow":
                    await asyncio.wait_for(entered_second.wait(), 2)
                if name == "next":
                    entered_second.set()
                calls.append((name, params.copy(), priority))
                return SimpleNamespace(
                    prompt_ids=[1],
                    response_ids=[2],
                    response_mask=[True],
                    extra_fields={
                        "global_steps": 7,
                        "min_global_steps": 7,
                        "max_global_steps": 7,
                        "finish_reason": "stop",
                    },
                )

        monkeypatch.setattr(generation, "CachedSingleTurn", Loop)
        monkeypatch.setattr(generation, "CheckedClient", lambda *args, **kwargs: None)
        engine = generation.NativeGenerator.__new__(generation.NativeGenerator)
        engine.config = OmegaConf.create({"actor_rollout_ref": {"rollout": {}}, "data": {"continuous_token": {}}})

        async def abort():
            entered_second.set()

        engine.servers = [SimpleNamespace(abort_all_requests=SimpleNamespace(remote=abort)) for _ in range(2)]
        engine.input_model = SimpleNamespace(
            tokenizer=SimpleNamespace(decode=lambda *args, **kwargs: "prompt"),
            hf_config=SimpleNamespace(model_type="qwen2"),
        )
        engine.max_model_len, engine.concurrency, engine.chunk_size, engine.timeout = 32, 2, 2, 5
        engine.token_mapping, engine.engine_contract, engine.engine_contract_id = None, {}, "contract"
        protocols = {
            "a": GenerationProtocol(max_new_tokens=8, prompt_limit=8),
            "b": GenerationProtocol(max_new_tokens=16, prompt_limit=8),
        }
        datasets = [
            (
                DatasetSpec("first", "math", "unused", "math", protocol="a"),
                [{"id": x, "prompt": [{"role": "user", "content": x}]} for x in ("slow", "fast")],
            ),
            (
                DatasetSpec("second", "math", "unused", "math", n=2, protocol="b"),
                [{"id": "next", "prompt": [{"role": "user", "content": "next"}]}],
            ),
        ]
        if cancel_publication:
            entered, release = threading.Event(), threading.Event()
            original = ArtifactStore.write_rows

            def delayed_write(self, *args, **kwargs):
                entered.set()
                assert release.wait(5)
                return original(self, *args, **kwargs)

            monkeypatch.setattr(ArtifactStore, "write_rows", delayed_write)
            task = asyncio.create_task(
                engine.generate(
                    tmp_path / "generation",
                    datasets,
                    protocols,
                    producer="current_student",
                    model_identity="model",
                    outer=7,
                    expected_version=7,
                )
            )
            try:
                assert await asyncio.to_thread(entered.wait, 3)
                task.cancel()
                await asyncio.sleep(0.02)
                assert not task.done()
            finally:
                release.set()
                with pytest.raises(asyncio.CancelledError):
                    await task
            artifact = ArtifactStore(tmp_path / "generation")
            published = artifact.manifest()
            assert published["status"] == "active" and len(published["files"]) == 1
            await asyncio.sleep(0.02)
            assert artifact.manifest() == published
            return

        store = await engine.generate(
            tmp_path / "generation",
            datasets,
            protocols,
            producer="current_student",
            model_identity="model",
            outer=7,
            expected_version=7,
        )
        rows = [row for name in store.manifest(complete=True)["files"] for row in store.read_rows(name)]
        assert len(rows) == len({row["id"] for row in rows}) == 4
        assert sorted(priority for _, _, priority in calls) == [0, 1, 2, 3]
        for row in rows:
            protocol = protocols["a" if row["dataset"] == "first" else "b"]
            assert row["seed"] == protocol.sample_seed(row["dataset"], row["source_id"], row["sample_index"])
            assert row["protocol_id"] == protocol.protocol_id
        assert {p["max_tokens"] for _, p, _ in calls} == {8, 16}

    asyncio.run(run())


def test_least_inflight_routes_around_slow_replica_and_releases_on_error(monkeypatch):
    from verl.experimental.rollout_policy.sessions import LocalPhaseClient
    from verl.workers.rollout.router import GlobalRequestLoadBalancer

    async def run():
        release, entered = asyncio.Event(), asyncio.Event()
        routes = {}

        async def transport(self, request_id, **kwargs):
            routes[request_id] = kwargs["sampling_params"]["_rollout_replica"]
            if request_id == "slow":
                entered.set()
                await release.wait()
            if request_id == "error":
                raise ValueError("transport failed")
            return object()

        monkeypatch.setattr(LocalPhaseClient, "generate", transport)
        client = generation.CheckedClient.__new__(generation.CheckedClient)
        client.max_model_len, client.prompt_limit, client.token_mapping = 32, 8, None
        client.router = GlobalRequestLoadBalancer({"0": object(), "1": object()})
        slow = asyncio.create_task(client.generate("slow", prompt_ids=[1], sampling_params={"max_tokens": 8}))
        await entered.wait()
        for key in ("a", "b"):
            await client.generate(key, prompt_ids=[1], sampling_params={"max_tokens": 8})
        with pytest.raises(ValueError, match="transport failed"):
            await client.generate("error", prompt_ids=[1], sampling_params={"max_tokens": 8})
        assert routes["a"] == routes["b"] == routes["error"] != routes["slow"]
        release.set()
        await slow
        assert client.router.get_total_inflight() == 0

    asyncio.run(run())


def test_verifier_consumes_ready_results_without_waiting_for_oldest():
    collector = AccuracyCollector.__new__(AccuracyCollector)
    slow, ready = Future(), Future()
    ready.set_result({"id": "ready", "answer": "answer", "score": 1.0})
    collector.pending = {slow: {"id": "slow"}, ready: {"id": "ready"}}
    collector.timeout, collector.chunk, collector.rows = 0.01, [], []
    collector._consume()
    assert list(collector.pending) == [slow]
    assert collector.rows == [{"id": "ready", "score": 1.0}]


def test_grading_follows_published_shards_without_blocking_writer(tmp_path, monkeypatch):
    from verl.trainer.evaluation import accuracy

    # Threads make verifier ordering controllable; production uses spawned processes.
    monkeypatch.setattr(accuracy, "ProcessPoolExecutor", lambda **kwargs: ThreadPoolExecutor(max_workers=1))
    release = threading.Event()
    entered = threading.Event()

    def verify(row, question, verifier, answer):
        entered.set()
        assert release.wait(5)
        return {**{k: v for k, v in row.items() if k != "seed"}, "answer": answer, "score": 1.0, "status": "ok"}

    monkeypatch.setattr(accuracy, "verify_sample", verify)
    protocol = GenerationProtocol(max_new_tokens=8, prompt_limit=8)
    spec = DatasetSpec("math", "math", "unused", "math", n=3)
    questions = [{"id": "q", "prompt": []}]
    specifications = [(spec, questions)]
    tokenizer = SimpleNamespace(decode=lambda *args, **kwargs: "answer", eos_token=None, pad_token=None)
    collector = AccuracyCollector(tmp_path / "accuracy", specifications, tokenizer, workers=1, pending_limit=1)
    store = ArtifactStore.create(
        tmp_path / "generation",
        kind="generation",
        metadata={"question_set_sha256": identity([("math", 3, "public", questions)])},
    )
    collector.follow_generation(store.root, {"public": protocol})
    try:
        for i in range(3):
            row = {
                "id": str(i),
                "source_id": "q",
                "dataset": "math",
                "domain": "math",
                "sample_index": i,
                "seed": i,
                "finish_reason": "stop",
                "response_tokens": 1,
                "response_ids": [2],
                "prompt_ids": [1],
                "response_mask": [True],
                "thinking_open": False,
                "protocol_id": protocol.protocol_id,
            }
            store.write_rows(f"generations/part-{i}.parquet", [row], schema=GENERATION)
        assert entered.wait(3)
        store.finish()
        assert not collector._reader.done()  # Verifier is blocked; all generation is already published.
        release.set()
        collector.finish(store)
        assert collector.store.manifest(complete=True)["metadata"]["generation_sha256"]
        assert len(collector.rows) == 3
    finally:
        release.set()
        collector.close()


def test_consumer_close_acknowledges_abort_before_returning():
    async def run():
        release, entered = asyncio.Event(), asyncio.Event()
        active = set()

        async def worker(i):
            active.add(i)
            try:
                if i:
                    entered.set()
                    await release.wait()
                return i
            finally:
                active.remove(i)

        async def abort():
            await asyncio.sleep(0.01)
            release.set()

        async with aclosing(
            generation.bounded_results(((i,) for i in range(100)), worker, concurrency=2, timeout=2, abort=abort)
        ) as stream:
            assert await anext(stream) == 0
            await entered.wait()
            assert active
        assert not active and release.is_set()

    asyncio.run(run())


def test_failed_grading_reader_propagates_and_close_handles_unpublished_generation(tmp_path, monkeypatch):
    from verl.trainer.evaluation import accuracy

    monkeypatch.setattr(accuracy, "ProcessPoolExecutor", lambda **kwargs: ThreadPoolExecutor(max_workers=1))
    tokenizer = SimpleNamespace(decode=lambda *args, **kwargs: "", eos_token=None, pad_token=None)
    specifications = [(DatasetSpec("math", "math", "unused", "math"), [{"id": "q", "prompt": []}])]
    collector = AccuracyCollector(tmp_path / "scores", specifications, tokenizer)
    source = ArtifactStore.create(tmp_path / "wrong", kind="generation", metadata={"question_set_sha256": "wrong"})
    collector.follow_generation(source.root, {})
    try:
        with pytest.raises(ValueError, match="different questions"):
            collector._reader.result(timeout=2)
        with pytest.raises(ValueError, match="different questions"):
            collector.check()
    finally:
        collector.close()
    unopened = AccuracyCollector(tmp_path / "unopened-scores", specifications, tokenizer)
    unopened.follow_generation(tmp_path / "never-published", {})
    unopened.close()
    assert unopened._reader.done()


def test_unacknowledged_abort_is_fail_stop():
    from verl.experimental.rollout_policy.collector import UnsafeDrainError

    async def run():
        release = asyncio.Event()
        tasks = []

        async def worker(i):
            tasks.append(asyncio.current_task())
            if i:
                await release.wait()
            return i

        async def abort():
            raise RuntimeError("transport unavailable")

        try:
            with pytest.raises(UnsafeDrainError, match="abort was not acknowledged"):
                async with aclosing(
                    generation.bounded_results(((i,) for i in range(4)), worker, concurrency=2, timeout=2, abort=abort)
                ) as stream:
                    await anext(stream)
        finally:
            release.set()
            await asyncio.gather(*tasks)

    asyncio.run(run())


def test_cancellation_waits_for_owned_disk_publication():
    async def run():
        entered, release, written = threading.Event(), threading.Event(), threading.Event()

        def write():
            entered.set()
            assert release.wait(5)
            written.set()

        task = asyncio.create_task(generation.settled_thread_call(write))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            task.cancel()
            await asyncio.sleep(0.02)
            assert not task.done() and not written.is_set()
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert written.is_set()

    asyncio.run(run())
