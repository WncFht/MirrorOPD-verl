# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
import asyncio

import pytest

from verl.experimental.rollout_policy.collector import collect_plan, settle_with_deadline
from verl.experimental.rollout_policy.plan import RolloutPolicyConfig, build_plan


@pytest.mark.parametrize("failure", [None, "second_source", "score", "duplicate", "missing", "source"])
def test_atomic_parent_completion(tmp_path, failure):
    plan = build_plan(
        RolloutPolicyConfig.parse({"mode": "trajectory_mixture"}),
        uids=["a", "b"],
        stable_ids=[1, 2],
        n=3,
        collection_step=1,
        optimizer_step=0,
        seed=4,
        student_path=str(tmp_path),
        teacher_path=str(tmp_path),
    )
    events = []

    class Runtime:
        async def prepare(self, phase, plan):
            events.append(("prepare", phase.decoder))

        async def generate(self, phase, plan):
            async def response(r):
                await asyncio.sleep(0.002 if r.response_index else 0.001)
                events.append(("write", r.key))
                if failure == "score" and r == phase.requests[0]:
                    raise ValueError("score failed")
                return r.key, r.source

            if failure == "second_source" and phase.decoder == "teacher":
                raise ValueError("teacher failed")
            results = await settle_with_deadline([response(r) for r in phase.requests], 1, 1)
            if failure == "duplicate":
                results.append(results[0])
            if failure == "missing":
                results = results[:-1]
            if failure == "source":
                results[0] = results[0][0], "wrong"
            return results

        async def release(self, phase):
            events.append(("release", phase.decoder))

        async def drain(self):
            events.append(("drain", None))

        async def release_all(self):
            events.append(("release_all", None))

    class Publisher:
        async def running(self, plan):
            events.append(("running", None))

        async def finished(self, plan):
            events.append(("finished", None))

        async def failed_and_clear(self, plan):
            events.append(("clear", None))

    async def run():
        if failure:
            with pytest.raises((ValueError, RuntimeError)):
                await collect_plan(plan, Runtime(), Publisher())
            assert not any(event[0] == "finished" for event in events)
            assert events[-3:] == [("drain", None), ("release_all", None), ("clear", None)]
        else:
            await collect_plan(plan, Runtime(), Publisher())
            assert events[-1][0] == "finished"
            assert sum(e[0] == "write" for e in events) == 6
        before = len(events)
        await asyncio.sleep(0.01)
        assert len(events) == before

    asyncio.run(run())


def test_deadline_drains_publications_before_return():
    events = []

    async def slow():
        try:
            await asyncio.sleep(0.03)
            events.append("write")
        finally:
            events.append("settled")

    async def run():
        with pytest.raises(TimeoutError):
            await settle_with_deadline([slow(), slow()], 0.01, 1)
        assert events == ["write", "settled", "write", "settled"]

    asyncio.run(run())


def test_unsafe_remote_drain_never_releases_or_clears(tmp_path):
    from verl.experimental.rollout_policy.collector import UnsafeDrainError

    plan = build_plan(
        RolloutPolicyConfig.parse({"mode": "trajectory_mixture"}),
        uids=["a", "b"],
        stable_ids=[1, 2],
        n=1,
        collection_step=1,
        optimizer_step=0,
        seed=4,
        student_path=str(tmp_path),
        teacher_path=str(tmp_path),
    )
    events = []

    class Runtime:
        async def prepare(self, phase, plan):
            pass

        async def generate(self, phase, plan):
            raise UnsafeDrainError("remote writer unresponsive")

        async def release(self, phase):
            events.append("release")

        async def drain(self):
            events.append("drain")

        async def release_all(self):
            events.append("release_all")

    class Publisher:
        async def running(self, plan):
            events.append("running")

        async def finished(self, plan):
            events.append("finished")

        async def failed_and_clear(self, plan):
            events.append("clear")

    with pytest.raises(UnsafeDrainError):
        asyncio.run(collect_plan(plan, Runtime(), Publisher()))
    assert events == ["running"]


def test_parent_publication_timeout_is_fail_stop_without_cancelling_writer():
    from verl.experimental.rollout_policy.collector import UnsafeDrainError
    from verl.experimental.rollout_policy.runtime import TQPublisher

    async def run():
        future = asyncio.get_running_loop().create_future()
        publisher = TQPublisher(deadline_seconds=0.001)
        with pytest.raises(UnsafeDrainError, match="Parent publication"):
            await publisher.publish(future)
        assert not future.cancelled()
        future.set_result(None)
        await asyncio.sleep(0)

    asyncio.run(run())
