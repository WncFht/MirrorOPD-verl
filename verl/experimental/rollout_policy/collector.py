# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Single owner of parent completion; phase workers publish only session results."""

import asyncio


class UnsafeDrainError(RuntimeError):
    """A writer may still exist: abort the run, never clear keys or switch GPU phase."""


async def settle_with_deadline(awaitables, deadline, drain):
    tasks = [asyncio.ensure_future(value) for value in awaitables]
    if not tasks:
        return []
    try:
        done, pending = await asyncio.wait(tasks, timeout=deadline)
        if pending:
            _, pending = await asyncio.wait(tasks, timeout=drain)
            if pending:
                raise UnsafeDrainError("Session publication did not drain; stop this run")
            raise TimeoutError("Rollout session deadline exceeded")
        results = [task.result() for task in tasks]
        return results
    except UnsafeDrainError:
        # Keep strong references until process cleanup: cancelled awaits cannot prove
        # a TransferQueue RPC stopped writing. The caller must terminate this run.
        for task in tasks:
            task.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
        raise
    except BaseException:
        # Cancellation of an awaiter is not cancellation of an already submitted
        # TQ RPC. Let every session acknowledge publication before cleanup.
        _, pending = await asyncio.wait(tasks, timeout=drain)
        if pending:
            raise UnsafeDrainError("Cancelled sessions did not drain; stop this run") from None
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def collect_plan(plan, runtime, publisher):
    """Runtime.generate must settle *all* remote writers before raising.

    UnsafeDrainError deliberately leaves TQ and GPU ownership intact for process cleanup.
    """
    expected = {request.key for request in plan.requests}
    completed = set()
    assigned_sources = {request.key: request.source for request in plan.requests}
    await publisher.running(plan)
    try:
        for phase in plan.phases:
            await runtime.prepare(phase, plan)
            results = await runtime.generate(phase, plan)
            for key, source in results:
                key = tuple(key)
                if key in completed or key not in expected:
                    raise RuntimeError(f"Unexpected or duplicate rollout session: {key}")
                if key not in {r.key for r in phase.requests}:
                    raise RuntimeError(f"Session returned in the wrong phase: {key}")
                if source != assigned_sources[key]:
                    raise RuntimeError(f"Rollout source changed for {key}")
                completed.add(key)
            await runtime.release(phase)
        if completed != expected:
            raise RuntimeError(f"Missing rollout sessions: {expected - completed}")
        await publisher.finished(plan)
    except UnsafeDrainError:
        raise
    except BaseException:
        await runtime.drain()
        await runtime.release_all()
        await publisher.failed_and_clear(plan)
        raise
