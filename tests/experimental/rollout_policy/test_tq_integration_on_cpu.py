# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Real Ray/TQ session publication and failure drain; model execution is a delayed stub."""

import asyncio
import tempfile
import time
import uuid

import pytest
import ray
import torch
import transfer_queue as tq
from omegaconf import OmegaConf

from verl.experimental.rollout_policy.collector import collect_plan
from verl.experimental.rollout_policy.plan import RolloutPolicyConfig, build_plan
from verl.experimental.rollout_policy.runtime import TQPublisher
from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopWorkerTQ


@ray.remote(num_cpus=1)
class DelayedSessionWorker:
    generate_sessions = AgentLoopWorkerTQ.__ray_metadata__.modified_class.generate_sessions
    drain_sessions = AgentLoopWorkerTQ.__ray_metadata__.modified_class.drain_sessions

    def __init__(self, fail, slow=False):
        tq.init()
        self.fail = fail
        self.slow = slow
        self.config = OmegaConf.create(
            {"actor_rollout_ref": {"rollout": {"agent": {"default_agent_loop": "single_turn_agent"}}}}
        )

    async def _run_agent_loop(self, params, trajectory, **prompt):
        source = prompt["__rollout_policy__"]["source"]
        # Fail one response while siblings are still waiting to publish.
        if self.fail == "score" and source == "teacher" and params["fail_this"]:
            await asyncio.sleep(0.01)
            raise ValueError("injected second-source scoring failure")
        await asyncio.sleep(0.2 if self.slow else 0.03)
        await tq.async_kv_put(
            key=f"{prompt['uid']}_{prompt['session_id']}_0",
            partition_id="train",
            fields={"responses": torch.tensor([7, 0])},
            tag={"status": "success", "source": source},
        )


@pytest.fixture(scope="module")
def tq_runtime():
    assert not ray.is_initialized()
    with tempfile.TemporaryDirectory(prefix="fusion-tq-") as root:
        ray.init(
            address="local",
            namespace="fusion-tq-" + uuid.uuid4().hex,
            num_cpus=6,
            num_gpus=0,
            include_dashboard=False,
            _temp_dir=root,
            object_store_memory=128 * 1024**2,
            logging_level="ERROR",
        )
        tq.init(OmegaConf.create({"backend": {"SimpleStorage": {"num_data_storage_units": 1}}}))
        try:
            yield
        finally:
            tq.close()
            ray.shutdown()


@pytest.mark.parametrize("fail", [None, "score", "cancel"])
def test_real_tq_cross_source_sessions_and_delayed_failure(tq_runtime, tmp_path, fail):
    uids = [uuid.uuid4().hex for _ in range(4)]
    plan = build_plan(
        RolloutPolicyConfig.parse({"mode": "trajectory_mixture"}),
        uids=uids,
        stable_ids=list(range(4)),
        n=3,
        collection_step=1,
        optimizer_step=0,
        seed=42,
        student_path=str(tmp_path),
        teacher_path=str(tmp_path),
    )
    assert any(len({request.source for request in plan.requests if request.uid == uid}) == 2 for uid in uids)
    tq.kv_batch_put(keys=uids, partition_id="train", tags=[{"is_prompt": True, "status": "pending"} for _ in uids])
    workers = [DelayedSessionWorker.remote(fail, i == 0) for i in range(2)]

    class Runtime:
        async def prepare(self, phase, plan):
            meta = (await tq.async_kv_list("train"))["train"]
            assert all(meta[uid]["status"] == "running" for uid in uids)

        async def generate(self, phase, plan):
            sessions = [
                {
                    "request": request,
                    "prompt": {"uid": request.uid, "global_steps": 1},
                    "sampling": {"fail_this": i == 0},
                    "metadata": {"source": request.source},
                }
                for i, request in enumerate(phase.requests)
            ]
            refs = [
                worker.generate_sessions.remote(sessions[i::2], None, 30.0, 5.0) for i, worker in enumerate(workers)
            ]
            if fail == "cancel" and phase.decoder == "teacher":
                await asyncio.sleep(0.02)
                ray.cancel(refs[0])
            results = await asyncio.gather(*refs, return_exceptions=True)
            errors = [value for value in results if isinstance(value, BaseException)]
            if errors:
                raise errors[0]
            return [value for group in results for value in group]

        async def release(self, phase):
            pass

        async def drain(self):
            await asyncio.gather(*(worker.drain_sessions.remote(5.0) for worker in workers))

        async def release_all(self):
            pass

    try:
        if fail:
            with pytest.raises((ray.exceptions.RayTaskError, ray.exceptions.TaskCancelledError)):
                asyncio.run(collect_plan(plan, Runtime(), TQPublisher()))
        else:
            asyncio.run(collect_plan(plan, Runtime(), TQPublisher()))
        before = tq.kv_list("train").get("train", {})
        relevant = {key: value for key, value in before.items() if any(key.startswith(uid) for uid in uids)}
        if fail:
            assert relevant == {}
        else:
            assert len(relevant) == 16
            assert all(relevant[uid]["status"] == "finished" for uid in uids)
        time.sleep(0.25)
        assert tq.kv_list("train").get("train", {}) == before, "A delayed writer survived collection"
    finally:
        for worker in workers:
            ray.kill(worker)
        tq.kv_clear(partition_id="train", keys=uids + [f"{r.uid}_{r.response_index}_0" for r in plan.requests])
