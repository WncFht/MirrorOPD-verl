# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Exercise telemetry at the real submission, serialization and worker boundaries."""

import asyncio
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch
from omegaconf import OmegaConf

import verl.trainer.ppo.v1.agent_loop_tq as tq_loop
import verl.trainer.ppo.v1.trainer_base as trainer_module
import verl.utils.pipeline_metrics as pm
import verl.workers.engine_workers as engine_module
from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput, AgentLoopWorker
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.utils import tensordict_utils as tu
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import append_to_dict
from verl.workers.engine_workers import TrainingWorker
from verl.workers.rollout.replica import TokenOutput


def _requests():
    submission = {"id": "batch", "expected_requests": 3, "clock_domain": "node:time", "started_at_ns": 10**9}
    # All requests took 2s, but staggered admission makes the batch take 6s.
    return [
        {
            "rollout_submission": deepcopy(submission),
            "generation_finished_at_ns": finish * 10**9,
            "generation_clock_domain": "node:time",
            "generate_sequences": 2.0,
            "teacher_request_latency_s": latency,
            "num_preempted": preemptions,
        }
        for finish, latency, preemptions in [(3, 1, 0), (7, 2, 2), (5, 9, 1)]
    ]


def test_staggered_batch_and_request_weighted_mean():
    metrics = pm.aggregate_pipeline_requests(["a_0_0", "a_1_0", "b_0_0"], _requests())
    assert metrics == {
        "timing_s/rollout_batch": 6,
        "teacher/request_latency_s/mean": 4,
        "rollout/preemptions": 3,
    }


@pytest.mark.parametrize("problem", ["missing", "duplicate", "multi_turn", "mixed_submission", "no_telemetry"])
def test_incomplete_submissions_never_report_full_batch(problem):
    keys, rows = ["a_0_0", "a_1_0", "b_0_0"], _requests()
    if problem == "missing":
        keys.pop()
        rows.pop()  # Failed/cancelled/filtered sessions never arrive in TQ.
    elif problem == "duplicate":
        keys[1] = keys[0]
    elif problem == "multi_turn":
        keys[1] = "a_0_1"
    elif problem == "mixed_submission":
        rows[1]["rollout_submission"]["id"] = "another"
    else:
        rows[1] = {}
    metrics = pm.aggregate_pipeline_requests(keys, rows)
    assert "timing_s/rollout_batch" not in metrics
    assert "rollout/preemptions" not in metrics


@pytest.mark.parametrize("problem", ["other_node", "other_namespace", "no_clock", "no_finish", "negative_span"])
def test_unverified_clocks_omit_span_but_preserve_counts(problem):
    rows = _requests()
    if problem in ("other_node", "other_namespace", "no_clock"):
        rows[1]["generation_clock_domain"] = {
            "other_node": "other:time",
            "other_namespace": "node:other",
            "no_clock": None,
        }[problem]
    else:
        rows[1]["generation_finished_at_ns"] = None if problem == "no_finish" else 0
    metrics = pm.aggregate_pipeline_requests(["a_0_0", "a_1_0", "b_0_0"], rows)
    assert "timing_s/rollout_batch" not in metrics
    assert metrics["rollout/preemptions"] == 3


@pytest.mark.parametrize("unknown", [-1, None])
def test_unknown_preemptions_are_not_zero_and_warn_once(caplog, unknown):
    pm._warn_once.cache_clear()
    rows = _requests()
    rows[1]["num_preempted"] = unknown
    for _ in range(2):
        metrics = pm.aggregate_pipeline_requests(["a_0_0", "a_1_0", "b_0_0"], rows)
        assert "rollout/preemptions" not in metrics
        assert metrics["timing_s/rollout_batch"] == 6
    assert sum("unknown is not zero" in record.message for record in caplog.records) == 1


def test_validation_sums_windows_and_omits_any_partially_missing_metric():
    windows = [{"timing_s/rollout_batch": 6, "rollout/preemptions": 0}, {"timing_s/rollout_batch": 2}]
    assert pm.aggregate_validation_windows(windows) == {"validation/timing_s/rollout_batch": 8}
    assert pm.aggregate_validation_windows(windows + [{}]) == {}


def _trainer():
    trainer = object.__new__(PPOTrainerSync)
    trainer.config = OmegaConf.create(
        {
            "trainer": {"v1": {}},
            "actor_rollout_ref": {"rollout": {"n": 2, "val_kwargs": {"n": 3}}},
        }
    )
    trainer.trainer_mode = "sync"
    trainer.global_steps = 5
    trainer.agent_loop_manager = MagicMock()
    return trainer


def test_submission_and_collection_work_by_default_without_distillation(monkeypatch):
    trainer = _trainer()
    put, get = MagicMock(), MagicMock()
    monkeypatch.setattr(trainer_module.tq, "kv_batch_put", put)
    monkeypatch.setattr(trainer_module.tq, "kv_batch_get", get)
    monkeypatch.setattr(pm, "monotonic_clock_domain", lambda: "node:time")
    batch = tu.get_tensordict({"uid": ["a", "b"], "__rollout_n__": [2, 1]})
    assert trainer._submit_batch_to_rollout(batch) == 2
    sent = trainer.agent_loop_manager.generate_sequences.call_args.args[0]
    assert sent["__rollout_submission__"]["expected_requests"] == 3
    assert sent["__rollout_submission__"]["clock_domain"] == "node:time"
    rows = _requests()
    for row in rows:
        row.pop("teacher_request_latency_s")
    get.return_value = tu.list_of_dict_to_tensordict([{"metrics": row} for row in rows])
    meta = SimpleNamespace(
        keys=["a_0_0", "a_1_0", "b_0_0", "padded_copy"],
        tags=[{}, {}, {}, {"is_padding": True}],
        partition_id="train",
    )
    metrics = trainer._collect_pipeline_metrics(meta)
    assert get.call_args.kwargs["keys"] == meta.keys[:3]
    assert metrics["timing_s/rollout_batch"] == 6
    assert metrics["rollout/preemptions"] == 3
    assert "teacher/request_latency_s/mean" not in metrics


@pytest.mark.parametrize(
    "validate,mode,filtering", [(False, "separate_async", False), (False, "sync", True), (True, "separate_async", True)]
)
def test_unsupported_training_modes_skip_batch_stamp_but_validation_can_measure(validate, mode, filtering):
    trainer = _trainer()
    trainer.trainer_mode = mode
    trainer.config.algorithm = {"filter_groups": {"enable": filtering}}
    batch = tu.get_tensordict({"uid": ["a", "b"]})
    trainer._stamp_rollout_submission(batch, validate=validate)
    assert ("__rollout_submission__" in batch) == validate
    if validate:
        assert batch["__rollout_submission__"]["expected_requests"] == 6


@pytest.mark.parametrize("submitted", [False, True])
def test_completion_stamp_precedes_merge_and_survives_tq(monkeypatch, submitted):
    import verl.experimental.agent_loop.single_turn_agent_loop as loop_module

    loop = object.__new__(SingleTurnAgentLoop)
    loop.response_length = 3
    loop.rollout_config = SimpleNamespace(full_determinism=False)
    loop.process_multi_modal_info = AsyncMock(return_value={})
    loop._get_mm_processor_kwargs = lambda _: None
    loop._assert_mm_supported = lambda _: None
    loop.ct_build_initial_tokens = AsyncMock(return_value=[101])
    loop.server_manager = SimpleNamespace(generate=AsyncMock(return_value=TokenOutput(token_ids=[11])))
    clock = MagicMock(return_value=7 * 10**9)
    monkeypatch.setattr(loop_module.time, "monotonic_ns", clock)
    monkeypatch.setattr(loop_module, "monotonic_clock_domain", lambda: "node:time")

    async def merge(*args, **kwargs):
        assert clock.call_count == int(submitted)
        return SimpleNamespace(token_ids=[101, 11]), [1], None

    loop.ct_merge_assistant_token = merge
    output = asyncio.run(
        loop.run(
            {}, raw_prompt=[], **({"__rollout_submission__": _requests()[0]["rollout_submission"]} if submitted else {})
        )
    )
    assert ("generation_finished_at_ns" in output.as_dict()["metrics"]) == submitted

    captured = {}

    async def put(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(tq_loop.tq, "async_kv_batch_put", put)
    worker = SimpleNamespace(
        _compute_score=AsyncMock(),
        _compute_teacher_logprobs=AsyncMock(),
        _compute_multi_modal_inputs=lambda *args: {},
        _compute_position_ids=lambda ids, mask, mm: torch.arange(ids.shape[-1]).unsqueeze(0),
    )
    cls = tq_loop.AgentLoopWorkerTQ.__ray_metadata__.modified_class
    asyncio.run(cls._agent_loop_postprocess(worker, [output], uid="a", session_id=0, validate=False, global_steps=5))
    row = tu.get(captured["fields"], "metrics")[0]
    assert ("generation_finished_at_ns" in row) == submitted
    if submitted:
        assert row["generation_finished_at_ns"] == 7 * 10**9
        assert row["rollout_submission"]["expected_requests"] == 3


@pytest.mark.parametrize("has_teacher,validate", [(False, False), (True, False), (True, True)])
def test_teacher_rpc_latency_and_validation_omission(monkeypatch, has_teacher, validate):
    import verl.experimental.agent_loop.agent_loop as loop_module

    worker = SimpleNamespace(
        distillation_enabled=has_teacher,
        teacher_key="teacher",
        teacher_server_manager=SimpleNamespace(compute_teacher_logprobs_single=AsyncMock(return_value=([1], [-0.1]))),
    )
    output = AgentLoopOutput(prompt_ids=[1], response_ids=[2], response_mask=[1], metrics=AgentLoopMetrics())
    clock = MagicMock(side_effect=[10.0, 12.5])
    monkeypatch.setattr(loop_module, "time", SimpleNamespace(monotonic=clock))
    asyncio.run(AgentLoopWorker._compute_teacher_logprobs(worker, output, [1], [2], validate, sample_kwargs={}))
    assert clock.call_count == (2 if has_teacher and not validate else 0)
    assert output.metrics.teacher_request_latency_s == (2.5 if has_teacher and not validate else None)
    assert worker.teacher_server_manager.compute_teacher_logprobs_single.await_count == int(
        has_teacher and not validate
    )


@pytest.mark.parametrize("strategy,source_rank", [("megatron", True), ("fsdp2", True), ("fsdp2", False)])
def test_fsdp_peak_reduces_all_ranks_once_per_update(monkeypatch, strategy, source_rank):
    is_fsdp = strategy == "fsdp2"
    worker = SimpleNamespace(
        engine=SimpleNamespace(
            train_mode=lambda **kw: nullcontext(),
            get_data_parallel_size=lambda: 1,
            get_data_parallel_rank=lambda: 0,
            is_mp_src_rank_with_outputs=lambda: source_rank,
        ),
        engine_config=SimpleNamespace(strategy=strategy),
        device_name="cpu",
        profiler=MagicMock(),
        train_batch=MagicMock(side_effect=lambda data: tu.get_tensordict({}, {"metrics": {"loss": 1.0}})),
    )
    device = MagicMock()
    device.max_memory_allocated.return_value = 3 * 1024**3
    monkeypatch.setattr(engine_module, "get_torch_device", lambda: device)
    reduce = MagicMock(side_effect=lambda tensor, **kw: tensor.fill_(9))  # A remote/SP rank has the largest peak.
    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    data = tu.get_tensordict({"index": torch.tensor([0, 1, 2, 3])}, {"mini_batch_size": 2, "epochs": 2})
    # register() accepts a TensorDict directly; only distributed/engine execution is mocked.
    output = TrainingWorker.train_mini_batch(worker, data)
    assert worker.train_batch.call_count == 4
    assert reduce.call_count == int(is_fsdp)
    if is_fsdp:
        assert reduce.call_args.kwargs["op"] == torch.distributed.ReduceOp.MAX
    device.reset_peak_memory_stats.assert_not_called()
    if source_rank:
        metrics = tu.get(output, "metrics")
        assert (pm.FSDP_PEAK_MEMORY in metrics) == is_fsdp
        if is_fsdp:
            # Exercise the same metadata merging primitive used by worker result collection.
            merged = {}
            append_to_dict(merged, metrics)
            append_to_dict(merged, metrics)
            assert reduce_metrics(merged)[pm.FSDP_PEAK_MEMORY] == 9
    else:
        assert output is None


def test_driver_logs_training_and_validation_in_one_offline_wandb_step(monkeypatch, tmp_path):
    """Use the actual fit/logger path and inspect the offline history, without cloud access."""
    import json

    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore

    import wandb

    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_SILENT", "true")
    trainer = _trainer()
    trainer.config.trainer.update(
        {
            "project_name": "pipeline-metrics-test",
            "experiment_name": "cpu",
            "logger": ["console", "wandb"],
            "val_before_train": False,
            "total_epochs": 1,
            "save_freq": -1,
            "test_freq": 1,
        }
    )
    trainer.config.global_profiler = {"steps": None}
    trainer.global_steps = 0
    trainer.steps_per_epoch = trainer.total_training_steps = 1
    for method in (
        "_reissue_inflight_prompts",
        "on_train_begin",
        "on_step_begin",
        "_start_profiling",
        "_stop_profiling",
        "on_step_end",
        "on_validate_begin",
        "on_validate_end",
        "_shutdown_dump_executor",
    ):
        setattr(trainer, method, MagicMock())
    trainer._consume_sync_metrics = lambda: {}
    batch = SimpleNamespace(keys=["a_0_0", "a_1_0", "b_0_0"], tags=[{}, {}, {}], partition_id="train")

    def step(metrics, timing):
        metrics[pm.FSDP_PEAK_MEMORY] = 9.0
        return batch

    trainer.step = step
    validation = {"validation/timing_s/rollout_batch": 8.0}
    trainer._validate = lambda: validation
    # Avoid running unrelated token/loss diagnostics; retain actual TQ collection and fit ordering.
    trainer._compute_metrics = lambda batch, metrics, *args, **kwargs: metrics.update(
        trainer._collect_pipeline_metrics(batch)
    )
    cleared = []

    def get(**kwargs):
        assert not cleared, "Telemetry must be read before TQ cleanup"
        return tu.list_of_dict_to_tensordict([{"metrics": row} for row in _requests()])

    monkeypatch.setattr(trainer_module.tq, "kv_batch_get", get)
    monkeypatch.setattr(trainer_module.tq, "kv_clear", lambda **kw: cleared.append(True))
    monkeypatch.setattr(trainer_module, "SkipManager", MagicMock())
    try:
        trainer.fit(MagicMock())
        assert cleared == [True]
    finally:
        wandb.finish()
    histories = []
    for path in tmp_path.glob("wandb/offline-run-*/run-*.wandb"):
        store = DataStore()
        store.open_for_scan(str(path))
        try:
            while (raw := store.scan_data()) is not None:
                record = wandb_internal_pb2.Record()
                record.ParseFromString(raw)
                if record.HasField("history"):
                    histories.append(
                        {
                            item.key or "/".join(item.nested_key): json.loads(item.value_json)
                            for item in record.history.item
                        }
                    )
        finally:
            store.close()
    assert len(histories) == 1
    assert histories[0]["_step"] == 0  # transport sequence, not the outer step
    assert histories[0]["step/outer"] == 1
    assert histories[0]["perf/timing/rollout_batch_seconds"] == 6
    assert histories[0]["perf/memory/fsdp_allocated_peak_gib"] == 9
    assert histories[0]["perf/timing/validation_rollout_seconds"] == 8
    assert "teacher/request_latency_s/mean" not in histories[0]
    assert histories[0]["rollout/preemptions"] == 3
    assert histories[0]["perf/teacher/request_latency_mean_seconds"] == 4
    assert "timing_s/rollout_batch" not in histories[0]


@pytest.mark.parametrize("has_peak", [False, True])
def test_actor_preserves_canonical_memory_key_without_metrics_metadata(has_peak):
    from verl.utils.metric import Metric

    trainer = _trainer()
    trainer.config.actor_rollout_ref.actor = {
        "ppo_mini_batch_size": 2,
        "calculate_entropy": False,
        "entropy_coeff": 0,
        "policy_loss": {},
        "ppo_epochs": 1,
        "data_loader_seed": 42,
        "shuffle": False,
    }
    trainer.config.actor_rollout_ref.rollout.temperature = 1.0
    returned = {"mfu": [0.1], "perf/max_memory_allocated_gb": [3, 9]}
    if has_peak:
        returned[pm.FSDP_PEAK_MEMORY] = Metric("max", 3)
        returned[pm.FSDP_PEAK_MEMORY].append(9)
    trainer.actor_rollout_wg = SimpleNamespace(update_actor=lambda batch: {"metrics": returned})
    batch = SimpleNamespace(extra_info={})
    metrics = {}
    trainer._update_actor(batch, metrics)
    assert "opd_metrics" not in batch.extra_info
    assert metrics["perf/actor/mfu"] == 0.1
    assert metrics["perf/actor/allocated_peak_gib"] == 9
    assert ("perf/memory/fsdp_allocated_peak_gib" in metrics) == has_peak
    assert f"actor/{pm.FSDP_PEAK_MEMORY}" not in metrics
    if has_peak:
        assert metrics["perf/memory/fsdp_allocated_peak_gib"] == 9
