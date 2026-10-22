# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Business axes, table writers, file logs, and real offline W&B records."""

import json
from unittest.mock import MagicMock

from verl.utils.tracking import Tracking, ValidationGenerationsLogger, _WandbMetricAdapter


def coordinates(outer, attempt, update):
    return {"step/outer": outer, "step/update": attempt, "step/optimizer": update}


def test_wandb_uses_exact_axes_without_explicit_transport_steps():
    backend = MagicMock()
    logger = _WandbMetricAdapter(backend)
    logger.log({**coordinates(1, 4, 3), "optim/actor/loss": 0.2}, 4)
    logger.log({**coordinates(1, 4, 3), "eval/math500/acc_mean": 0.7}, 1)
    backend.define_metric.assert_any_call("optim/actor/loss", step_metric="step/update", step_sync=False)
    backend.define_metric.assert_any_call("eval/math500/acc_mean", step_metric="step/optimizer", step_sync=False)
    assert all("step" not in call.kwargs for call in backend.log.call_args_list)


def test_unknown_counters_use_a_fixed_known_outer_axis():
    logger = _WandbMetricAdapter(MagicMock())
    logger.log({"step/outer": 3, "optim/actor/loss": 0.2}, 3)
    logger.backend.define_metric.assert_any_call("optim/actor/loss", step_metric="step/outer", step_sync=False)
    logger.log({**coordinates(4, 8, 8), "optim/actor/loss": 0.1}, 8)
    assert logger.axes["optim/actor/loss"] == "step/outer"
    assert sum(c.args == ("optim/actor/loss",) for c in logger.backend.define_metric.call_args_list) == 1


def test_validation_table_uses_same_writer_and_explicit_coordinates():
    backend = MagicMock()
    backend.run = object()
    tracking = MagicMock()
    logger = ValidationGenerationsLogger(tracking=tracking)
    logger.coordinates = coordinates(2, 8, 7)
    logger._log_generations_to_wandb([["question", "answer", 1]], 2, backend)
    backend.log.assert_not_called()
    payload = tracking.log.call_args.args[0]
    assert "eval/samples" in payload and "val/generations" not in payload
    assert payload["step/optimizer"] == 7


def test_tensorboard_uses_each_metrics_business_coordinate():
    tracking = Tracking.__new__(Tracking)
    writer = MagicMock()
    tracking.logger = {"tensorboard": writer}
    tracking.log({**coordinates(2, 8, 7), "optim/actor/loss": 0.2, "eval/x/acc_mean": 0.4}, 2)
    assert [c.kwargs["step"] for c in writer.log.call_args_list] == [8, 7]
    tracking._finished = True


def test_real_offline_history_contains_updates_outer_validation_and_table(tmp_path, monkeypatch):
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore

    import wandb

    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("VERL_FILE_LOGGER_PATH", str(tmp_path / "metrics.jsonl"))
    tracking = Tracking("metrics-contract-test", "offline", ["wandb", "file"], {"trainer": {}})
    path = wandb.run.dir
    run_id = wandb.run.id
    tracking.log({**coordinates(0, 0, 0), "eval/x/acc_mean": 0.4}, 0)
    for attempt in range(1, 9):
        outer = (attempt - 1) // 4 + 1
        successful = attempt if attempt < 3 else attempt - 1
        tracking.log({**coordinates(outer, attempt, successful), "optim/actor/loss": 1 / attempt}, attempt)
        if attempt % 4 == 0:
            tracking.log({**coordinates(outer, attempt, successful), "perf/timing/outer_step_seconds": 10.0}, outer)
    tracking.log({**coordinates(2, 8, 7), "eval/x/acc_mean": 0.6}, 2)
    table_logger = ValidationGenerationsLogger(tracking=tracking)
    table_logger.log(["wandb"], [["q", "a", 1]], 2, coordinates=coordinates(2, 8, 7))
    tracking.finish()
    tracking.finish()
    from pathlib import Path

    reader = DataStore()
    reader.open_for_scan(str(Path(path).parent / f"run-{run_id}.wandb"))
    rows = []
    while True:
        data = reader.scan_data()
        if data is None:
            break
        record = wandb_internal_pb2.Record()
        record.ParseFromString(data)
        if record.HasField("history"):
            rows.append(
                {
                    (".".join(x.nested_key) if x.nested_key else x.key): json.loads(x.value_json)
                    for x in record.history.item
                }
            )
    reader.close()
    assert len(rows) == 13
    train = [r for r in rows if "optim/actor/loss" in r]
    assert [r["step/update"] for r in train] == list(range(1, 9))
    assert [r["step/optimizer"] for r in train] == [1, 2, 2, 3, 4, 5, 6, 7]
    assert any(k.startswith("eval/samples") for k in rows[-1])
    assert not any(k.startswith("val/generations") for r in rows for k in r)
    files = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert len(files) == 12  # the table is routed only to W&B


def test_console_summary_does_not_filter_recorded_metrics():
    tracking = object.__new__(Tracking)
    console, file = MagicMock(), MagicMock()
    tracking.logger = {"console": console, "file": file}
    update = {
        "step/outer": 1,
        "step/update": 2,
        "optim/actor/loss": 3,
        "policy/current_behavior/ratio_p99": 4,
        "rollout/data_reuse/update_index": 2,
    }
    tracking.log(update, 2)
    console.log.assert_not_called()
    assert file.log.call_args.kwargs["data"] == update
    outer = {"step/outer": 1, "perf/timing/outer_step_seconds": 5, "rollout/staleness/trajectory_spans/max": 1}
    tracking.log(outer, 1)
    assert console.log.call_args.kwargs["data"]["optim/actor/loss"] == 3
    assert file.log.call_args.kwargs["data"] == outer


def test_sample_table_survives_empty_and_variable_completion_counts():
    from types import SimpleNamespace

    import wandb

    logger = ValidationGenerationsLogger(project_name="test", experiment_name="test", tracking=MagicMock())
    logger.coordinates = {"step/outer": 0}
    backend = SimpleNamespace(Table=wandb.Table, run=object())
    logger._log_generations_to_wandb([], 0, backend)
    logger._log_generations_to_wandb([["q1", "a1", 1], ["q2", "a2", 0]], 1, backend)
    logger._log_generations_to_wandb([["q3", "a3", 1]], 2, backend)
    table = logger.validation_table
    assert len(table.columns) == 7
    assert len(table.data) == 3
    assert table.data[0] == [0, None, None, None, None, None, None]
    assert table.data[2] == [2, "q3", "a3", 1, None, None, None]
