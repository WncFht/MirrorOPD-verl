# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""A local dashboard must preserve trajectories across partial writes and crashes."""

import json

import pytest

from examples.mirroropd.tools.trackio_dashboard import HISTORY, OVERVIEW, PROJECT, observations, poll_run


class Storage:
    def __init__(self):
        self.points = {}
        self.fail_overview = False

    def get_log_count(self, project, *, run_id):
        return len(self.points.get((project, run_id), {}))

    def bulk_log(self, *, project, run_id, log_ids, metrics_list, steps, **kwargs):
        if project == OVERVIEW and self.fail_overview:
            raise RuntimeError("simulated second-project commit failure")
        points = self.points.setdefault((project, run_id), {})
        for log_id, metrics, step in zip(log_ids, metrics_list, steps, strict=True):
            points.setdefault(log_id, (step, metrics))


def record(values):
    return json.dumps(dict(step=1, data=values)) + "\n"


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "metrics.jsonl"
    return dict(id="adaptive", name="adaptive", metrics=str(path), run_id="test", overview=True)


def test_coordinates_exact_datasets_and_online_undefined(source):
    raw, summary, _ = observations(
        dict(data={"step/optimizer": 16, "optim/actor/loss": 0.3, "optim/actor/update_applied": 1}), source
    )
    assert raw[1] == summary[1] == 15
    points = observations(
        dict(
            data={
                "step/optimizer": 160,
                "eval/rkl/public_initial/dataset/math/question_mean": 0.2,
                "eval/rkl/public_initial/math/question_mean": 99,
                "eval/rkl/public_initial/dataset/development_math/question_mean": 88,
                "reuse/s_defined": 0,
            }
        ),
        source,
    )
    assert points[1] == (HISTORY, 160, {"heldout/math/public_initial": 0.2, "reuse/s_defined": 0})
    assert not any("controller/S" in p[2] for p in points)
    with pytest.raises(ValueError, match="coordinate"):
        observations(dict(data={"step/outer": 10}), source)


def test_training_diagnostics_preserve_values_and_coordinates(source):
    values = {
        "step/optimizer": 16,
        "optim/actor/loss": 0.3,
        "optim/actor/lr": 1e-6,
        "optim/actor/grad_norm_preclip": 5.2,
        "distill/teacher_rkl": 0.31,
        "distill/teacher_rkl_position/p00_10/mean": 0.4,
    }
    raw, summary, _ = observations(dict(data=values), source)
    assert raw[1] == summary[1] == 15
    assert summary[2]["optim/actor/lr"] == 1e-6
    assert summary[2]["optim/actor/grad_norm_preclip"] == 5.2
    assert summary[2]["distill/teacher_rkl"] == 0.31
    assert "distill/teacher_rkl_position/p00_10/mean" not in summary[2]
    assert "distill/teacher_rkl_position/p00_10/mean" in raw[2]
    with pytest.raises(ValueError, match="Mixed pre-update"):
        observations(dict(data={**values, "reuse/used": 16}), source)

    points = observations(dict(data={"step/optimizer": 16, "perf/timing/outer_step_seconds": 20}), source)
    assert points[1] == (HISTORY, 16, {"perf/timing/outer_step_seconds": 20})


def test_partial_line_append_restart_and_source_mutation(source):
    from pathlib import Path

    path = Path(source["metrics"])
    storage = Storage()
    first = record({"step/optimizer": 1, "optim/actor/loss": 0.4})
    second = record({"step/optimizer": 2, "optim/actor/loss": 0.3})
    path.write_text(first + second[:12])
    cursor, report = poll_run(source, {}, storage)
    assert report["rows"] == 1 and report["partial_tail_bytes"] == 12
    with path.open("a") as stream:
        stream.write(second[12:])
    cursor, _ = poll_run(source, cursor, storage)
    assert storage.get_log_count(PROJECT, run_id="test") == 2
    poll_run(source, json.loads(json.dumps(cursor)), storage)
    assert storage.get_log_count(OVERVIEW, run_id="test") == 2
    path.write_text(first.replace("0.4", "0.9") + second)
    with pytest.raises(ValueError, match="modified"):
        poll_run(source, cursor, storage)


def test_crash_between_project_commits_replays_without_duplicates(source):
    from pathlib import Path

    Path(source["metrics"]).write_text(record({"step/optimizer": 8, "reuse/gap_over_drop": 1.1}))
    storage = Storage()
    storage.fail_overview = True
    with pytest.raises(RuntimeError):
        poll_run(source, {}, storage)
    assert storage.get_log_count(PROJECT, run_id="test") == 1
    storage.fail_overview = False
    cursor, _ = poll_run(source, {}, storage)
    assert all(storage.get_log_count(p, run_id="test") == 1 for p in (PROJECT, OVERVIEW))
    # Rebuild a deleted derived project, even when the source did not change.
    del storage.points[(OVERVIEW, "test")]
    poll_run(source, cursor, storage)
    assert all(storage.get_log_count(p, run_id="test") == 1 for p in (PROJECT, OVERVIEW))


def test_missing_logs_and_malformed_complete_rows_are_distinct(source):
    from pathlib import Path

    storage = Storage()
    _, status = poll_run(source, {}, storage)
    assert status["state"] == "pending"
    Path(source["metrics"]).write_text('{"bad":\n')
    with pytest.raises(ValueError, match="Malformed"):
        poll_run(source, {}, storage)


def test_add_comparison_project_backfills_without_duplicating_history(source):
    from pathlib import Path

    Path(source["metrics"]).write_text(record({"step/optimizer": 1, "optim/actor/loss": 0.4}))
    storage = Storage()
    cursor, _ = poll_run(source, {}, storage)
    source.update(comparison_project="optimizer-lr", comparison_name="run - AdamW - lr=1e-6")
    cursor, _ = poll_run(source, cursor, storage)
    assert storage.get_log_count("optimizer-lr", run_id="test") == 1
    assert storage.get_log_count(PROJECT, run_id="test") == 1
    assert storage.get_log_count(HISTORY, run_id="test") == 1
    poll_run(source, cursor, storage)
    assert storage.get_log_count("optimizer-lr", run_id="test") == 1
