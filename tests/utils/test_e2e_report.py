# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU checks for report attribution; synthetic inputs are never report evidence."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "examples/mirroropd/docs/infra/profile/e2e-4gpu/scripts"
spec = importlib.util.spec_from_file_location("e2e_collect", SCRIPTS / "collect.py")
collect = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collect)


def write_fixture(root):
    job = root / "baseline"
    (job / "events").mkdir(parents=True)
    (job / "resolved_config.yaml").write_text("trainer: {total_training_steps: 2}\n")

    def event(name, start, end, **fields):
        return dict(
            name=name,
            start_ns=int(start * 1e9),
            end_ns=int(end * 1e9),
            pid=1,
            tid=1,
            host="test",
            clock_domain="test-clock",
            **fields,
        )

    rows = [
        event("timer/step", 10, 20),
        event("timer/gen", 10, 14),
        event("timer/teacher_forward", 14, 17),
        event("timer/update_actor", 16, 19),  # Deliberate overlap, not 10 s of exclusive work.
        event("rollout/request", 10, 13, output_tokens=100, input_tokens=10),
        event("timer/testing", 20, 25),
        event("rollout/request", 21, 24, output_tokens=10, input_tokens=10),
        event("timer/step", 26, 36),
        event("timer/gen", 26, 30),
        event("rollout/request", 26, 29, output_tokens=200, input_tokens=10),
    ]
    (job / "events/events-1.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    (job / "resources.jsonl").write_text(json.dumps({"mono_ns": int(2e9), "gpus": []}) + "\n")
    return job


def test_overlap_cycle_and_validation_are_not_misattributed(tmp_path):
    result = collect.collect_job(write_fixture(tmp_path))
    first, last = result["steps"]
    assert first["seconds"] == 10
    assert first["phase_overlap_seconds"] == 1
    assert first["unattributed_seconds"] == 1
    assert first["start_to_next_start_seconds"] == 16  # Includes validation and gap.
    assert last["start_to_next_start_seconds"] is None
    assert result["coverage"]["step_requests"] == {1: 1, 2: 1}
    assert result["coverage"]["validation_requests"] == 1
    assert [x["tokens"] for x in result["request_summary"]] == [100, 200]
    assert result["origin_ns"] == 2e9  # Resource sampling precedes first event.
    assert result["status"]["state"] == "not_started"  # No launch record in fixture.


def test_reject_cross_clock_domains(tmp_path):
    job = write_fixture(tmp_path)
    file = job / "events/events-1.jsonl"
    rows = collect.jsonl(file)
    rows[-1]["clock_domain"] = "another-clock"
    file.write_text("".join(json.dumps(x) + "\n" for x in rows))
    with pytest.raises(AssertionError, match="Cross-clock-domain"):
        collect.collect_job(job)


def test_failed_requests_do_not_become_zero_token_completions(tmp_path):
    job = write_fixture(tmp_path)
    file = job / "events/events-1.jsonl"
    failed = collect.jsonl(file)[4] | {"status": "error", "output_tokens": None}
    with file.open("a") as stream:
        stream.write(json.dumps(failed) + "\n")
    result = collect.collect_job(job)
    assert result["coverage"]["observed_requests"] == 4
    assert result["coverage"]["requests"] == 3
    assert result["coverage"]["failed_requests"] == 1
    assert result["request_summary"][0]["count"] == 1
    assert result["request_summary"][0]["tokens"] == 100


def test_partial_tail_allowed_but_complete_corruption_fails(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text('{"ok": 1}\n{"partial":')
    assert collect.jsonl(path) == [{"ok": 1}]
    path.write_text('{"ok": 1}\nbroken\n')
    with pytest.raises(ValueError, match="Invalid complete"):
        collect.jsonl(path)


def test_empty_measurement_does_not_become_zero_duration(tmp_path):
    (tmp_path / "resolved_config.yaml").write_text("{}")
    result = collect.collect_job(tmp_path)
    assert result["steps"] == []
    assert result["request_summary"] == []
    assert result["coverage"]["events"] == 0
    assert collect.quantile([], 0.5) is None


def test_union_and_interpolated_quantile():
    assert collect.union_seconds([(0, 4e9), (2e9, 3e9), (3e9, 6e9), (8e9, 9e9)]) == 7
    assert collect.quantile([0, 10], 0.9) == 9
