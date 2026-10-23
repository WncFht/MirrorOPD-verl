# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Check clock alignment and coverage using explicitly synthetic CPU fixtures."""

import gzip
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "examples/mirroropd/docs/infra/profile/e2e-4gpu/scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("cycle_review", SCRIPTS / "analyze_cycle.py")
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


def test_aligned_interval_unions_and_missing_capture(tmp_path):
    trace = {
        "baseTimeNanoseconds": int(1000e9),
        "traceEvents": [
            {"cat": "Trace", "ts": 0, "dur": 3e6},
            {
                "cat": "user_annotation",
                "name": "full_vocab/backbone",
                "pid": 1,
                "tid": 1,
                "ph": "X",
                "ts": 1e6,
                "dur": 1e6,
            },
            {"cat": "kernel", "name": "gemm", "ts": 1.2e6, "dur": 0.5e6},
            {"cat": "kernel", "name": "nccl", "ts": 1.5e6, "dur": 0.5e6},
            {"cat": "gpu_memcpy", "name": "copy", "ts": 2.1e6, "dur": 0.1e6},
        ],
    }
    raw = [
        {"pid": 1, "name": "full_vocab/backbone", "start_ns": int(101e9), "wall_ns": int(1002e9), "mono_ns": int(102e9)}
    ]
    job = {
        "origin_ns": int(100e9),
        "steps": [{"step": 4, "start": 0, "end": 5}],
        "spans": [
            {"name": "timer/update_actor", "start": 1, "end": 2.5},
            {"name": "timer/update_weights", "start": 4, "end": 5},
        ],
    }
    path = tmp_path / "actor-rollout-test.json.gz"
    with gzip.open(path, "wt") as stream:
        json.dump(trace, stream)
    result = review.trace_review(path, job, raw)
    actor, sync = result["phases"]
    assert result["calibration"]["max_error_ms"] == 0
    assert actor["gpu_active_s"] == pytest.approx(0.9)
    assert actor["compute_nccl_overlap_s"] == pytest.approx(0.2)
    assert actor["nccl_without_compute_s"] == pytest.approx(0.3)
    assert actor["no_gpu_event_s"] == pytest.approx(0.6)
    assert not sync["covered"]
    assert "gpu_active_s" not in sync  # Missing capture must never become zero work.
    raw[0]["start_ns"] += int(0.01e9)
    with pytest.raises(ValueError, match="exceeds 5 ms"):
        review.calibrate(trace, raw, job["origin_ns"])


def test_request_boundaries_sum_to_generation_interval():
    job = {
        "steps": [{"step": 1, "start": 0, "end": 10}],
        "spans": [
            {"name": "timer/gen", "start": 1, "end": 8},
            {"name": "rollout/request", "start": 2, "end": 5, "output_tokens": 10},
            {"name": "rollout/sample", "start": 1.5, "end": 7},
            {"name": "timer/compute_score", "start": 5, "end": 6.8},
        ],
    }
    row = review.request_boundaries(job)[0]
    assert [
        row[k] for k in ("before_request_s", "request_envelope_s", "after_requests_to_samples_s", "after_samples_s")
    ] == [1, 3, 2, 1]
    assert row["max_score_rpc_s"] == pytest.approx(1.8)
