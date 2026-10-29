# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Guard against reporting overlapping GPU streams or CPU-only traces as busy GPU time."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "examples/mirroropd/research/studies/pipeline/summarize_decode_trace.py"
spec = importlib.util.spec_from_file_location("decode_trace", SCRIPT)
trace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trace)


def event(start, duration, device=0, category="kernel"):
    return {"ph": "X", "cat": category, "name": "test", "ts": start, "dur": duration, "args": {"device": device}}


def test_overlapping_streams_count_once():
    result = trace.summarize({"traceEvents": [event(0, 10), event(5, 10), event(20, 5)]})
    gpu = result["devices"][0]
    assert gpu["device_activity_union_ms"] == pytest.approx(0.020)
    assert gpu["device_activity_envelope_fraction"] == pytest.approx(0.8)


def test_devices_and_memcpy_are_accounted_separately():
    result = trace.summarize(
        {
            "traceEvents": [
                event(0, 10),
                event(10, 10, category="gpu_memcpy"),
                event(0, 5, device=1),
            ]
        }
    )
    assert result["devices"][0]["kernel_union_ms"] == pytest.approx(0.010)
    assert result["devices"][0]["device_activity_union_ms"] == pytest.approx(0.020)
    assert result["devices"][1]["device_activity_union_ms"] == pytest.approx(0.005)


def test_cpu_only_trace_is_rejected():
    with pytest.raises(ValueError, match="no CUDA kernels"):
        trace.summarize({"traceEvents": [event(0, 100, category="cpu_op")]})
