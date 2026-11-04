# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Stopping rules must not turn truncation into apparent natural completion."""

import pytest

from verl.utils.metric.finish_reason import finish_reason_metrics


def test_finish_reason_denominator_and_disjoint_counts():
    result = finish_reason_metrics(["stop", "length", "repetition", "abort", "completed", None, "other"])
    assert result["response_count"] == 7
    assert result["unknown_count"] == 2
    assert result["aborted_count"] == 1
    assert result["truncated_count"] == 2
    assert result["truncated_fraction"] == pytest.approx(2 / 7)
    assert sum(
        result[f"{r}_fraction"] for r in ("stop", "length", "repetition", "aborted", "completed", "unknown")
    ) == pytest.approx(1)
    assert finish_reason_metrics([]) == {}


def test_early_repetition_can_lower_cap_hits_without_more_natural_stops():
    before = finish_reason_metrics(["stop", "length", "length"])
    after = finish_reason_metrics(["stop", "length", "repetition"])
    assert after["length_fraction"] < before["length_fraction"]
    assert after["stop_fraction"] == before["stop_fraction"]
    assert after["truncated_fraction"] == before["truncated_fraction"]
