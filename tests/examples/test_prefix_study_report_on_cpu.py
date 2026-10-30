# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Keep fixed-batch learning comparisons distinct from changing-prefix observations."""

import csv
import hashlib
import json

import pytest

from examples.mirroropd.benchmarks.fusion.report_prefix_study import load_snapshot, summarize_outers


def batch(outer, start):
    return [
        dict(
            pair="J",
            arm="S0",
            run="synthetic",
            outer=outer,
            optimizer_step=(outer - 1) * 16 + i,
            reuse_index=i,
            rkl=start - i + 1,
            valid_update=True,
            loss_tokens=100,
        )
        for i in range(1, 17)
    ]


def test_compare_fixed_batch_endpoints_without_cross_batch_subtraction():
    first, second = summarize_outers(batch(1, 16) + batch(2, 32))
    assert first["within_batch_reduction_pct"] == pytest.approx(93.75)
    assert second["within_batch_reduction_pct"] == pytest.approx(46.875)
    assert first["batch_loss_tokens"] == 100
    assert first["reused_loss_tokens"] == 1600
    assert second["first_rkl"] == 32 and second["sixteenth_rkl"] == 17


@pytest.mark.parametrize("fault", ["partial", "invalid", "duplicate"])
def test_incomplete_or_invalid_reuse_never_becomes_endpoint_comparison(fault):
    rows = batch(1, 16)
    if fault == "partial":
        rows.pop()
    elif fault == "invalid":
        rows[5]["valid_update"] = False
    else:
        rows[5]["reuse_index"] = 5
    result = summarize_outers(rows)[0]
    assert not result["complete_reuse16"]
    assert result["within_batch_reduction_pct"] is None
    assert result["sixteenth_rkl"] is None


def test_zero_baseline_has_no_relative_reduction():
    rows = batch(1, 16)
    for row in rows:
        row["rkl"] = 0
    result = summarize_outers(rows)[0]
    assert result["complete_reuse16"]
    assert result["within_batch_reduction_pct"] is None


def test_frozen_replay_is_independent_of_live_sources_and_detects_changed_data(tmp_path):
    rows = batch(1, 16)
    rows[0]["valid_update"] = False
    rows[0]["rkl"] = None
    path = tmp_path / "updates.csv"
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    status = dict(
        recorded_update_rows=16,
        sources=[{"path": "/nonexistent/live/metrics.jsonl"}],
        updates_csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    (tmp_path / "status.json").write_text(json.dumps(status))
    replay, _ = load_snapshot(tmp_path)
    assert replay == rows
    path.write_text(path.read_text().replace("synthetic", "tampered", 1))
    with pytest.raises(ValueError, match="hash"):
        load_snapshot(tmp_path)
