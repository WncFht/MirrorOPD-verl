# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Plots must use exact held-out datasets, honest update coordinates and live records."""

import json

import pytest

from examples.mirroropd.tools.reuse_report import block_means, extract, read_live_records


def test_report_excludes_development_aggregate_and_preserves_preupdate_coordinate():
    rows = [
        {"step/optimizer": 1, "optim/actor/update_applied": 1, "optim/actor/loss": 0.5, "distill/teacher_rkl": 0.5},
        {
            "step/optimizer": 160,
            "eval/rkl/public_initial/math/question_mean": 99,
            "eval/rkl/public_initial/dataset/math/question_mean": 0.3,
            "eval/rkl/public_teacher/dataset/math/question_mean": 0.4,
            "reuse/gap_over_drop": -0.2,
            "reuse/used": 16,
            "reuse/next": 8,
            "reuse/switched": 1,
        },
    ]
    result = extract(rows, "adaptive")
    assert result["training"][0]["update"] == 0
    assert [r["rkl"] for r in result["heldout"]] == [0.3, 0.4]
    assert all(r["update"] == 160 for r in result["heldout"])
    assert result["signals"][0]["s"] == -0.2
    assert result["schedule"][0]["next"] == 8
    with pytest.raises(ValueError, match="Duplicate"):
        extract(rows + rows, "adaptive")


def test_live_reader_ignores_only_unfinished_final_line(tmp_path):
    path = tmp_path / "metrics.jsonl"
    complete = json.dumps(dict(data={"step/optimizer": 1})) + "\n"
    path.write_text(complete + '{"da')
    rows, digest = read_live_records(path)
    assert rows == [{"step/optimizer": 1}] and len(digest) == 64
    path.write_text(complete + '{"da\n')
    with pytest.raises(ValueError, match="Malformed"):
        read_live_records(path)


def test_online_missing_s_is_not_filled_and_block_means_do_not_extrapolate():
    rows = [dict(update=i, loss=float(i)) for i in range(17)]
    assert block_means(rows) == [dict(update=7.5, loss=7.5), dict(update=16, loss=16)]
    result = extract([{"step/optimizer": 1, "reuse/s_defined": 0}], "online")
    assert result["signals"] == []
