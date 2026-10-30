# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Overall held-out ranking pools risk, never averages percentage reductions."""

import pytest

from examples.mirroropd.research.studies.fixed_transfer.overview import heldout_summary


def test_pooled_risk_is_not_mean_of_domain_percentages():
    rows = [
        dict(
            family="Qwen3",
            student="q3-1.7",
            teacher="q3-4",
            arm="a",
            steps=64,
            support="native",
            metric="rkl",
            panel=panel,
            prompts=n,
            initial=a,
            final=b,
        )
        for panel, n, a, b in [("held_math", 64, 1, 0.5), ("held_code", 32, 10, 9), ("held_if", 32, 10, 9)]
    ]
    value = heldout_summary(rows)[0]
    assert value["prompts"] == 128
    assert value["initial"] == 5.5 and value["final"] == 4.75
    assert value["drop_percent"] == pytest.approx(100 * (1 - 4.75 / 5.5))
    assert value["drop_percent"] != pytest.approx((50 + 10 + 10) / 3)
    assert value["model_label"] == "Qwen3-4B" and not value["same_size"]
    other = [dict(r, arm="b", teacher="e3", initial=1.0, final=0.8) for r in rows]
    ordered = heldout_summary(rows + other)
    assert [r["arm"] for r in ordered] == ["b", "a"]
    assert ordered[0]["same_size"] and not ordered[1]["same_size"]
    with pytest.raises(ValueError, match="three held-out domains"):
        heldout_summary(rows[:-1])


def test_generated_report_blocks_preserve_interpretation_and_reject_ambiguous_markers():
    from examples.mirroropd.research.studies.fixed_transfer.overview import replace_block

    original = "interpretation\n<!-- generated:x:start -->\nold\n<!-- generated:x:end -->\nlimitations"
    result = replace_block(original, "x", "new measurements")
    assert result.startswith("interpretation\n") and result.endswith("\nlimitations")
    assert "old" not in result and "new measurements" in result
    assert replace_block(result, "x", "new measurements") == result
    with pytest.raises(ValueError, match="exactly one"):
        replace_block(original + original, "x", "new")
    with pytest.raises(ValueError, match="exactly one"):
        replace_block(original, "missing", "new")
