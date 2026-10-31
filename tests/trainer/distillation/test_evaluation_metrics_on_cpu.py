# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Scientific aggregation differs intentionally from the token-weighted training loss."""

import pytest

from verl.trainer.evaluation.metrics import readout_metrics, summarize_rkl
from verl.trainer.evaluation.schedule import Schedule


def row(key, risk, tokens, group=None):
    return dict(
        id=key,
        source_id=key,
        rkl_sum=risk * tokens,
        tokens=tokens,
        bank="history" if group is not None else "source_heldout",
        history_batch=group,
    )


def test_question_weighting_and_equal_batch_weighting_are_distinct():
    rows = [row("short", 1, 1, 1), row("long", 3, 100, 1), row("other", 10, 2, 2)]
    assert summarize_rkl(rows)["mean"] == pytest.approx(14 / 3)
    assert summarize_rkl(rows, equal_groups="history_batch")["mean"] == 6
    metrics = readout_metrics(rows)
    assert metrics["eval/rkl/history/equal_batch_question_mean"] == 6
    assert "eval/rkl/history/equal_batch_token_mean" not in metrics


@pytest.mark.parametrize("bad", [row("bad", 0, 0), row("bad", float("nan"), 1), row("bad", float("inf"), 1)])
def test_invalid_rows_have_coverage_but_no_primary_mean(bad):
    summary = summarize_rkl([row("ok", 1, 3), bad])
    assert summary["valid_rows"] == summary["invalid_rows"] == 1
    assert "mean" not in summary


def test_baseline_and_history_pairing_do_not_compare_different_sets():
    baseline = [row("a", 2, 1), row("b", 4, 20)]
    current = [row("a", 1, 1), row("b", 2, 20), row("old", 2, 5, 1), row("new", 100, 1, 2)]
    previous = [row("old", 3, 5, 1), row("gone", 0, 10, 2)]
    metrics = readout_metrics(current, baseline=baseline, previous_history=previous)
    assert metrics["eval/rkl/source_heldout/reduction_percent"] == 50
    assert metrics["eval/rkl/history/paired_rows"] == 1
    assert metrics["eval/rkl/history/paired_absolute_reduction"] == 1
    zero = [row("a", 0, 1)]
    assert "eval/rkl/source_heldout/reduction_percent" not in readout_metrics(zero, baseline=zero)
    with pytest.raises(ValueError, match="baseline"):
        readout_metrics(current, baseline=[row("different", 2, 1)])
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_rkl([zero[0], zero[0]])


def test_schedules_union_final_and_independent_frequencies():
    bank = Schedule(at=(0, 1, 2, 2), every=5)
    assert bank.at == (0, 1, 2)
    assert [step for step in range(11) if bank.due(step, 10)] == [0, 1, 2, 5, 10]
    assert bank.due(7, 7)
    assert not Schedule(every=1).due(0, 50)
    for value in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            Schedule(every=value)
    with pytest.raises(ValueError):
        bank.due(11, 10)
