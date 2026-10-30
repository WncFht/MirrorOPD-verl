# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Causal evidence boundaries and protection of changing mathematical content."""

from examples.mirroropd.research.studies.termination.compare_repetition_rules import (
    completed_blocks,
    review_auc,
    step_stops,
)


def evaluate(text):
    """Use one character per synthetic token to isolate boundary semantics."""
    return step_stops(text, list(range(1, len(text) + 1)))


def test_step_numbers_can_change_but_math_is_preserved():
    body = "The expression x + 17 = 23 is unchanged. " * 8
    text = "".join(f"### Step {i}: Compute\n{body}\n\n" for i in range(1, 6))
    result = evaluate(text)
    assert result["step-n3-t512"] < len(text)
    assert result["step-raw-n3-t512"] == len(text)
    changed = "".join(f"### Step {i}: Compute\n{body} x = {i}\n\n" for i in range(1, 7))
    assert evaluate(changed)["step-n3-t512"] == len(changed)


def test_third_repeat_is_not_complete_before_next_heading():
    body = "This is a long repeated derivation. " * 9
    text = "".join(f"### Step {i}: Compute\n{body}\n\n" for i in range(1, 5))
    assert evaluate(text)["step-n3-t512"] == len(text)
    extended = text + "### Step 5: Compute\n" + "new material " * 50
    assert evaluate(extended)["step-n3-t512"] == len(text) + len("### Step 5: Compute\n")


def test_future_heading_does_not_change_paragraph_decisions():
    text = (("a long identical paragraph " * 12) + "\n\n") * 5
    suffix = "### Step 1: New mode\n" + "new content " * 40
    assert evaluate(text)["step-n3-t512"] == evaluate(text + suffix)["step-n3-t512"]
    for start, end, observed in completed_blocks(text + suffix):
        assert start <= end <= observed <= len(text + suffix)


def test_short_formula_repetition_is_not_a_long_step():
    text = "x = 1\n\n" * 200
    assert evaluate(text)["step-n3-t512"] == len(text)


def test_review_auc_counts_ties_as_half_and_uses_only_labeled_records():
    rows = [
        {"source": "s", "prompt_id": str(i), "scores": {w: value for w in ("1024", "4096", "8192")}}
        for i, value in enumerate([3, 2, 2, 100])
    ]
    labels = {("s", "0"): 1, ("s", "1"): 1, ("s", "2"): 0}
    assert review_auc(rows, labels) == {"1024": 0.75, "4096": 0.75, "8192": 0.75}
