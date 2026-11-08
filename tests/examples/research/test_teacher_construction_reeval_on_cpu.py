# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Check resumable multi-sample coverage independently of the GPU transport."""

import json

import pytest

from examples.mirroropd.research.studies.fixed_transfer import teacher_construction_reeval as evaluation


def sample(index=0):
    protocol = evaluation.protocol("math")
    return dict(
        id=f"sample-{index}",
        dataset="MATH-500",
        source_id="question",
        domain="math",
        sample_index=index,
        seed=protocol.sample_seed("MATH-500", "question", index),
        protocol_id=protocol.protocol_id,
        response_ids=[42],
        response_tokens=1,
        finish_reason="stop",
        status="ok",
        score=float(index == 0),
        canonical_score=float(index == 0),
    )


QUESTIONS = [dict(dataset="MATH-500", id="question", domain="math")]


def test_resume_validates_slot_seed_protocol_and_tokens():
    seen, expected = evaluation.validate_rows([sample()], QUESTIONS, ["MATH-500"])
    assert seen == {("MATH-500", "question", 0)}
    assert len(expected) == 4
    for changed in (
        dict(sample_index=4),
        dict(source_id="foreign"),
        dict(seed=1),
        dict(protocol_id="different"),
        dict(response_tokens=2),
    ):
        with pytest.raises(ValueError):
            evaluation.validate_rows([sample() | changed], QUESTIONS, ["MATH-500"])
    with pytest.raises(ValueError, match="Duplicate"):
        evaluation.validate_rows([sample(), sample()], QUESTIONS, ["MATH-500"])


def test_sample0_and_average_are_separate_and_resume_requires_complete_coverage(tmp_path):
    rows = [sample(i) for i in range(4)]
    for filename in ("responses.jsonl", "scores.jsonl"):
        (tmp_path / filename).write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert evaluation.summarize(tmp_path, QUESTIONS, ["MATH-500"])
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["eval/accuracy/MATH-500/avg_at_n"] == 0.25
    assert metrics["four_domain_primary/MATH-500/avg_at_1"] == 1
    assert metrics["canonical/MATH-500/avg_at_n"] == 0.25
    (tmp_path / "responses.jsonl").write_text(json.dumps(rows[0]) + "\n")
    assert not evaluation.summarize(tmp_path, QUESTIONS, ["MATH-500"])


def test_declared_budgets_match_the_retest():
    assert evaluation.SPECS["AMC23"][2] == 16
    assert evaluation.SPECS["GPQA-Diamond"][2] == 4
    assert evaluation.SPECS["LCBv5"][2] == 10
    assert evaluation.protocol("math").max_new_tokens == 24576
    assert evaluation.protocol("if").max_new_tokens == 8192


def test_answer_channel_preserves_thinking_and_strips_only_terminal_controls():
    assert evaluation.answer_text(dict(text="<think>unfinished reasoning")) == ""
    assert evaluation.answer_text(dict(text="<think>work</think>42<eos>", terminal_tokens=["<eos>"])) == "42"
    assert evaluation.answer_text(dict(text="<other>42<eos>", terminal_tokens=["<eos>"])) == "<other>42"
