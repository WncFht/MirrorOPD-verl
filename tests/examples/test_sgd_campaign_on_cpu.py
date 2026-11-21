# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Protect the development split and the decision that gates 21 GPU runs."""

import pytest

from examples.mirroropd.research.studies.absorption_dynamics.sgd_protocol import (
    candidate_score,
    development_split,
    next_lr,
)


def test_split_excludes_formal_training_and_heldout_text_even_with_duplicates():
    def row(text):
        return {"prompt": [{"role": "user", "content": text}]}

    training = [row(str(i)) for i in range(3000)]
    training[2560] = row(" 0 ")
    heldout = [{"messages": row("2561")["prompt"]}]
    train, dev = development_split(training, heldout)
    assert train[0]["prompt"][0]["content"] == "2562"
    train_text = {r["prompt"][0]["content"] for r in train}
    dev_text = {r["prompt"][0]["content"] for r in dev}
    assert len(train_text) == 256 and len(dev_text) == 32
    assert not train_text & dev_text


def test_selection_measures_future_batches_and_both_fixed_banks():
    events = []
    for outer in range(5):
        metrics = {}
        for bank, baseline in (("public_initial", 2.0), ("public_teacher", 4.0)):
            prefix = f"eval/rkl/{bank}/math/"
            metrics.update(
                {
                    prefix + "question_mean": baseline * (1 - outer * 0.1),
                    prefix + "invalid_rows": 0,
                    prefix + "valid_rows": 32,
                }
            )
        events.append(metrics)
    assert candidate_score(events) == pytest.approx(0.35)
    events[4]["eval/rkl/public_teacher/math/invalid_rows"] = 1
    with pytest.raises(ValueError, match="coverage"):
        candidate_score(events)


def test_edge_probes_are_bounded_and_interior_winner_finishes():
    rows = [dict(lr=0.003, score=0.1), dict(lr=0.01, score=0.2), dict(lr=0.03, score=0.3)]
    assert next_lr(rows, 0) == pytest.approx(0.09)
    assert next_lr(rows, 2) is None
    rows[1]["score"] = 0.4
    assert next_lr(rows, 0) is None


def test_adopt_completed_job_does_not_restart_failed_training(tmp_path):
    from examples.mirroropd.research.core.io import dump
    from examples.mirroropd.tools.sgd_campaign import adopt_job

    dump(tmp_path / "process.json", dict(pid=-1, identity=[-1, "unavailable"]))
    dump(tmp_path / "exit.json", dict(returncode=0))
    adopt_job(tmp_path, tmp_path)
    dump(tmp_path / "exit.json", dict(returncode=1))
    with pytest.raises(RuntimeError, match="Refuse restarting"):
        adopt_job(tmp_path, tmp_path)
