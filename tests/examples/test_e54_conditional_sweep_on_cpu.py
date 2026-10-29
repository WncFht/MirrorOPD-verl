# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Only complete paired pilot evidence can authorize the conditional sweep."""

import pytest

from examples.mirroropd.research.studies.termination.queue_e54_mdpo_sweep import decide


def summary(improved=True):
    records = []
    for arm in ("k0-eta01", "k1-eta01", "k0-eta02", "k1-eta02"):
        better = improved and arm == "k1-eta01"
        for dataset in ("math", "dapo"):
            records.append(
                dict(
                    arm=arm,
                    dataset=dataset,
                    responses=64,
                    accuracy=0.5,
                    correct_and_stopped=22 if better else 20,
                    cap_count=18 if better else 20,
                    mean_length=19000 if better else 20000,
                    repeat8_mean=0.35 if better else 0.4,
                )
            )
    return {"evaluation": records, "paired_training_prompts": True}


def test_clear_stopping_improvement_with_preserved_accuracy_triggers():
    result = decide(summary())
    assert result["launch"]
    assert [r["eligible"] for r in result["comparisons"]] == [True, False]
    assert result["comparisons"][0]["eta"] == 0.1


def test_unchanged_results_do_not_trigger():
    assert not decide(summary(False))["launch"]


@pytest.mark.parametrize(
    "field,value", [("accuracy", 31 / 64), ("mean_length", 22000), ("repeat8_mean", 0.5), ("cap_count", 20)]
)
def test_regressions_prevent_automatic_sweep(field, value):
    data = summary()
    for row in data["evaluation"]:
        if row["arm"] == "k1-eta01":
            row[field] = value
    assert not decide(data)["launch"]


def test_one_dataset_cannot_hide_large_accuracy_regression_in_the_other():
    data = summary()
    for row in data["evaluation"]:
        if row["arm"] == "k1-eta01":
            row["accuracy"] = (29 if row["dataset"] == "math" else 35) / 64
    assert not decide(data)["launch"]


@pytest.mark.parametrize("failure", ["missing", "duplicate", "unpaired", "nonfinite", "denominator", "counts"])
def test_invalid_evidence_cannot_trigger(failure):
    data = summary()
    if failure == "missing":
        data["evaluation"].pop()
    elif failure == "duplicate":
        data["evaluation"].append(data["evaluation"][0])
    elif failure == "unpaired":
        data["paired_training_prompts"] = False
    elif failure == "nonfinite":
        data["evaluation"][0]["mean_length"] = float("nan")
    elif failure == "denominator":
        data["evaluation"][0]["responses"] = 63
    elif failure == "counts":
        data["evaluation"][0]["correct_and_stopped"] = 65
    with pytest.raises(ValueError):
        decide(data)
