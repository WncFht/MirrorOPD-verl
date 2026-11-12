# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Regression checks for variable-length normalization and checkpoint selection."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from examples.mirroropd.research.studies.teacher_difference.inputs import normalize_records
from examples.mirroropd.research.studies.teacher_difference.reductions import record_denominator, response_denominator
from examples.mirroropd.research.studies.teacher_difference.report_response import factors
from examples.mirroropd.research.studies.teacher_difference.trajectory import tasks


def test_variable_length_report_uses_record_diagonal_before_averaging():
    data = dict(
        signature=dict(task=dict(cap=8, reduction="valid_tokens"), probes=2),
        groups=[dict(field="live", split="heldout", rows=["a", "b"])],
        records=[
            dict(
                row_id=key,
                token_count=length,
                denominator=length,
                moments=[1, 2, 3, 1, 1, 0, 1],
                native_parallel_inner=1,
                norm_sq={"live": 10, "parallel": 1, "sign-0": energy, "sign-1": energy},
            )
            for key, length, energy in (("a", 2, 1), ("b", 4, 3))
        ],
    )
    out = factors(data, "heldout", np.ones((1, 1)))
    # D = E[sum ||g_token||² / length²], U = E[length * D_question].
    np.testing.assert_allclose(out["diagonal"], [2])
    np.testing.assert_allclose(out["U"], [7])
    np.testing.assert_allclose(out["offdiagonal"], [8])
    np.testing.assert_allclose(out["aggregation_over_diagonal"], [5])
    np.testing.assert_allclose(out["A"], out["A_token"] * out["R_agg"])


def test_protocol_rejects_unknown_policy_and_missing_lengths():
    assert response_denominator(2, 8) == 8
    assert response_denominator(2, 8, "valid_tokens") == 2
    assert record_denominator({}, dict(cap=8)) == 8
    for length in (0, 9):
        with pytest.raises(ValueError):
            response_denominator(length, 8)
    with pytest.raises(ValueError):
        response_denominator(2, 8, "typo")
    with pytest.raises(ValueError):
        record_denominator({}, dict(cap=8, reduction="valid_tokens"))
    with pytest.raises(ValueError):
        record_denominator(dict(token_count=2, denominator=8), dict(cap=8, reduction="valid_tokens"))


def test_manifest_accepts_step640_but_rejects_missing_checkpoint(tmp_path):
    manifest = tmp_path / "manifest.json"
    initial = dict(id="base", model="qwen", branch="native", stage=0, student="base", target="teacher")
    final = dict(initial, id="final", stage=640, student="step640")
    manifest.write_text(json.dumps(dict(tasks=[initial, final])))
    args = SimpleNamespace(
        manifest=manifest, model="qwen", branch="native", stages=[0, 640], token_chunk=128, max_groups=None
    )
    base, selected = tasks(args)
    assert base == "base" and [item["stage"] for item in selected] == [0, 640]
    args.stages = [320]
    with pytest.raises(ValueError, match="absent"):
        tasks(args)


def test_bank_multiple_eos_and_invalid_inputs():
    row = dict(prompt_ids=[1], response_ids=[2, 7, 8, 9], eos_token_id=[7, 9])
    assert normalize_records([row])[0]["response_ids"] == [2, 7]
    for rows, cap in (
        ([row], 0),
        ([dict(row, prompt_ids=[])], 8),
        ([dict(row, id="x"), dict(row, id="x", prompt_ids=[3])], 8),
    ):
        with pytest.raises(ValueError):
            normalize_records(rows, cap=cap)


def test_independent_source_has_no_future_training_batch_claim():
    from examples.mirroropd.research.studies.teacher_difference.inputs import (
        future_source_groups,
        validate_source_exposure,
    )

    rows = [dict(split="train", group="new-panel")]
    task = dict(stage=640, source_policy="independent")
    assert future_source_groups(task, rows) == set()
    validate_source_exposure(task, rows)
    with pytest.raises(ValueError, match="batch_index"):
        future_source_groups(dict(stage=640), rows)
    historical = [dict(split="train", group="old", batch_index=1), dict(split="train", group="future", batch_index=20)]
    assert future_source_groups(dict(stage=320), historical) == {"future"}
    with pytest.raises(ValueError, match="unseen"):
        validate_source_exposure(dict(stage=320), historical)


def test_multi_bank_selection_is_explicit(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            dict(
                tasks=[
                    dict(id=bank, bank=bank, model="qwen", branch="native", stage=0, student="base", target="teacher")
                    for bank in ("initial", "teacher")
                ]
            )
        )
    )
    args = SimpleNamespace(
        manifest=manifest, model="qwen", branch="native", stages=[0], token_chunk=128, max_groups=None
    )
    with pytest.raises(ValueError, match="--bank"):
        tasks(args)
    args.bank = "initial"
    assert len(tasks(args)[1]) == 1


def test_source_group_selection_rejects_silent_negative_indexing():
    from examples.mirroropd.research.studies.teacher_difference.trajectory import select_source_groups

    groups = [(("heldout", "receiver"), []), (("train", "source"), ["row"])]
    assert select_source_groups(groups, [0]) == [(("train", "source"), ["row"])]
    for indices in ([], [-1], [1], [0, 0]):
        with pytest.raises(ValueError, match="source indices"):
            select_source_groups(groups, indices)
