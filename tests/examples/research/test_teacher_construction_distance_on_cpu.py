# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Sequence sums cannot be replaced by equally weighted per-question means."""

import pytest

from examples.mirroropd.research.studies.fixed_transfer.teacher_construction_distance import aggregate


def test_variable_lengths_preserve_three_distinct_fkl_statistics():
    rows = [
        dict(step=0, domain="math", id="short", tokens=1, fkl=2, rkl=3),
        dict(step=0, domain="math", id="long", tokens=3, fkl=4, rkl=5),
    ]
    value = aggregate(rows)[0]
    assert value["fkl_question_token_mean"] == 3
    assert value["fkl_sequence_sum"] == 7
    assert value["fkl_pooled_token_mean"] == 3.5
    assert value["rkl_question_token_mean"] == 4
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate([rows[0], rows[0]])
