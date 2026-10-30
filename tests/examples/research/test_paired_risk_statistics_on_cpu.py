# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Paired inference preserves shared sample variation and rejects invalid data."""

import pytest

from examples.mirroropd.research.core.statistics import paired_mean_difference


def test_shared_variation_cancels_for_constant_paired_shift():
    result = paired_mean_difference([1, 100, -50], [3, 102, -48])
    assert result["delta"] == result["ci_low"] == result["ci_high"] == 2
    assert result["increased_prompts"] == 3
    reverse = paired_mean_difference([3, 102, -48], [1, 100, -50])
    assert reverse["delta"] == reverse["ci_low"] == reverse["ci_high"] == -2


@pytest.mark.parametrize("before,after", [([], []), ([1], [1, 2]), ([float("nan")], [1]), ([[1]], [[2]])])
def test_rejects_invalid_paired_inputs(before, after):
    with pytest.raises(ValueError):
        paired_mean_difference(before, after)
