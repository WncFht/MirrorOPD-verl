# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Data-independent positional aggregation, masking semantics and global merging."""

import math

import pytest
import torch

from verl.utils.metric.observations import Observation
from verl.utils.metric.position import position_observations, position_statistics, relative_position_bins


def test_original_positions_preserve_gaps_and_short_sequence_midpoints():
    positions = torch.tensor([0] + list(range(2, 10)) + list(range(20)) + [0])
    lengths = torch.tensor([10] * 9 + [20] * 20 + [1])
    bins = relative_position_bins(positions, lengths)
    expected_bins = [0] + list(range(2, 10)) + [i // 2 for i in range(20)] + [5]
    assert bins.tolist() == expected_bins
    values = torch.arange(1, 31, dtype=torch.float64, requires_grad=True)
    sums, counts = position_statistics(values, bins)
    assert counts.tolist() == [expected_bins.count(i) for i in range(10)]
    assert sums.tolist() == [sum(j + 1 for j, b in enumerate(expected_bins) if b == i) for i in range(10)]
    assert sums.sum() == values.detach().sum() and counts.sum() == 30
    assert not sums.requires_grad


def test_same_bins_support_signed_values_indicators_and_independent_namespaces():
    bins = relative_position_bins(torch.tensor([0, 1, 0, 1]), torch.tensor([2, 2, 2, 2]))
    signed = position_observations(
        *position_statistics(torch.tensor([-1.0, 2.0, 3.0, -4.0]), bins), prefix="advantage/position"
    )
    indicators = position_observations(
        *position_statistics(torch.tensor([True, False, True, True]), bins), prefix="policy/clip_position/"
    )
    assert signed["advantage/position/p20_30/mean"].aggregate() == 1
    assert signed["advantage/position/p70_80/mean"].aggregate() == -1
    assert indicators["policy/clip_position/p70_80/mean"].aggregate() == 0.5
    assert not any("distillation" in k or "kl" in k for k in signed | indicators)


def test_unequal_partition_counts_reconstruct_prefix_mean():
    bins = torch.tensor([0, 0, 1, 2], dtype=torch.int64)
    values = torch.tensor([1.0, 3.0, 5.0, 9.0])
    parts = [
        position_observations(*position_statistics(values[a:b], bins[a:b]), prefix="token/nll")
        for a, b in [(0, 1), (1, 4), (4, 4)]
    ]
    merged = {k: Observation.merge_partitions([part[k] for part in parts]) for k in parts[0]}
    total = sum(merged[f"token/nll/{label}/sum"] for label in ["p00_10", "p10_20", "p20_30"])
    count = sum(merged[f"token/nll/{label}/token_count"] for label in ["p00_10", "p10_20", "p20_30"])
    assert total / count == 4.5
    assert merged["token/nll/p00_10/mean"] == 2
    assert math.isnan(merged["token/nll/p90_100/mean"])


def test_empty_and_nonfinite_values_are_not_silently_dropped():
    empty = torch.empty(0, dtype=torch.int64)
    sums, counts = position_statistics(torch.empty(0), relative_position_bins(empty, empty))
    assert sums.sum() == counts.sum() == 0
    sums, counts = position_statistics(torch.tensor([float("nan"), 2.0]), torch.tensor([0, 0]))
    assert torch.isnan(sums[0]) and counts[0] == 2


@pytest.mark.parametrize("positions,lengths", [([-1], [5]), ([5], [5]), ([0], [0])])
def test_out_of_sequence_coordinates_fail(positions, lengths):
    with pytest.raises(ValueError, match="inside"):
        relative_position_bins(torch.tensor(positions), torch.tensor(lengths))


def test_shapes_and_bin_contract_are_checked():
    with pytest.raises(ValueError, match="integers"):
        relative_position_bins(torch.tensor([0.5]), torch.tensor([1]))
    with pytest.raises(ValueError, match="aligned"):
        position_statistics(torch.ones(2), torch.tensor([0]))
    with pytest.raises(ValueError, match="range"):
        position_statistics(torch.ones(1), torch.tensor([10]))


def test_response_prediction_coordinates_preserve_prompt_offsets_and_masked_gaps():
    from verl.workers.utils.padding import as_padded_response, prediction_positions, response_target_coordinates

    ids = torch.nested.nested_tensor([torch.arange(6), torch.arange(4)], layout=torch.jagged)
    responses = torch.nested.nested_tensor([torch.arange(3), torch.arange(2)], layout=torch.jagged)
    masks = torch.nested.nested_tensor([torch.tensor([1, 0, 1]), torch.tensor([0, 1])], layout=torch.jagged)
    predictions, packed = prediction_positions(ids, masks)
    target, lengths = response_target_coordinates(ids, responses, predictions)
    assert target.tolist() == [0, 2, 1]
    assert lengths.tolist() == [3, 3, 2]
    assert packed.tolist() == [2, 4, 8]
    values = torch.tensor([[1.0, 2.0, 3.0]])
    assert as_padded_response(values) is values
    assert as_padded_response(None) is None
    assert as_padded_response(masks, False).tolist() == [[1, 0, 1], [0, 1, 0]]
