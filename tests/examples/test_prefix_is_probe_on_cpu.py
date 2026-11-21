# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Check probability-ratio numerics and the prefix/action boundary."""

import numpy as np
import pytest

from examples.mirroropd.tools.prefix_is_probe import prefix_table, weight_summary


def test_weights_and_clipping_remain_stable_for_extreme_log_ratios():
    result = weight_summary([1000.0, 0.0, -1000.0])
    assert result["ess"] == pytest.approx(1)
    assert result["mean_weight"] is None
    assert result["clip2"]["ess_fraction"] == pytest.approx(3 / 5)
    assert result["clip2"]["removed_sample_weight_fraction"] == 1
    assert weight_summary(np.zeros(64))["ess"] == pytest.approx(64)


def test_prefix_excludes_current_action_and_masks_only_state_not_its_history():
    behavior = [np.zeros(33), np.zeros(33)]
    current = [np.zeros(33), np.zeros(33)]
    current[0][32] = 100  # The action at state32 must not enter its prefix ratio.
    result = prefix_table(current, behavior)
    assert result["32"]["ess_fraction"] == pytest.approx(1)
    current[0][0] = np.log(2)
    masks = [np.ones(33, dtype=bool), np.ones(33, dtype=bool)]
    masks[0][0] = False  # Earlier sampled actions still belong to later prefixes.
    result = prefix_table(current, behavior, masks)
    assert result["0"]["n"] == 1
    assert result["32"]["ess"] == pytest.approx(9 / 5)


def test_refuse_nonfinite_or_misaligned_probabilities():
    with pytest.raises(ValueError):
        weight_summary([float("nan")])
    with pytest.raises(ValueError):
        prefix_table([[0, 0]], [[0]])


def test_initial_validation_restores_rollout_residency():
    from types import SimpleNamespace
    from unittest.mock import Mock

    from examples.mirroropd.tools.prefix_is_probe import PrefixISProbe

    probe = object.__new__(PrefixISProbe)
    probe._banks = Mock(return_value={"rkl": 0.25})
    service = SimpleNamespace(ensure_current=Mock())
    trainer = SimpleNamespace(actor_optimizer_step=480, global_steps=30, rollout_service=service)
    assert probe.validate(trainer) == {"rkl": 0.25}
    service.ensure_current.assert_called_once_with(30)


def test_chunked_token_projection_matches_direct_softmax_with_sparse_support():
    import torch

    from examples.mirroropd.tools.prefix_is_probe import selected_token_log_probs

    generator = torch.Generator().manual_seed(42)
    h = torch.randn(7, 3, generator=generator)
    w = torch.randn(9, 3, generator=generator)
    legal = [0, 2, 4, 8]
    indices = torch.tensor([0, 2, 1, 3, 0, 1, 2])
    actions = torch.tensor(legal)[indices]
    actual = selected_token_log_probs(h, w, actions, legal, token_chunk=2)
    expected = (h @ w[legal].T).log_softmax(-1).gather(-1, indices[:, None]).squeeze(-1)
    torch.testing.assert_close(actual, expected)
    with pytest.raises(ValueError):
        selected_token_log_probs(h[:1], w, torch.tensor([1]), legal)
