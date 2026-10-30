# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Audit catches post-restore corruption of weights and Adam slots."""

import pytest
import torch
from safetensors.torch import save_file

from examples.mirroropd.research.core import training_state
from examples.mirroropd.research.core.restore_audit import verify_loaded


def test_exact_restoration_and_corruption_detection(tmp_path):
    model = torch.nn.Linear(3, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    model(torch.ones(1, 3)).sum().backward()
    opt.step()
    weights = tmp_path / "weights"
    weights.mkdir()
    save_file(model.state_dict(), weights / "model.safetensors")
    state = tmp_path / "resume"
    training_state.save(state, model, opt, step=1, signature={"a": 1}, weights=weights, weight_identity="test")
    assert verify_loaded(state, weights, model, opt) == (8, 6)
    with torch.no_grad():
        model.weight[0, 0] += 1
    with pytest.raises(AssertionError):
        verify_loaded(state, weights, model, opt)
    from safetensors.torch import load_file

    model.load_state_dict(load_file(weights / "model.safetensors"))
    opt.state[model.weight]["exp_avg"].add_(1)
    with pytest.raises(AssertionError):
        verify_loaded(state, weights, model, opt)
