# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Single-card evaluation preserves its shape when the physical allocation expands."""

import pytest

from examples.mirroropd.research.studies.prefix_fusion.scoring import require_allocated_gpu


@pytest.mark.parametrize("device", [str(i) for i in range(8)])
def test_scoring_accepts_each_allocated_single_card(monkeypatch, device):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", device)
    require_allocated_gpu()


@pytest.mark.parametrize("devices", ["", "-1", "8", "0,1", "6,7", "0,0"])
def test_scoring_rejects_wrong_shape_or_unknown_card(monkeypatch, devices):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", devices)
    with pytest.raises(ValueError):
        require_allocated_gpu()
