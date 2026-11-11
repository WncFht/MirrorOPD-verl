# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Check CPU state placement against native AdamW, including repeated updates."""

import pytest
import torch

from examples.mirroropd.research.studies.spectral.iso.optimizers import WeightAdam


def test_offloaded_adam_matches_native_history_and_does_not_alias_master():
    torch.manual_seed(72)
    original = torch.nn.Parameter(torch.randn(17, 9))
    reference = torch.nn.Parameter(original.detach().clone())
    offload = WeightAdam([("matrix", original)], lr=1e-3)
    native = torch.optim.AdamW([reference], lr=1e-3, weight_decay=0)
    assert offload.entries[0][1].data_ptr() != original.data_ptr()
    for _ in range(4):
        gradient = torch.randn_like(original)
        original.grad, reference.grad = gradient, gradient.clone()
        offload.step()
        native.step()
        torch.testing.assert_close(original, reference, rtol=0, atol=0)
    assert offload.offload
    original.grad = None
    with pytest.raises(ValueError, match="no loss gradient"):
        offload.step()
