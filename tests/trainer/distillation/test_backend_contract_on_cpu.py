# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Fused is an execution contract, not a request that silently selects a reference."""

import pytest
import torch

from verl.trainer.distillation.forward_vocab import forward_rows, full_forward_kl_result
from verl.trainer.distillation.full_vocab import full_vocab_objective_result
from verl.trainer.distillation.reverse_vocab import full_reverse_kl_result
from verl.workers.config.distillation_spec import resolve_distillation_spec


@pytest.mark.parametrize("tokens", [0, 3])
@pytest.mark.parametrize("objective", ["reverse", "forward", "mdpo", "bpo"])
def test_cpu_fused_request_fails_including_empty_ranks(tokens, objective):
    hs = torch.ones(tokens, 4, dtype=torch.float64, requires_grad=True)
    ws = torch.ones(7, 4, dtype=torch.float64, requires_grad=True)
    ht, wt = hs.detach(), ws.detach()
    with pytest.raises(ValueError, match="requires CUDA"):
        if objective == "reverse":
            full_reverse_kl_result(hs, ws, ht, wt, backend="precompute_fused")
        elif objective == "forward":
            full_forward_kl_result(hs, ws, ht, wt, backend="precompute_fused")
        else:
            spec = resolve_distillation_spec({"loss_mode": "mirror", "mirror": {"objective": objective, "eta": 0.3}})
            full_vocab_objective_result(hs, ws, ht, wt, spec=spec, hq=ht, wq=wt, backend="precompute_fused")


def test_forward_row_api_has_no_implicit_reference_fallback():
    z = torch.zeros(3, 7)
    with pytest.raises(ValueError, match="CUDA FP32"):
        forward_rows(z, z, fused=True)
    assert torch.isfinite(forward_rows(z, z, fused=False)[0]).all()
