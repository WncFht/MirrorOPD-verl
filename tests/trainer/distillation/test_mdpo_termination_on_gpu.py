# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Production vocabulary and BF16 projection checks for coarsened MDPO."""

import pytest
import torch

from verl.trainer.distillation.full_vocab import full_vocab_objective_result, project
from verl.workers.config.distillation_spec import resolve_distillation_spec

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("kappa", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("eta", [0.05, 0.1, 0.2, 0.3])
def test_coarse_mdpo_production_vocabulary(kappa, eta):
    torch.manual_seed(17)
    vocab = 151936
    hs, ws = [
        (torch.randn(*s, device="cuda", dtype=torch.bfloat16) * 0.2).requires_grad_() for s in [(5, 16), (vocab, 16)]
    ]
    ht, wt, hq, wq = [
        torch.randn(*s, device="cuda", dtype=torch.bfloat16) * 0.2 for s in [(5, 12), (vocab, 12), (5, 16), (vocab, 16)]
    ]
    ids = [151643, 151645]
    keep = torch.ones(vocab, device="cuda", dtype=torch.bool)
    keep[ids] = False

    def probabilities(h, w):
        p = project(h, w).double().softmax(-1)
        return torch.cat([p[:, keep], p[:, ids].sum(-1, keepdim=True)], -1)

    p, q, t = probabilities(hs, ws), probabilities(hq, wq), probabilities(ht, wt)
    log_m = (
        (1 - eta) * q.log() + eta * t.log()
        if kappa == 0
        else ((1 - eta) * q.pow(kappa) + eta * t.pow(kappa)).log() / kappa
    )
    reference = (p * (p.log() - log_m)).sum() / eta
    gradients = torch.autograd.grad(reference, (hs, ws))
    spec = resolve_distillation_spec(
        {"loss_mode": "mirror", "mirror": {"objective": "mdpo", "kappa": kappa, "eta": eta, "diagnostics": True}}
    )
    actual = full_vocab_objective_result(
        hs, ws, ht, wt, hq=hq, wq=wq, spec=spec, backend="precompute", token_chunk=2, termination_token_ids=ids
    )
    torch.testing.assert_close(actual.loss_sum.double() / eta, reference, atol=2e-6, rtol=2e-5)
    for got, wanted in zip(torch.autograd.grad(actual.loss_sum / eta, (hs, ws)), gradients, strict=True):
        relative = (got.float() - wanted.float()).norm() / wanted.float().norm().clamp_min(1e-12)
        assert relative < 0.03, relative.item()
