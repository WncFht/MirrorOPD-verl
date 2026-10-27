# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""FP16 inputs require FP32 projection and gradient accumulation, like BF16."""

import pytest
import torch

from verl.trainer.distillation.full_vocab import full_vocab_objective_result, project
from verl.workers.config.distillation_spec import resolve_distillation_spec

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("objective,eta", [("mdpo", 1.0), ("mdpo", 0.3), ("bpo", 0.3)])
@pytest.mark.parametrize("backend", ["dense", "precompute", "precompute_fused"])
@pytest.mark.parametrize("vocab", [31, 8193])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_fp16_full_vocab_matches_fp64_hidden_and_head_vjp(objective, eta, backend, vocab, dtype):
    torch.manual_seed(87)
    hs, ws = [
        (torch.randn(*shape, device="cuda", dtype=dtype) * 0.2).requires_grad_() for shape in [(5, 16), (vocab, 16)]
    ]
    ht, wt, hq, wq = [
        torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * 0.2
        for shape in [(5, 12), (vocab + 3, 12), (5, 16), (vocab, 16)]
    ]
    spec = resolve_distillation_spec(
        {"loss_mode": "mirror", "mirror": {"objective": objective, "kappa": 0.5, "eta": eta, "diagnostics": True}}
    )
    rh, rw = [x.detach().double().requires_grad_() for x in (hs, ws)]
    expected = full_vocab_objective_result(
        rh, rw, ht.double(), wt.double(), spec=spec, hq=hq.double(), wq=wq.double(), backend="dense"
    )
    reference = torch.autograd.grad(expected.loss_sum / eta, (rh, rw))
    actual = full_vocab_objective_result(hs, ws, ht, wt, spec=spec, hq=hq, wq=wq, backend=backend, token_chunk=2)
    assert project(hs, ws).dtype == torch.float32
    torch.testing.assert_close(actual.loss_sum.double(), expected.loss_sum, atol=4e-6, rtol=2e-4)
    gradients = torch.autograd.grad(actual.loss_sum / eta, (hs, ws))
    for gradient, wanted in zip(gradients, reference, strict=True):
        relative = (gradient.double() - wanted).norm() / wanted.norm()
        assert relative < 0.003, relative.item()


@pytest.mark.parametrize("eta", [1e-3, 1e-2])
@pytest.mark.parametrize("backend", ["precompute", "precompute_fused"])
def test_fp16_precompute_preserves_small_vjp_until_upstream_scaling(eta, backend):
    from verl.trainer.distillation.targets import build_log_weights

    torch.manual_seed(91)
    hs, ws = [
        (torch.randn(*shape, device="cuda", dtype=torch.float16) * 0.2).requires_grad_()
        for shape in [(5, 16), (8193, 16)]
    ]
    ht, wt = [torch.randn_like(x) * 0.2 for x in (hs, ws)]
    hq, wq = hs.detach().clone(), ws.detach().clone()
    spec = resolve_distillation_spec({"loss_mode": "mirror", "mirror": {"objective": "mdpo", "kappa": 0.0, "eta": eta}})
    with torch.no_grad():
        z = project(hs, ws).double()
        b = build_log_weights(project(ht, wt).log_softmax(-1), project(hq, wq).log_softmax(-1), kappa=0.0, eta=eta)
        lp = z.log_softmax(-1)
        p = lp.exp()
        gap = lp - b.double()
        dz = p * (gap - (p * gap).sum(-1, keepdim=True)) / eta
        reference = (dz @ ws.double(), dz.T @ hs.double())
    actual = full_vocab_objective_result(hs, ws, ht, wt, spec=spec, hq=hq, wq=wq, backend=backend, token_chunk=2)
    scale = 1024.0
    gradients = torch.autograd.grad(actual.loss_sum * scale / eta, (hs, ws))
    for gradient, wanted in zip(gradients, reference, strict=True):
        relative = (gradient.double() / scale - wanted).norm() / wanted.norm()
        assert relative < 0.02, relative.item()


def test_fp16_termination_vjp_keeps_small_gradients_before_amp_scale():
    from verl.trainer.distillation.full_vocab import full_reverse_kl_result

    torch.manual_seed(94)
    hs, ws = [
        (torch.randn(*shape, device="cuda", dtype=torch.float16) * 0.2).requires_grad_()
        for shape in [(5, 16), (8193, 16)]
    ]
    ht = hs.detach().clone()
    wt = ws.detach() + torch.randn_like(ws) * 2e-4
    rh, rw = [x.detach().double().requires_grad_() for x in (hs, ws)]
    lp, lt = (rh @ rw.T).log_softmax(-1), (ht.double() @ wt.double().T).log_softmax(-1)
    gap = lp - lt
    gap[:, [2, 3]] = (lp[:, [2, 3]].logsumexp(-1) - lt[:, [2, 3]].logsumexp(-1))[:, None]
    expected = (lp.exp() * gap).sum()
    reference = torch.autograd.grad(expected, (rh, rw))
    actual = full_reverse_kl_result(hs, ws, ht, wt, backend="precompute", token_chunk=2, termination_token_ids=(2, 3))
    scale = 1024.0
    gradients = torch.autograd.grad(actual.loss_sum * scale, (hs, ws))
    for gradient, wanted in zip(gradients, reference, strict=True):
        relative = (gradient.double() / scale - wanted).norm() / wanted.norm()
        assert relative < 0.02, relative.item()
