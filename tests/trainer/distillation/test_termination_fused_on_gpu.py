# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""EOS fused numerical budget, identity checks, and public backend dispatch."""

from types import SimpleNamespace

import pytest
import torch
from test_termination_action_map_on_cpu import coarse_oracle

from verl.trainer.distillation.full_vocab import full_vocab_objective_result
from verl.trainer.distillation.full_vocab_triton import fused_mirror_logits
from verl.trainer.distillation.termination import TerminationActionMap
from verl.workers.config.distillation import DistillationLossConfig, FullVocabConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("kappa", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("eta", [1e-6, 0.1, 1.0])
@pytest.mark.parametrize("offset", [0.0, 32.0, 128.0])
def test_eos_vjp_against_independent_oracle(kappa, eta, offset):
    q = torch.full((1, 17), -2.0, device="cuda")
    q[0, 1], q[0, 4] = 0.0, -2.0
    t = q.clone()
    t[0, 4] += 2**-10
    z, q, t = q + offset, q + offset, t + offset
    mapping = TerminationActionMap.create(17, [1, 4], z.device)
    pairs = [mapping.coarsen(x, compensated=True) for x in (z, q, t)]
    spec = SimpleNamespace(objective="mdpo", kappa=kappa, eta=eta, diagnostics=True)
    value, gc, diag, _ = fused_mirror_logits(
        *(p[0] for p in pairs), torch.float32, spec=spec, corrections=tuple(p[1] for p in pairs)
    )
    gradient = mapping.expand_gradient(gc, z)
    rv, rg, rk = coarse_oracle(z[0].tolist(), q[0].tolist(), t[0].tolist(), [1, 4], kappa, eta)
    torch.testing.assert_close(gradient[0].double().cpu(), rg, rtol=2e-4, atol=1e-22)
    assert value.item() == pytest.approx(rv, rel=2e-4, abs=1e-24)
    assert diag[0, 0].item() == pytest.approx(rk, rel=2e-4, abs=1e-24)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("kappa", [0.0, 0.5, 1.0])
def test_public_fused_loss_diagnostics_and_teacher_freeze(dtype, kappa):
    torch.manual_seed(19)
    hs = torch.randn(7, 16, dtype=dtype, device="cuda", requires_grad=True)
    ws = (torch.randn(37, 16, dtype=dtype, device="cuda") * 0.1).requires_grad_()
    ht, wt = [torch.randn_like(x, requires_grad=True) for x in (hs, ws)]
    cfg = DistillationLossConfig(
        loss_mode="mirror",
        mirror=dict(objective="mdpo", kappa=kappa, eta=0.1, diagnostics=True),
        full_vocab=FullVocabConfig(backend="precompute_fused", termination_token_ids=[1, 4]),
        use_policy_gradient=False,
        use_task_rewards=False,
        policy_loss_mode="vanilla",
        loss_max_clamp=None,
        log_prob_min_clamp=None,
    )
    kwargs = dict(spec=cfg.loss_settings, hq=hs.detach(), wq=ws.detach(), token_chunk=3, termination_token_ids=[1, 4])
    expected = full_vocab_objective_result(hs, ws, ht, wt, backend="precompute", **kwargs)
    actual = full_vocab_objective_result(hs, ws, ht, wt, backend="precompute_fused", **kwargs)
    torch.testing.assert_close(actual.loss_sum, expected.loss_sum, rtol=2e-5, atol=2e-6)
    for key in actual.token_diagnostics:
        torch.testing.assert_close(actual.token_diagnostics[key], expected.token_diagnostics[key], rtol=2e-5, atol=2e-6)
    reference = torch.autograd.grad(expected.loss_sum * 1024, (hs, ws, ht, wt), allow_unused=True)
    result = torch.autograd.grad(actual.loss_sum * 1024, (hs, ws, ht, wt), allow_unused=True)
    assert reference[2:] == result[2:] == (None, None)
    for got, ref in zip(result[:2], reference[:2], strict=True):
        relative = (got.float() - ref.float()).norm() / ref.float().norm()
        assert relative < (0.01 if dtype == torch.bfloat16 else 0.002)
