# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent probability-space references for TOP-D-shaped full MDPO."""

import math

import pytest
import torch

from verl.trainer.distillation.full_vocab import full_reverse_kl_result, full_vocab_objective_result
from verl.trainer.distillation.targets import mirror_signal
from verl.workers.config.distillation import DistillationLossConfig, FullVocabConfig


def config(kappa, eta, objective="mdpo", estimator="full_vocab"):
    return DistillationLossConfig(
        loss_mode="mirror",
        mirror=dict(objective=objective, estimator=estimator, kappa=kappa, eta=eta, diagnostics=True),
        full_vocab=FullVocabConfig(backend="precompute", termination_token_ids=[1, 4]),
        use_policy_gradient=estimator == "sampled",
        use_task_rewards=False,
        policy_loss_mode="token_is" if estimator == "sampled" else "vanilla",
        loss_max_clamp=None,
        log_prob_min_clamp=None,
    )


def coarse(z):
    p = z.softmax(-1)
    keep = [i for i in range(z.shape[-1]) if i not in (1, 4)]
    return torch.cat([p[:, keep], p[:, [1, 4]].sum(-1, keepdim=True)], -1)


@pytest.mark.parametrize("eta", [0.1, 0.2, 1.0])
def test_kappa_one_is_the_paper_reward_but_not_the_complete_mdpo_update(eta):
    delta = torch.tensor([-1000.0, -30.0, -1.0, -1e-10, 0.0, 1e-10, 1.0, 30.0, 1000.0], dtype=torch.float64)
    ref = delta if eta == 1 else torch.logaddexp(delta + math.log(eta), delta.new_full(delta.shape, math.log1p(-eta)))
    actual = mirror_signal(delta, 1.0, eta)
    torch.testing.assert_close(actual, ref, atol=2e-15, rtol=2e-14)
    assert actual[4] == 0
    if eta < 1:
        assert actual.min() >= math.log1p(-eta) - 1e-15
    torch.manual_seed(3)
    q = torch.randn(2, 9, dtype=torch.float64).softmax(-1)
    t = torch.randn(2, 9, dtype=torch.float64).softmax(-1)
    reward = ((1 - eta) + eta * t / q).log().detach()
    for at_anchor in (True, False):
        z = (q.log() if at_anchor else torch.randn_like(q)).detach().requires_grad_()
        lp = z.log_softmax(-1)
        p = lp.exp()
        mdpo = (p * (lp - ((1 - eta) * q + eta * t).log())).sum() / eta
        reward_only = -(p * reward).sum()
        anchor_kl = (p * (lp - q.log())).sum()
        torch.testing.assert_close(eta * mdpo, anchor_kl + reward_only, atol=1e-12, rtol=1e-12)
        gm = torch.autograd.grad(mdpo, z, retain_graph=True)[0]
        gr = torch.autograd.grad(reward_only, z, retain_graph=True)[0]
        ga = torch.autograd.grad(anchor_kl, z)[0]
        torch.testing.assert_close(eta * gm, ga + gr, atol=1e-12, rtol=1e-12)
        if at_anchor:
            torch.testing.assert_close(eta * gm, gr, atol=1e-12, rtol=1e-12)
        else:
            assert ga.norm() > 0.01


@pytest.mark.parametrize("kappa", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("eta", [0.1, 0.2])
@pytest.mark.parametrize("backend", ["dense", "precompute"])
def test_coarsened_mdpo_value_and_both_student_vjps(kappa, eta, backend):
    torch.manual_seed(23)
    hs = torch.randn(7, 3, dtype=torch.float64, requires_grad=True)
    ws = torch.randn(9, 3, dtype=torch.float64, requires_grad=True)
    ht = torch.randn(7, 4, dtype=torch.float64, requires_grad=True)
    wt = torch.randn(9, 4, dtype=torch.float64, requires_grad=True)
    hq = torch.randn(7, 3, dtype=torch.float64, requires_grad=True)
    wq = torch.randn(9, 3, dtype=torch.float64, requires_grad=True)
    p = coarse(hs @ ws.T)
    q = coarse((hq @ wq.T).detach())
    t = coarse((ht @ wt.T).detach())
    log_m = (
        ((1 - eta) * q.log() + eta * t.log())
        if kappa == 0
        else ((1 - eta) * q.pow(kappa) + eta * t.pow(kappa)).log() / kappa
    )
    expected = (p * (p.log() - log_m)).sum() / eta
    actual = full_vocab_objective_result(
        hs,
        ws,
        ht,
        wt,
        hq=hq,
        wq=wq,
        spec=config(kappa, eta).loss_settings,
        backend=backend,
        token_chunk=3,
        termination_token_ids=[1, 4],
    )
    torch.testing.assert_close(actual.loss_sum / eta, expected, atol=2e-11, rtol=2e-11)
    grad = torch.autograd.grad(actual.loss_sum / eta, (hs, ws, ht, wt, hq, wq), allow_unused=True)
    reference = torch.autograd.grad(expected, (hs, ws))
    for value, wanted in zip(grad[:2], reference, strict=True):
        torch.testing.assert_close(value, wanted, atol=2e-11, rtol=2e-11)
    assert grad[2:] == (None,) * 4
    torch.testing.assert_close(actual.token_diagnostics["teacher_kl"], (p * (p.log() - t.log())).sum(-1))


def test_endpoint_and_teacher_eos_identity_invariance():
    torch.manual_seed(42)
    hs, ws, ht, wt, hq, wq = [
        torch.randn(*s, dtype=torch.float64) for s in [(4, 3), (9, 3), (4, 3), (9, 3), (4, 3), (9, 3)]
    ]
    hs.requires_grad_()
    ws.requires_grad_()
    kwargs = dict(backend="precompute", token_chunk=2, termination_token_ids=[1, 4])
    old = full_reverse_kl_result(hs, ws, ht, wt, **kwargs)
    endpoint = full_vocab_objective_result(hs, ws, ht, wt, spec=config(1.0, 1.0).loss_settings, **kwargs)
    torch.testing.assert_close(old.loss_sum, endpoint.loss_sum, atol=0, rtol=0)
    torch.testing.assert_close(
        torch.autograd.grad(old.loss_sum, hs, retain_graph=True)[0],
        torch.autograd.grad(endpoint.loss_sum, hs, retain_graph=True)[0],
        atol=0,
        rtol=0,
    )
    spec = config(1.0, 0.1).loss_settings
    one = full_vocab_objective_result(hs, ws, ht, wt, hq=hq, wq=wq, spec=spec, **kwargs)
    swapped = wt.clone()
    swapped[[1, 4]] = wt[[4, 1]]
    two = full_vocab_objective_result(hs, ws, ht, swapped, hq=hq, wq=wq, spec=spec, **kwargs)
    torch.testing.assert_close(one.loss_sum, two.loss_sum, atol=1e-12, rtol=0)
    torch.testing.assert_close(
        torch.autograd.grad(one.loss_sum, hs)[0], torch.autograd.grad(two.loss_sum, hs)[0], atol=1e-12, rtol=0
    )


def test_unsupported_objectives_stay_rejected():
    with pytest.raises(ValueError, match="termination equivalence"):
        config(1.0, 0.1, objective="bpo")
    with pytest.raises(ValueError, match="termination equivalence"):
        config(1.0, 0.1, estimator="sampled")
