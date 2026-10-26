# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent high-precision and structural regressions for small KL/targets."""

import pytest
import torch

from verl.trainer.distillation.full_vocab import full_reverse_kl_result
from verl.trainer.distillation.numerics import kl_rows, mirror_logits_rows
from verl.trainer.distillation.objectives import mdpo_sampled_terms


def oracle(z, anchor, teacher, kappa, eta, objective):
    mp = pytest.importorskip("mpmath")
    with mp.workdps(80):

        def probability(values):
            values = [mp.exp(mp.mpf(float(v))) for v in values]
            total = sum(values)
            return [v / total for v in values]

        p, q, t = [probability(v) for v in (z, anchor, teacher)]
        e, k = mp.mpf(eta), mp.mpf(kappa)
        b = [
            (1 - e) * mp.log(qi) + e * mp.log(ti) if kappa == 0 else mp.log((1 - e) * qi**k + e * ti**k) / k
            for qi, ti in zip(q, t[: len(q)], strict=True)
        ]
        log_z = mp.log(sum(mp.exp(bi) for bi in b))
        gap = [mp.log(pi) - bi for pi, bi in zip(p, b, strict=True)]
        mdpo = sum(pi * ri for pi, ri in zip(p, gap, strict=True))
        w = p if objective == "mdpo" else q
        mu = sum(wi * ri for wi, ri in zip(w, gap, strict=True))
        g = [wi * (ri - mu) for wi, ri in zip(w, gap, strict=True)]
        value = mdpo if objective == "mdpo" else sum(wi * (ri - mu) ** 2 for wi, ri in zip(w, gap, strict=True)) / 2
        return float(value), torch.tensor([float(v) for v in g], dtype=torch.float64), float(mdpo + log_z), float(log_z)


@pytest.mark.parametrize("backend", ["dense", "precompute", "precompute_fused", "recompute", "cached", "streaming"])
def test_endpoint_small_positive_kl_and_gauge_invariance(backend):
    teacher = torch.tensor([[0.0, -23.97, -26.02]])
    expected, gradient, _, _ = oracle([0.0, -24.0, -26.0], [0.0, -24.0, -26.0], teacher[0], 0, 1, "mdpo")
    for offset in [0.0, 32.0, 128.0]:
        hidden = (torch.tensor([[0.0, -24.0, -26.0]]) + offset).requires_grad_()
        weight = torch.eye(3)
        result = full_reverse_kl_result(hidden, weight, teacher, weight, backend=backend, token_chunk=1, vocab_chunk=2)
        actual = torch.autograd.grad(result.loss_sum, hidden)[0][0]
        assert result.loss_sum > 0
        assert result.loss_sum.item() == pytest.approx(expected, rel=2e-5)
        torch.testing.assert_close(actual.double(), gradient, atol=1e-18, rtol=2e-5)


@pytest.mark.parametrize("objective", ["mdpo", "bpo"])
@pytest.mark.parametrize("kappa", [0.0, 1e-7, 0.5, 1.0])
@pytest.mark.parametrize("eta", [1e-6, 0.3, 1.0])
def test_split_target_preserves_tiny_increment(objective, kappa, eta):
    z = torch.tensor([[32.0, 22.0, 20.0]])
    t = z + torch.tensor([[0.0, 0.00003, -0.00002]])
    value, g, info = mirror_logits_rows(z, z, t, objective=objective, kappa=kappa, eta=eta, diagnostics=True)
    ref, rg, target, log_z = oracle(z[0], z[0], t[0], kappa, eta, objective)
    assert g.norm() > 0
    torch.testing.assert_close(g[0].double(), rg, atol=1e-24, rtol=3e-5)
    assert value.item() == pytest.approx(ref, rel=1e-4, abs=1e-27)
    assert info["target_kl"].item() == pytest.approx(target, rel=1e-4, abs=1e-27)
    assert info["target_log_z"].item() == pytest.approx(log_z, rel=1e-4, abs=1e-27)


@pytest.mark.parametrize("objective", ["mdpo", "bpo"])
@pytest.mark.parametrize("kappa", [0.0, 0.5, 1.0])
def test_true_self_target_and_extra_teacher_support(objective, kappa):
    z = torch.tensor([[32.0, 22.0, 20.0]], requires_grad=True)
    for extra in [False, True]:
        t = torch.cat([z.detach(), z.new_tensor([[8.0]])], -1) if extra else z.detach()
        value, g, info = mirror_logits_rows(
            z, z.detach(), t, objective=objective, kappa=kappa, eta=0.3, diagnostics=True
        )
        assert torch.count_nonzero(g) == 0
        assert torch.count_nonzero(info["target_kl"]) == 0
        if extra:
            assert info["teacher_kl"].item() > 0 and info["target_log_z"].item() < 0
        else:
            assert value.item() == 0
        torch.testing.assert_close(torch.autograd.grad(value.sum(), z)[0], torch.zeros_like(z), atol=1e-20, rtol=0)


def test_sampled_residual_survives_target_encoding():
    lp = torch.tensor([-32.0], requires_grad=True)
    lq, lt = lp.detach(), torch.tensor([-31.9999])
    eta = 1e-6
    residual = (lq - lp) + eta * (lt - lq)
    assert torch.equal(lq + eta * (lt - lq), lq)  # The former b representation lost the signal.
    loss = mdpo_sampled_terms(lp, lq, residual=residual).sum() / eta
    gradient = torch.autograd.grad(loss, lp)[0]
    torch.testing.assert_close(gradient, lq - lt, rtol=1e-6, atol=0)


def test_same_logits_dense_kl_has_zero_autograd():
    z = (torch.randn(3, 19) * 4 + 32).requires_grad_()
    value, gradient = kl_rows(z, z.detach())
    assert torch.count_nonzero(value) == torch.count_nonzero(gradient) == 0
    assert torch.count_nonzero(torch.autograd.grad(value.sum(), z)[0]) == 0


def test_bpo_without_diagnostics_never_normalizes_student(monkeypatch):
    import verl.trainer.distillation.numerics as numerics

    z = torch.randn(2, 7, requires_grad=True)
    anchor, teacher = torch.randn_like(z), torch.randn_like(z)
    original = numerics.log_softmax

    def checked(x):
        assert x is not z, "BPO without diagnostics must not compute student softmax"
        return original(x)

    monkeypatch.setattr(numerics, "log_softmax", checked)
    value, gradient, info = numerics.mirror_logits_rows(
        z, anchor, teacher, objective="bpo", kappa=0.5, eta=0.3, diagnostics=False
    )
    assert info == {} and torch.isfinite(value).all() and torch.isfinite(gradient).all()


@pytest.mark.parametrize("kappa", [0.0, 1e-7, 0.5, 1.0])
def test_teacher_mass_outside_support_preserves_finite_normalizer(kappa):
    z = torch.tensor([[32.0, 22.0, 20.0]])
    teacher = torch.cat([z - 1000, z.new_tensor([[32.0]])], -1)
    value, gradient, info = mirror_logits_rows(z, z, teacher, objective="mdpo", kappa=kappa, eta=0.3, diagnostics=True)
    ref, rg, target, logz = oracle(z[0], z[0], teacher[0], kappa, 0.3, "mdpo")
    assert torch.isfinite(value).all()
    torch.testing.assert_close(value.double(), torch.tensor([ref], dtype=torch.float64), atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(
        info["target_log_z"].double(), torch.tensor([logz], dtype=torch.float64), atol=3e-5, rtol=3e-5
    )
    assert torch.count_nonzero(gradient) == 0


@pytest.mark.parametrize(
    "kappa,eta", [(0.0, 0.999999), (0.5, 0.999999), (0.99999999, 0.3), (0.99999999, 1e-6), (0.99999999, 0.999999)]
)
def test_normalizer_and_teacher_endpoint_boundary(kappa, eta):
    anchor = torch.tensor([[32.0, 22.0, 20.0]])
    teacher = anchor + anchor.new_tensor([[0.0, 0.00003, -0.00002]])
    value, g, info = mirror_logits_rows(
        teacher, anchor, teacher, objective="mdpo", kappa=kappa, eta=eta, diagnostics=True
    )
    rv, rg, rk, rz = oracle(teacher[0], anchor[0], teacher[0], kappa, eta, "mdpo")
    assert info["target_log_z"].item() < 0
    torch.testing.assert_close(g[0].double(), rg, atol=1e-25, rtol=3e-4)
    assert value.item() == pytest.approx(rv, rel=3e-4, abs=1e-29)
    assert info["target_kl"].item() == pytest.approx(rk, rel=3e-4, abs=1e-29)
    assert info["target_log_z"].item() == pytest.approx(rz, rel=3e-4, abs=1e-29)


def test_cumulant_of_constant_has_exact_zero_value_and_gradient():
    from verl.trainer.distillation.numerics import cumulant, log_softmax

    torch.manual_seed(19)
    z = torch.randn(3, 97, requires_grad=True)
    x = torch.full_like(z, 31.0, requires_grad=True)
    value = cumulant(log_softmax(z), x)
    assert torch.count_nonzero(value) == 0
    gradients = torch.autograd.grad(value.sum(), (z, x))
    assert all(torch.count_nonzero(g) == 0 for g in gradients)
