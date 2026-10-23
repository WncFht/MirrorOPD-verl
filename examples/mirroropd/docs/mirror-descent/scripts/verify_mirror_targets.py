# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU/FP64 oracle for kappa-eta targets and conditional off-policy gradients.

This is a finite-vocabulary reference, not a production training backend.
All action expectations are enumerated. J is integrated by Gauss-Legendre quadrature.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def signal(log_ratio, kappa, eta):
    """Frozen f when fed log(T/q); refreshed f when fed log(T/p)."""
    if not 0 <= kappa <= 1 or not 0 < eta <= 1:
        raise ValueError("Require kappa in [0, 1] and eta in (0, 1]")
    if kappa == 0 or eta == 1:
        return eta * log_ratio
    x = kappa * log_ratio
    small = x.abs() < 0.5
    result = torch.empty_like(x)
    result[small] = torch.log1p(eta * torch.expm1(x[small]))
    result[~small] = torch.logaddexp(torch.full_like(x[~small], np.log1p(-eta)), x[~small] + np.log(eta))
    return result / kappa


def frozen_target(log_q, log_teacher, kappa, eta):
    log_q, log_teacher = log_q.detach(), log_teacher.detach()
    f = signal(log_teacher - log_q, kappa, eta)
    unnormalized = log_q + f
    log_z = unnormalized.logsumexp(-1, keepdim=True)
    return unnormalized - log_z, f, log_z.squeeze(-1)


def teacher_objective(log_p, log_teacher, kappa, eta):
    """J = sum T * integral_0^log(T/p) A(exp(t))*exp(-t) dt."""
    nodes, weights = np.polynomial.legendre.leggauss(64)
    nodes = torch.as_tensor(nodes, dtype=log_p.dtype, device=log_p.device)
    weights = torch.as_tensor(weights, dtype=log_p.dtype, device=log_p.device)
    r = log_teacher.detach() - log_p
    t = r[..., None] * (nodes + 1) / 2
    integral = r / 2 * (weights * signal(t, kappa, eta) / eta * (-t).exp()).sum(-1)
    return (log_teacher.detach().exp() * integral).sum()


def current_nois_objective(log_p, log_teacher, beta, kappa, eta):
    nodes, weights = np.polynomial.legendre.leggauss(64)
    nodes = torch.as_tensor(nodes, dtype=log_p.dtype, device=log_p.device)
    weights = torch.as_tensor(weights, dtype=log_p.dtype, device=log_p.device)
    y = log_p - log_teacher.detach()
    u = y[..., None] * (nodes + 1) / 2
    integral = y / 2 * (weights * -signal(-u, kappa, eta) / eta).sum(-1)
    return (beta.detach() * integral).sum()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    rng = torch.Generator().manual_seed(2310)
    errors = {}
    comparisons = 0

    def check(name, actual, expected, tolerance=2e-10):
        nonlocal comparisons
        error = (actual - expected).abs().max().item()
        comparisons += 1
        errors[name] = max(errors.get(name, 0), error)
        assert error < tolerance, (name, error)

    def grad(loss, z):
        return torch.autograd.grad(loss, z, retain_graph=True)[0]

    def logits():
        return torch.randn(7, generator=rng, dtype=torch.float64)

    for _ in range(6):
        lq, lt, lb = [logits().log_softmax(-1) for _ in range(3)]
        q, beta = lq.exp(), lb.exp()
        for kappa in (0.0, 1e-7, 0.25, 0.5, 1.0):
            for eta in (0.2, 0.5, 0.8, 1.0):
                z = logits().requires_grad_()
                lp = z.log_softmax(-1)
                p = lp.exp()
                lm, f, log_z = frozen_target(lq, lt, kappa, eta)
                check("target_normalized", lm.exp().sum(), torch.ones_like(log_z))
                U = lp - lq
                md = (p * (U - f)).sum() / eta
                fit = (p * (lp - lm)).sum() / eta
                check("MDPO_value_constant", fit - md, log_z / eta)
                check("MDPO_fit_gradient", grad(md, z), grad(fit, z))
                coefficient = (U - f) / eta
                is_score = (beta * ((p / beta) * coefficient).detach() * lp).sum()
                ratio_form = ((p / beta) * coefficient.detach() * beta).sum()
                check("MDPO_action_IS", grad(is_score, z), grad(fit, z))
                check("MDPO_ratio_autograd", grad(ratio_form, z), grad(fit, z))
                centered = f / eta - (q * f).sum() / eta
                centered_score = (p * (U / eta - centered)).detach().mul(lp).sum()
                check("MDPO_center_cancels", grad(centered_score, z), grad(fit, z))

                Acur = signal(lt - lp, kappa, eta) / eta
                current_score = -(p * Acur).detach().mul(lp).sum()
                J = teacher_objective(lp, lt, kappa, eta)
                check("current_J_gradient", grad(current_score, z), grad(J, z))
                current_m, _, _ = frozen_target(lp, lt, kappa, eta)
                current_target_proxy = (p * (lp - current_m)).sum() / eta
                check("refreshed_detached_target_J_gradient", grad(current_target_proxy, z), grad(J, z))
                current_nois_score = -(beta * Acur.detach() * lp).sum()
                H = current_nois_objective(lp, lt, beta, kappa, eta)
                check("current_NoIS_potential", grad(current_nois_score, z), grad(H, z))
                if kappa == 0 or eta == 1:
                    DT = (p * (lp - lt)).sum()
                    Dq = (p * U).sum()
                    check("geometric_MDPO", grad(fit, z), grad(DT + (1 / eta - 1) * Dq, z))
                    check("J_reverse_KL_endpoint", J, DT)
                    check("current_NoIS_log_square_endpoint", H, (beta * (lp - lt).square()).sum() / 2)
                if kappa == 1:
                    mixture = (1 - eta) * q + eta * lt.exp()
                    check("arithmetic_target", lm.exp(), mixture)
                    if eta < 1:
                        mix = ((1 - eta) * p + eta * lt.exp()).log()
                        js = (1 - eta) * (p * (lp - mix)).sum() + eta * (lt.exp() * (lt - mix)).sum()
                        check("J_scaled_JS", J, js / (eta * (1 - eta)))

                R = U - f
                C_R = R - (q * R).sum()
                G = f - (q * f).sum()
                part = (q * (R + log_z).square()).sum() / 2
                bpo = (q * C_R.square()).sum() / (2 * eta)
                mean = (q * (U - G).square()).sum() / 2
                Ks = -(q * U).sum()
                check("Mean_BPO_identity", mean, eta * bpo + Ks.square() / 2)
                for name, loss, residual, scale in (
                    ("Part", part, R + log_z, 1.0),
                    ("BPO", bpo, C_R, 1 / eta),
                    ("Mean", mean, U - G, 1.0),
                ):
                    sampled = (beta * ((q / beta) * residual * scale).detach() * lp).sum()
                    check(name + "_q_over_beta", grad(sampled, z), grad(loss, z))
                whole_nois = (beta * R.square()).sum() / (2 * eta)
                nois_score = (beta * (R / eta).detach() * lp).sum()
                check("whole_NoIS_regression", grad(whole_nois, z), grad(nois_score, z))

                zm = lm.detach().clone().requires_grad_()
                lpm = zm.log_softmax(-1)
                Rm = lpm - lq - f
                target_losses = {
                    "MDPO": (lpm.exp() * (lpm - lm)).sum() / eta,
                    "Part": (q * (Rm + log_z).square()).sum() / 2,
                    "BPO": (q * (Rm - (q * Rm).sum()).square()).sum() / (2 * eta),
                }
                for name, loss in target_losses.items():
                    check(name + "_stationary_at_m", grad(loss, zm), torch.zeros_like(zm))
                zq = lq.clone().requires_grad_()
                lpq = zq.log_softmax(-1)
                initial_md = (lpq.exp() * (lpq - lm)).sum() / eta
                initial_J = teacher_objective(lpq, lt, kappa, eta)
                check("initial_MDPO_equals_J", grad(initial_md, zq), grad(initial_J, zq))

    q = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
    T = torch.tensor([0.1, 0.2, 0.7], dtype=torch.float64)
    beta = torch.tensor([0.7, 0.2, 0.1], dtype=torch.float64)
    eta, kappa = 0.3, 0.5
    lm, f, log_z = frozen_target(q.log(), T.log(), kappa, eta)
    z = lm.clone().requires_grad_()
    lp = z.log_softmax(-1)
    R = lp - q.log() - f
    raw_nois = (beta * R.square()).sum() / (2 * eta)
    G = f - (q * f).sum()
    mean = (q * (lp - q.log() - G).square()).sum() / 2
    counterexamples = {
        "whole_NoIS_gradient_at_m": grad(raw_nois, z).norm().item(),
        "Mean_gradient_at_m": grad(mean, z).norm().item(),
    }
    assert min(counterexamples.values()) > 1e-5

    z = torch.tensor([1.2, -0.5, 0.3], dtype=torch.float64, requires_grad=True)
    lp = z.log_softmax(-1)
    reward = ((lm - lp) / eta).detach()
    ratio = (lp - beta.log()).exp()
    exact = (lp.exp() * (lp - lm)).sum() / eta
    clipped = (beta * torch.maximum(-ratio * reward, -ratio.clamp(0.8, 1.2) * reward)).sum()
    counterexamples["PPO_clip_gradient_difference"] = (grad(clipped, z) - grad(exact, z)).norm().item()
    assert counterexamples["PPO_clip_gradient_difference"] > 1e-5
    refreshed_log_h = lp + signal(T.log() - lp, kappa, eta)
    differentiable_target = refreshed_log_h.log_softmax(-1)
    wrong_loss = (lp.exp() * (lp - differentiable_target)).sum() / eta
    J = teacher_objective(lp, T.log(), kappa, eta)
    counterexamples["refreshed_target_without_detach_gradient_difference"] = (
        (grad(wrong_loss, z) - grad(J, z)).norm().item()
    )
    assert counterexamples["refreshed_target_without_detach_gradient_difference"] > 1e-5
    qz, tz = [v.log().requires_grad_() for v in (q, T)]
    frozen, _, _ = frozen_target(qz.log_softmax(-1), tz.log_softmax(-1), kappa, eta)
    loss = (lp.exp() * (lp - frozen)).sum() / eta
    assert torch.autograd.grad(loss, (qz, tz), allow_unused=True) == (None, None)

    first_gradients = []
    for eta in (0.2, 0.5, 1.0):
        zq = q.log().requires_grad_()
        lpq = zq.log_softmax(-1)
        lm, _, _ = frozen_target(q.log(), T.log(), 0.0, eta)
        first_gradients.append(grad((lpq.exp() * (lpq - lm)).sum() / eta, zq))
    for gradient in first_gradients[1:]:
        check("kappa0_single_update_eta_independent", gradient, first_gradients[0])
    result = {
        "scope": "CPU FP64 finite-vocabulary identities; not a language-model training or quality result",
        "seed": 2310,
        "random_cases": 120,
        "comparisons": comparisons,
        "maximum_errors": errors,
        "counterexamples": counterexamples,
        "anchor_teacher_detached": True,
        "J_oracle": "64-point Gauss-Legendre quadrature in log probability ratio",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
