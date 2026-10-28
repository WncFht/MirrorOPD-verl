# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Enumerate fixed-prefix OPD and mirror-descent objectives on CPU in FP64."""

import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    generator = torch.Generator(device="cpu").manual_seed(23)
    errors = {}
    comparisons = 0

    def grad(loss, z):
        return torch.autograd.grad(loss, z, retain_graph=True)[0]

    def check(name, actual, expected):
        nonlocal comparisons
        comparisons += 1
        error = (actual - expected).abs().max().item()
        errors[name] = max(errors.get(name, 0.0), error)
        assert error < 2e-12, (name, error)

    def randlog():
        return torch.randn(6, generator=generator, dtype=torch.float64).log_softmax(-1)

    for _ in range(20):
        z = torch.randn(6, generator=generator, dtype=torch.float64, requires_grad=True)
        lp = z.log_softmax(-1)
        p = lp.exp()
        lq, lt, lb = randlog(), randlog(), randlog()
        q, beta = lq.exp(), lb.exp()
        eta = 0.1 + torch.rand((), generator=generator).item()
        c = 1 / eta
        U, r = lp - lq, lt - lq
        Dq, DT = (p * U).sum(), (p * (lp - lt)).sum()
        for current in (False, True):
            advantage = lt - lp if current else r
            gamma = c + int(current)
            for use_is in (False, True):
                nu = p if use_is else beta
                surrogate = -((nu * advantage).detach() * lp).sum()
                if use_is:
                    potential = DT if current else DT - Dq
                elif current:
                    potential = 0.5 * (beta * (lp - lt).square()).sum()
                else:
                    potential = -(beta * r * lp).sum()
                name = f"current={current},is={use_is}"
                check("base_" + name, grad(surrogate, z), grad(potential, z))
                check("exact_KL_" + name, grad(surrogate + c * Dq, z), grad(potential + c * Dq, z))
                residual = gamma * U - r
                residual_surrogate = ((nu * residual).detach() * lp).sum()
                residual_potential = potential + c * Dq if use_is else (beta * residual.square()).sum() / (2 * gamma)
                check("whole_residual_" + name, grad(residual_surrogate, z), grad(residual_potential, z))
            log_m = (lq + r / gamma).log_softmax(-1)
            target_kl = gamma * (p * (lp - log_m)).sum()
            check("geometric_target_" + str(current), grad(DT + (gamma - 1) * Dq, z), grad(target_kl, z))

            # Verify the normalized equation and variance objectives at their common target.
            zm = log_m.detach().clone().requires_grad_()
            lpm = zm.log_softmax(-1)
            log_z = torch.logsumexp(lq + r / gamma, dim=0)
            R = lpm - lq - r / gamma
            part = gamma / 2 * (beta * (R + log_z).square()).sum()
            centered = R - (beta * R).sum()
            bpo = gamma / 2 * (beta * centered.square()).sum()
            check("part_target_gradient_" + str(current), grad(part, zm), torch.zeros_like(zm))
            check("bpo_target_gradient_" + str(current), grad(bpo, zm), torch.zeros_like(zm))

        K_s = -(q * U).sum()
        G = eta * (r - (q * r).sum())
        Rb = U - eta * r
        C_Rb = Rb - (q * Rb).sum()
        bpo = (q * C_Rb.square()).sum() / (2 * eta)
        mean = (q * (U - G).square()).sum() / 2
        check("mean_BPO_value", mean, eta * bpo + K_s.square() / 2)
        check("mean_BPO_gradient", grad(mean, z), grad(eta * bpo + K_s.square() / 2, z))
        raw = (beta * (lp - lt)).sum()
        forward = (beta * (lb - lp)).sum()
        check("literal_old_weight_current_gap", grad(raw, z), -grad(forward, z))
        check("full_KL_logit_derivative", grad(DT, z), p * (lp - lt - DT))
        for measure, nu in (("p", p), ("beta", beta)):
            shifted = -((nu * (lt - lp + 0.37)).detach() * lp).sum()
            base = -((nu * (lt - lp)).detach() * lp).sum()
            expected = grad(-0.37 * (nu.detach() * lp).sum(), z)
            check("baseline_change_" + measure, grad(shifted - base, z), expected)
            if measure == "p":
                check("baseline_p_cancels", expected, torch.zeros_like(z))

    q = torch.tensor([0.5, 0.5], dtype=torch.float64)
    teacher = torch.tensor([0.8, 0.2], dtype=torch.float64)
    r = (teacher / q).log()
    boundary = []
    for epsilon in (1e-2, 1e-6, 1e-12):
        p = torch.tensor([1 - epsilon, epsilon], dtype=torch.float64)
        value = -(q * r * p.log()).sum() + (p * (p / q).log()).sum()
        boundary.append({"epsilon": epsilon, "frozen_nois_exact_KL_eta_1": value.item()})
    assert all(
        a["frozen_nois_exact_KL_eta_1"] > b["frozen_nois_exact_KL_eta_1"]
        for a, b in zip(boundary[:-1], boundary[1:], strict=True)
    )
    result = {
        "device": "cpu",
        "dtype": "float64",
        "seed": 23,
        "random_cases": 20,
        "gradient_and_value_comparisons": comparisons,
        "maximum_errors": errors,
        "boundary_example": boundary,
        "scope": "Finite-vocabulary identities and counterexample; no model training or quality claim.",
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
