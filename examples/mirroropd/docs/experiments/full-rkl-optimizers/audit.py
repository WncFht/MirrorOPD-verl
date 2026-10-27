#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent FP64 audit of full-vocabulary RKL and optimizer controls."""

import argparse
import json
from pathlib import Path

import torch

from verl.trainer.distillation.numerics import kl_rows


def oracle(z, teacher):
    lp = z.log_softmax(-1)
    lt = teacher.log_softmax(-1)[..., : z.shape[-1]]
    return (lp.exp() * (lp - lt)).sum(-1)


def run():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    checks = []
    for extra in (0, 3):
        z = torch.randn(5, 19, dtype=torch.float64, requires_grad=True)
        teacher = torch.randn(5, 19 + extra, dtype=torch.float64)
        actual, dz = kl_rows(z, teacher)
        expected = oracle(z, teacher)
        expected_g = torch.autograd.grad(expected.sum(), z)[0]
        errors = {
            "value": (actual - expected).abs().max().item(),
            "gradient": (dz - expected_g).abs().max().item(),
            "gauge": dz.sum(-1).abs().max().item(),
        }
        assert max(errors.values()) < 1e-12, errors
        checks.append({"teacher_extra_rows": extra, **errors})
    z = torch.randn(7, dtype=torch.float64)
    hessian = torch.autograd.functional.hessian(lambda x: oracle(x, z), z)
    p = z.softmax(-1)
    fisher = torch.diag(p) - p[:, None] * p[None, :]
    fisher_error = (hessian - fisher).abs().max().item()
    assert fisher_error < 1e-12

    # A cold Adam step is independent of beta1 after bias correction.
    g = torch.randn(9, 5, dtype=torch.float64)
    outputs = []
    for beta1 in (0.0, 0.5, 0.9):
        x = torch.nn.Parameter(torch.zeros_like(g))
        opt = torch.optim.AdamW([x], lr=1e-4, betas=(beta1, 0.999), weight_decay=0, eps=1e-8)
        x.grad = g.clone()
        opt.step()
        outputs.append(x.detach().clone())
    cold_error = max((x - outputs[0]).abs().max().item() for x in outputs)
    assert cold_error < 1e-15

    # Chunking optimizer work by parameter must preserve native Muon semantics.
    params = [torch.nn.Parameter(torch.randn(7, 5)), torch.nn.Parameter(torch.randn(4, 9))]
    split = [torch.nn.Parameter(p.detach().clone()) for p in params]
    kwargs = dict(lr=1e-3, momentum=0.95, nesterov=True, weight_decay=0, adjust_lr_fn="match_rms_adamw")
    joint = torch.optim.Muon(params, **kwargs)
    states = [{} for _ in split]
    for step in range(3):
        for i, (p, q) in enumerate(zip(params, split, strict=False)):
            grad = torch.randn_like(p)
            p.grad = grad.clone()
            q.grad = grad.clone()
            opt = torch.optim.Muon([q], **kwargs)
            opt.state[q] = states[i]
            opt.step()
            states[i] = opt.state[q]
        joint.step()
    chunk_error = max((p - q).abs().max().item() for p, q in zip(params, split, strict=False))
    assert chunk_error == 0
    # Enumerate every prefix of a two-token binary autoregressive model. This
    # isolates occupancy differentiation; it is not an LLM performance claim.
    features = torch.tensor([[1.0, 0.3], [0.2, 1.0], [-0.8, 0.7]], dtype=torch.float64)
    theta_teacher = torch.tensor([0.6, -0.2], dtype=torch.float64)
    target = torch.stack([torch.zeros(3, dtype=torch.float64), features @ theta_teacher], -1)

    def local_losses(x):
        logits = torch.stack([torch.zeros(3, dtype=x.dtype), features @ x], -1)
        return oracle(logits, target)

    def occupancy(x):
        prob = (features[0] @ x).sigmoid()
        return torch.stack([prob * 0 + 0.5, 0.5 * (1 - prob), 0.5 * prob])

    def semigrad(x):
        losses = local_losses(x)
        local = torch.stack([torch.autograd.grad(loss, x, create_graph=True, retain_graph=True)[0] for loss in losses])
        return (occupancy(x)[:, None] * local).sum(0)

    theta = torch.tensor([-1.0, 0.7], dtype=torch.float64, requires_grad=True)
    jac = torch.autograd.functional.jacobian(semigrad, theta)
    objective = (occupancy(theta) * local_losses(theta)).sum()
    true_gradient = torch.autograd.grad(objective, theta)[0]
    missing_occupancy = (semigrad(theta) - true_gradient).norm().item()
    curl = (jac - jac.T).norm().item()
    near_jac = torch.autograd.functional.jacobian(semigrad, theta_teacher)
    assert missing_occupancy > 1e-4 and curl > 1e-4
    assert (near_jac - near_jac.T).norm() < 1e-12
    return {
        "dtype": "float64 unless native Muon float32",
        "row_checks": checks,
        "fisher_hessian_max_abs": fisher_error,
        "cold_adam_beta1_max_abs": cold_error,
        "native_muon_parameter_split_max_abs": chunk_error,
        "enumerated_two_token_AR": {
            "semigrad_jacobian": jac.tolist(),
            "antisymmetric_norm": curl,
            "missing_occupancy_gradient_norm": missing_occupancy,
            "at_teacher_antisymmetric_norm": (near_jac - near_jac.T).norm().item(),
        },
        "scope": "Finite-dimensional algebra audit, not an LLM quality experiment.",
    }


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--out", type=Path, required=True)
    args = a.parse_args()
    result = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
