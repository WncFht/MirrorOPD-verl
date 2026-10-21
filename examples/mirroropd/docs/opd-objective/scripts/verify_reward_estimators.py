# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Finite-action audits for the research note; CPU only, no training pipeline."""

import argparse
import ast
import importlib.util
import json
from pathlib import Path

import torch


def variance(vectors, probability):
    mean = (probability[:, None] * vectors).sum(0)
    return (probability * (vectors - mean).square().sum(-1)).sum().item()


def audit_case(z, log_q, k, mu):
    z = z.detach().requires_grad_()
    lp = z.log_softmax(-1)
    p = lp.exp()
    ell = lp - log_q
    exact_loss = (p * ell).sum()
    exact = torch.autograd.grad(exact_loss, z, retain_graph=True)[0]
    ids = log_q.topk(k).indices
    head_mask = torch.zeros_like(p, dtype=torch.bool)
    head_mask[ids] = True
    score = torch.eye(len(p), dtype=z.dtype) - p[None, :]
    token_gradient = ell[:, None] * score
    head = (p[ids, None] * token_gradient[ids]).sum(0)
    tau_p = 1 - p[ids].sum()
    tau_q = 1 - log_q[ids].exp().sum()
    c = (tau_p / tau_q).log()
    tail_score = -(p[ids, None] * score[ids]).sum(0) / tau_p
    bucket = c * tail_score
    coarse = head + tau_p * bucket
    coarse_loss = (p[ids] * ell[ids]).sum() + tau_p * c
    coarse_autograd = torch.autograd.grad(coarse_loss, z, retain_graph=True)[0]
    cond_lp = lp[ids].log_softmax(-1)
    cond_lq = log_q[ids].log_softmax(-1)
    baseline_topk = (cond_lp.exp() * (cond_lp - cond_lq)).sum()
    outside = (~head_mask)[:, None]
    estimators = {
        "sampled": token_gradient,
        "vopd_full": (ell - exact_loss)[:, None] * score,
        "vopd_teacher_topk": (ell - baseline_topk)[:, None] * score,
        "vopd_coarse": (ell - coarse_loss)[:, None] * score,
        "hets": head + outside * token_gradient,
        "resopd": coarse + outside * (token_gradient - bucket),
        "sc_ta": coarse + outside * ((ell - c)[:, None] * score),
        "full": exact.expand_as(score),
    }
    errors = {name: ((p[:, None] * g).sum(0) - exact).abs().max().item() for name, g in estimators.items()}
    errors["ta_coarse_autograd"] = (coarse - coarse_autograd).abs().max().item()
    errors["ta_chain_rule_value"] = abs(
        (exact_loss - coarse_loss - tau_p * ((lp[~head_mask] - tau_p.log()).exp() * (ell[~head_mask] - c)).sum()).item()
    )
    rho = p / mu
    off_policy = {
        "vopd_full_is": rho[:, None] * estimators["vopd_full"],
        "hets_tail_is": head + outside * rho[:, None] * token_gradient,
        "resopd_tail_is": coarse + outside * rho[:, None] * (token_gradient - bucket),
        "sc_ta_tail_is": coarse + outside * rho[:, None] * ((ell - c)[:, None] * score),
    }
    for name, g in off_policy.items():
        errors[name] = ((mu[:, None] * g).sum(0) - exact).abs().max().item()
    return dict(
        errors=errors,
        variances={name: variance(g, p) for name, g in estimators.items()},
        ta_gradient_bias=(coarse - exact).norm().item(),
        resopd_sc_ta_max_per_action_difference=(estimators["resopd"] - estimators["sc_ta"]).abs().max().item(),
    ), (z, lp, log_q, ids, estimators["resopd"])


def streaming_audit():
    """Independent two-pass vocabulary tiling, not a benchmark or production kernel."""
    gen = torch.Generator().manual_seed(123)
    xs = torch.randn(3, 5, dtype=torch.float64, generator=gen, requires_grad=True)
    xt = torch.randn(3, 7, dtype=torch.float64, generator=gen)
    ws = torch.randn(17, 5, dtype=torch.float64, generator=gen, requires_grad=True)
    wt = torch.randn(17, 7, dtype=torch.float64, generator=gen)
    lp, lq = (xs @ ws.T).log_softmax(-1), (xt @ wt.T).log_softmax(-1)
    exact = (lp.exp() * (lp - lq)).sum(-1)
    expected_dx, expected_dw = torch.autograd.grad(exact.sum(), (xs, ws))
    with torch.no_grad():
        mp, mq = torch.full((3,), -torch.inf), torch.full((3,), -torch.inf)
        sp, sq, up = [torch.zeros(3, dtype=torch.float64) for _ in range(3)]
        for start in range(0, 17, 4):
            zp, zq = xs @ ws[start : start + 4].T, xt @ wt[start : start + 4].T
            next_mp, next_mq = torch.maximum(mp, zp.max(-1).values), torch.maximum(mq, zq.max(-1).values)
            scale_p, scale_q = (mp - next_mp).exp(), (mq - next_mq).exp()
            ep, eq = (zp - next_mp[:, None]).exp(), (zq - next_mq[:, None]).exp()
            sp, sq = scale_p * sp + ep.sum(-1), scale_q * sq + eq.sum(-1)
            up = scale_p * up + (ep * (zp - zq)).sum(-1)
            mp, mq = next_mp, next_mq
        logzp, logzq = mp + sp.log(), mq + sq.log()
        kl = up / sp - logzp + logzq
        dx, dw = torch.zeros_like(xs), torch.zeros_like(ws)
        for start in range(0, 17, 4):
            lp_chunk = xs @ ws[start : start + 4].T - logzp[:, None]
            lq_chunk = xt @ wt[start : start + 4].T - logzq[:, None]
            dz = lp_chunk.exp() * (lp_chunk - lq_chunk - kl[:, None])
            dx += dz @ ws[start : start + 4]
            dw[start : start + 4] = dz.T @ xs
        errors = {
            "value": (kl - exact).abs().max().item(),
            "student_hidden_grad": (dx - expected_dx).abs().max().item(),
            "student_head_grad": (dw - expected_dw).abs().max().item(),
        }
    assert max(errors.values()) < 1e-10, errors
    return errors


def audit_ema_primitive(path):
    """Check the reviewed search/ kernel in isolation, without importing its RL stack."""
    names = {"compute_memory_efficient_kl", "compute_not_in_topk_mask"}
    nodes = [
        node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert len(nodes) == 2
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    fn = namespace["compute_memory_efficient_kl"]
    z = torch.tensor([0.35, 0.30, 0.20, 0.15], dtype=torch.float64).log().requires_grad_()
    q = torch.tensor([0.60, 0.25, 0.10, 0.05], dtype=torch.float64)
    p, lp = z.softmax(-1), z.log_softmax(-1)
    ids = torch.tensor([0, 1])
    exact = torch.autograd.grad((p * (lp - q.log())).sum(), z, retain_graph=True)[0]
    mass_grad = torch.autograd.grad(p[ids].sum(), z, retain_graph=True)[0]
    gradients = []
    for action in range(4):
        head, tail = fn(
            z[ids].reshape(1, 1, 2),
            z.logsumexp(-1).reshape(1, 1),
            q[ids].log().reshape(1, 1, 2),
            q.log().logsumexp(-1).reshape(1, 1),
            kl_type="full_reverse",
            use_tail_sampling=True,
            actor_topk_indices=ids.reshape(1, 1, 2),
            sampled_indices=torch.tensor([[action]]),
            log_prob=lp[action].reshape(1, 1),
            ref_log_prob=q[action].log().reshape(1, 1),
        )
        gradients.append(torch.autograd.grad((head + tail).sum(), z, retain_graph=True)[0])
    mean = (p[:, None] * torch.stack(gradients)).sum(0)
    residual = mean - exact
    assert (residual - mass_grad).abs().max() < 1e-10
    return dict(
        scope="Reviewed EMA-PG search/ primitive only; ordinary differentiable inputs, no outer task loss",
        max_error_to_full_rkl=residual.abs().max().item(),
        extra_support_mass_gradient=mass_grad.tolist(),
        max_error_after_subtracting_support_mass_gradient=(residual - mass_grad).abs().max().item(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resopd-source", type=Path, help="Optional reviewed authors resopd.py; never downloaded here")
    parser.add_argument("--ema-source", type=Path, help="Optional reviewed EMA-PG search/core_algos.py primitive")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    kernel = None
    if args.resopd_source:
        spec = importlib.util.spec_from_file_location("reviewed_resopd", args.resopd_source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        kernel = module.resopd_loss
    rows, source_errors = [], []
    for seed in range(16):
        gen = torch.Generator().manual_seed(seed)
        z = torch.randn(7, dtype=torch.float64, generator=gen)
        lq = torch.randn(7, dtype=torch.float64, generator=gen).log_softmax(-1)
        mu = torch.randn(7, dtype=torch.float64, generator=gen).softmax(-1)
        for k in [1, 2, 4, 6]:
            row, (leaf, lp, log_q, ids, reference) = audit_case(z, lq, k, mu)
            row.update(seed=seed, support_size=k)
            rows.append(row)
            if kernel:
                for action in range(7):
                    out = kernel(
                        leaf[None, :],
                        log_q[ids][None, :],
                        ids[None, :],
                        log_q[action].reshape(1, 1),
                        torch.tensor([[action]]),
                    )
                    actual = torch.autograd.grad(out["distillation_losses"].sum(), leaf, retain_graph=True)[0]
                    source_errors.append((actual - reference[action]).abs().max().item())
    maximum = max(max(row["errors"].values()) for row in rows)
    assert maximum < 1e-10, maximum
    if source_errors:
        assert max(source_errors) < 1e-10, max(source_errors)
    example, _ = audit_case(
        torch.tensor([0.9, 0.1], dtype=torch.float64).log(),
        torch.tensor([0.5, 0.5], dtype=torch.float64).log(),
        1,
        torch.tensor([0.9, 0.1], dtype=torch.float64),
    )
    assert example["variances"]["vopd_full"] > example["variances"]["sampled"]
    result = dict(
        cases=len(rows),
        max_identity_error=maximum,
        author_resopd_checked=kernel is not None,
        author_resopd_max_error=max(source_errors) if source_errors else None,
        ema_primitive_audit=audit_ema_primitive(args.ema_source) if args.ema_source else None,
        streaming_reverse_kl_errors=streaming_audit(),
        baseline_variance_counterexample=dict(p=[0.9, 0.1], q=[0.5, 0.5], variances=example["variances"]),
        resopd_vs_sc_ta_max_action_difference=max(r["resopd_sc_ta_max_per_action_difference"] for r in rows),
        rows=rows,
        scope="Finite fixed-prefix CPU math and optional author kernel check; no LLM/GPU or accuracy claim.",
    )
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
