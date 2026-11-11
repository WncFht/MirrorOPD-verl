# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU checks of proposed teacher construction; no LLM or rollout claims.

Four synthetic domains share a linear softmax model. All teachers and OPD
students use the identical parameterization and initialization. Exact action
expectations isolate objectives from finite-sample noise and state coverage.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch


def kl(p, q):
    return (p * (p.log() - q.log())).sum(-1)


def pearson_weights(scores, radius=1.0, max_weight=4.0, floor=0.05):
    """Solve the finite-candidate reward maximization with a Pearson budget."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Expected a nonempty finite score vector")
    if radius < 0 or not 0 <= floor < 1 < max_weight:
        raise ValueError("Invalid weight constraints")
    if radius == 0 or np.ptp(scores) < 1e-14:
        return np.ones_like(scores)

    def normalized(alpha):
        lo = float(scores.min() - max_weight / alpha)
        hi = float(scores.max() + max_weight / alpha)
        for _ in range(64):
            center = (lo + hi) / 2
            weights = np.clip(1 + alpha * (scores - center), floor, max_weight)
            if weights.mean() > 1:
                lo = center
            else:
                hi = center
        return np.clip(1 + alpha * (scores - (lo + hi) / 2), floor, max_weight)

    upper = 1.0
    while upper < 2**24 and np.mean((normalized(upper) - 1) ** 2) < radius:
        upper *= 2
    lower, best = 0.0, np.ones_like(scores)
    for _ in range(48):
        alpha = (lower + upper) / 2
        weights = normalized(alpha)
        if np.mean((weights - 1) ** 2) <= radius:
            lower, best = alpha, weights
        else:
            upper = alpha
    if abs(best.mean() - 1) > 1e-7:
        raise ArithmeticError("Weight normalization failed")
    return best


def matched_success_check():
    """Compare two full-support teachers of identical binary task success."""
    base = torch.tensor([0.30, 0.01, 0.45, 0.24], dtype=torch.float64)
    near = base.clone()
    near[:2] *= 0.8 / near[:2].sum()
    near[2:] *= 0.2 / near[2:].sum()
    far = near.clone()
    far[:2] = torch.tensor([0.04, 0.76], dtype=torch.float64)
    result = {}
    for name, target in [("base-style", near), ("rare-expert-style", far)]:
        logits = torch.nn.Parameter(base.log().clone())
        optimizer = torch.optim.Adam([logits], lr=0.03)
        for _ in range(64):
            loss = kl(logits.softmax(-1), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        student = logits.detach().softmax(-1)
        initial = float(kl(base, target))
        residual = float(kl(student, target))
        result[name] = {
            "teacher_success": float(target[:2].sum()),
            "teacher_to_base_kl": float(kl(target, base)),
            "initial_rkl": initial,
            "residual_rkl": residual,
            "residual_fraction": residual / initial,
            "student_success": float(student[:2].sum()),
        }
    return result


def identity_checks():
    base = np.array([0.30, 0.01, 0.45, 0.24], dtype=np.float64)
    reward = np.array([1.0, 1.0, 0.0, 0.0])
    eta = 2.0
    target = base * np.exp(eta * reward)
    target /= target.sum()
    weight = target / base - 1
    score = np.eye(4) - base[None, :]
    ce_update = target - base
    centered_update = ((base * weight)[:, None] * score).sum(0)
    # Independent-proposal MH, initialized at a rare expert answer.
    proposal = np.array([0.15, 0.70, 0.10, 0.05])
    transition = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            if i != j:
                accept = min(1.0, target[j] * proposal[i] / (target[i] * proposal[j]))
                transition[i, j] = proposal[j] * accept
        transition[i, i] = 1 - transition[i].sum()
    distribution = np.array([0.0, 1.0, 0.0, 0.0])
    path = []
    for step in range(129):
        positive = distribution > 0
        path.append(float(np.sum(distribution[positive] * np.log(distribution[positive] / target[positive]))))
        distribution = distribution @ transition
    # Exact enumeration of two independent draws verifies RLOO expectation.
    rloo = np.zeros(4)
    self_centered = np.zeros(4)
    for i in range(4):
        for j in range(4):
            mass = base[i] * base[j]
            rloo += mass * (reward[i] - reward[j]) * score[i]
            self_centered += mass * (reward[i] - (reward[i] + reward[j]) / 2) * score[i]
    exact = base * (reward - base @ reward)
    # Exact finite-pool marginal, with Pearson and maximum-weight constraints.
    pearson = np.zeros(4)
    pool_size, radius, max_weight, floor = 4, 1.0, 4.0, 0.05
    max_pool_chi = 0.0
    for encoded in range(4**pool_size):
        indices = [(encoded // 4**j) % 4 for j in range(pool_size)]
        r = reward[indices]
        fraction = r.mean()
        weights = np.ones(pool_size)
        if 0 < fraction < 1:
            shift = min(
                np.sqrt(radius * fraction * (1 - fraction)),
                (1 - floor) * (1 - fraction),
                (max_weight - 1) * fraction,
            )
            weights[r == 1] += shift / fraction
            weights[r == 0] -= shift / (1 - fraction)
        max_pool_chi = max(max_pool_chi, float(np.mean((weights - 1) ** 2)))
        mass = float(np.prod(base[indices]))
        for i, action in enumerate(indices):
            pearson[action] += mass * weights[i] / pool_size
    marginal_chi = float(np.sum(base * (pearson / base - 1) ** 2))
    binary_weights = pearson_weights([1, 0, 0, 0])
    expected_weights = np.array([1 + math.sqrt(3)] + [1 - 1 / math.sqrt(3)] * 3)
    assert np.max(np.abs(binary_weights - expected_weights)) < 1e-8
    general_scores = np.array([0.0, 0.1, 0.4, 0.7, 1.0])
    general_weights = pearson_weights(general_scores)
    checks = {
        "centered_mean_error": abs(float(base @ weight)),
        "sft_rewrite_error": float(np.max(np.abs(ce_update - centered_update))),
        "mh_stationarity_error": float(np.max(np.abs(target @ transition - target))),
        "mh_max_kl_increase": float(np.max(np.diff(path))),
        "mh_final_total_variation": float(np.abs(distribution - target).sum() / 2),
        "rloo_expected_gradient_error": float(np.max(np.abs(rloo - exact))),
        "self_including_mean_half_gradient_error": float(np.max(np.abs(self_centered - exact / 2))),
        "density_ratio_max": float((target / base).max()),
        "density_ratio_bound": float(np.exp(eta)),
        "chi_square": float(base @ weight**2),
        "hard_correct_minimum_kl": float(-np.log(base @ reward)),
        "pearson_pool_max_chi_square": max_pool_chi,
        "pearson_marginal_chi_square": marginal_chi,
        "pearson_marginal_success": float(pearson @ reward),
        "pearson_marginal_mass_error": float(abs(pearson.sum() - 1)),
        "pearson_general_weight_mean_error": float(abs(general_weights.mean() - 1)),
        "pearson_general_weight_chi_square": float(np.mean((general_weights - 1) ** 2)),
        "mh_kl_path": path,
    }
    for key in ["centered_mean_error", "sft_rewrite_error", "mh_stationarity_error", "rloo_expected_gradient_error"]:
        assert checks[key] < 1e-12, (key, checks[key])
    assert checks["mh_max_kl_increase"] < 1e-12
    assert checks["density_ratio_max"] <= checks["density_ratio_bound"]
    assert marginal_chi <= radius + 1e-12
    assert checks["pearson_marginal_mass_error"] < 1e-12
    return checks


def run_seed(seed, teacher_steps, opd_steps):
    torch.manual_seed(seed)
    dtype = torch.float64
    # Domain-specific bias plus shared features; held-out features are fresh.
    x = torch.randn(4, 128, 6, dtype=dtype)
    domains = torch.eye(4, dtype=dtype)[:, None, :].expand(4, 128, 4)
    x = torch.cat([domains, x], dim=-1).reshape(-1, 10)
    base_w = torch.randn(10, 4, dtype=dtype) * 0.12
    base_w[:4] += torch.tensor([0.0, -3.0, 0.4, 0.1], dtype=dtype)
    base = (x @ base_w).softmax(-1)
    reward = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=dtype)
    correct = base * reward
    correct /= correct.sum(-1, keepdim=True)
    raw = correct.clone()
    raw[:, 0], raw[:, 1] = 0.03, 0.97
    # Soft tilting and hard conditioning intentionally differ in task strength.
    tilted = base * (2.0 * reward).exp()
    tilted /= tilted.sum(-1, keepdim=True)
    # Exact marginal of K=16 candidate selection for binary rewards.
    z = (base * reward).sum(-1)
    success = torch.zeros_like(z)
    for count in range(17):
        fraction = count / 16
        shift = min(math.sqrt(fraction * (1 - fraction)), 0.95 * (1 - fraction), 3 * fraction)
        mass = math.comb(16, count) * z**count * (1 - z) ** (16 - count)
        success += mass * (fraction + shift)
    pearson = base.clone()
    pearson[:, :2] *= (success / z)[:, None]
    pearson[:, 2:] *= ((1 - success) / (1 - z))[:, None]
    train = torch.tensor([d * 128 + i for d in range(4) for i in range(64)])
    held = torch.tensor([d * 128 + i for d in range(4) for i in range(64, 128)])
    rows = []
    teachers = {}
    for method in [
        "raw-expert-sft",
        "base-correct-sft",
        "bounded-tilt-sft",
        "centered-pearson-sft",
        "direct-reward-kl",
    ]:
        w = torch.nn.Parameter(base_w.clone())
        optimizer = torch.optim.Adam([w], lr=0.04)
        target = {
            "raw-expert-sft": raw,
            "base-correct-sft": correct,
            "bounded-tilt-sft": tilted,
            "centered-pearson-sft": pearson,
        }.get(method)
        for _ in range(teacher_steps):
            lp = (x[train] @ w).log_softmax(-1)
            if method == "direct-reward-kl":
                loss = -(lp.exp() * reward).sum(-1).mean() + 0.5 * kl(lp.exp(), base[train]).mean()
            else:
                loss = -(target[train] * lp).sum(-1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        teachers[method] = (x @ w.detach()).softmax(-1)
        student = torch.nn.Parameter(base_w.clone())
        optimizer = torch.optim.Adam([student], lr=0.03)
        teacher = teachers[method]
        initial_kl = kl(base, teacher)
        for update in range(opd_steps + 1):
            with torch.no_grad():
                p = (x @ student).softmax(-1)
                residual = kl(p, teacher)
                for domain in range(4):
                    indices = held[domain * 64 : (domain + 1) * 64]
                    rows.append(
                        {
                            "seed": seed,
                            "method": method,
                            "domain": domain,
                            "update": update,
                            "base_success": float((base[indices] * reward).sum(-1).mean()),
                            "teacher_success": float((teacher[indices] * reward).sum(-1).mean()),
                            "student_success": float((p[indices] * reward).sum(-1).mean()),
                            "teacher_to_base_kl": float(kl(teacher[indices], base[indices]).mean()),
                            "initial_rkl": float(initial_kl[indices].mean()),
                            "residual_rkl": float(residual[indices].mean()),
                            "residual_fraction": float(residual[indices].mean() / initial_kl[indices].mean()),
                            "data_chi_square": None
                            if target is None
                            else float((((target[indices] / base[indices] - 1) ** 2) * base[indices]).sum(-1).mean()),
                        }
                    )
            if update == opd_steps:
                break
            p = (x[train] @ student).softmax(-1)
            loss = kl(p, teacher[train]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return rows, float((teachers["bounded-tilt-sft"] - teachers["direct-reward-kl"]).abs().max())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--teacher-steps", type=int, default=120)
    parser.add_argument("--opd-steps", type=int, default=64)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "results.csv").exists():
        raise FileExistsError("Use a fresh output directory")
    torch.set_num_threads(1)
    rows, discrepancies = [], []
    for seed in range(args.seeds):
        batch, discrepancy = run_seed(seed, args.teacher_steps, args.opd_steps)
        rows.extend(batch)
        discrepancies.append(discrepancy)
    with (args.output / "results.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for method in sorted({row["method"] for row in rows}):
        final = [r for r in rows if r["method"] == method and r["update"] == args.opd_steps]
        summary[method] = {
            field: float(np.mean([r[field] for r in final]))
            for field in [
                "base_success",
                "teacher_success",
                "student_success",
                "initial_rkl",
                "residual_rkl",
                "residual_fraction",
            ]
        }
    result = {
        "settings": vars(args) | {"output": str(args.output), "dtype": "float64", "device": "cpu"},
        "identity_checks": identity_checks(),
        "matched_success_check": matched_success_check(),
        "summary": summary,
        "bounded_sft_vs_direct_max_probability_error": max(discrepancies),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch_version": torch.__version__,
        "limitations": ["synthetic domains", "one-token actions", "exact expectations", "unmatched teacher success"],
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "summary": summary,
                "identity_checks": {k: v for k, v in result["identity_checks"].items() if k != "mh_kl_path"},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
