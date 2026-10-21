# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Audit full-KL computation graphs and optional reviewed One-Shot support helpers on CPU."""

import argparse
import ast
import json
from pathlib import Path

import torch


def graph_audit():
    maxima = {}
    wrong_norms = []
    for seed in range(20):
        generator = torch.Generator().manual_seed(seed)
        z = torch.randn(7, dtype=torch.float64, generator=generator).requires_grad_()
        log_T = torch.randn(7, dtype=torch.float64, generator=generator).log_softmax(-1)
        beta = torch.randn(7, dtype=torch.float64, generator=generator).softmax(-1)
        log_p = z.log_softmax(-1)
        p = log_p.exp()
        ell = log_p - log_T
        K = (p * ell).sum()
        exact = torch.autograd.grad(K, z, retain_graph=True)[0]
        delta = p * (ell - K)
        surrogates = {
            "stopped_gap_full_sum": (p * ell.detach()).sum(),
            "stopped_score_coefficient": ((p * ell).detach() * log_p).sum(),
            "analytic_logit_vjp": (delta.detach() * z).sum(),
        }
        for name, loss in surrogates.items():
            actual = torch.autograd.grad(loss, z, retain_graph=True)[0]
            maxima[name] = max(maxima.get(name, 0), (actual - exact).abs().max().item())
        wrong = (p.detach() * ell).sum()
        wrong_grad = torch.autograd.grad(wrong, z, retain_graph=True)[0]
        assert wrong_grad.abs().max() < 1e-14
        wrong_norms.append((wrong_grad - exact).norm().item())
        score = torch.eye(7, dtype=torch.float64) - p[None, :]
        value_is = (p / beta) * ell
        corrected = (p / beta) * (ell - 1) + log_T.exp() / beta
        for name, values in [("value_is", value_is), ("corrected_value", corrected)]:
            assert abs((beta * values).sum() - K) < 1e-12
            gradients = torch.stack([torch.autograd.grad(values[a], z, retain_graph=True)[0] for a in range(7)])
            error = ((beta[:, None] * gradients).sum(0) - exact).abs().max().item()
            maxima[name + "_expected_gradient"] = max(maxima.get(name + "_expected_gradient", 0), error)
            if name == "corrected_value":
                expected = (p / beta * ell)[:, None] * score
                assert (gradients - expected).abs().max() < 1e-12
        singleton = torch.tensor([1])
        normalized = (
            log_p[singleton].softmax(-1) * (log_p[singleton].log_softmax(-1) - log_T[singleton].log_softmax(-1))
        ).sum()
        assert normalized.item() == 0
        assert torch.autograd.grad(normalized, z, retain_graph=True)[0].abs().max() == 0
    assert max(maxima.values()) < 1e-12
    return dict(
        cases=20,
        maximum_gradient_errors=maxima,
        wrong_detached_probability_max_gradient_error=max(wrong_norms),
        singleton_normalized_kl_has_zero_value_and_gradient=True,
    )


def oneshot_audit(root):
    path = root / "verl/verl/workers/fsdp_teacher_worker.py"
    functions = {"compute_top_k_teacher_log_probs", "compute_reward_weights", "compute_top_k_reward"}
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in functions]
    assert len(nodes) == len(functions)
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    q = torch.tensor([[0.4, 0.3, 0.2, 0.1]])
    T = torch.tensor([[0.1, 0.25, 0.45, 0.2]])
    sq = q.topk(2).indices
    log_q = q.log().gather(-1, sq)
    results = {}
    for strategy in ["only_stu", "only_tch", "intersection", "union", "union-intersection"]:
        scored = namespace["compute_top_k_teacher_log_probs"](T.log(), sq, 2, strategy)
        reward = namespace["compute_top_k_reward"](
            log_q,
            scored["log_probs"],
            scored["student_in_teacher"],
            scored["teacher_in_student"],
            strategy,
            "student_p",
        )
        results[strategy] = dict(
            candidate_ids=sq[0].tolist(),
            reward=reward[0].tolist(),
            active_ids=sq[0][reward[0] != 0].tolist(),
            width=reward.shape[-1],
        )
    assert results["only_stu"]["reward"] == results["only_tch"]["reward"] == results["union"]["reward"]
    assert all(row["width"] == 2 for row in results.values())
    st = T.topk(2).indices
    scored = namespace["compute_top_k_teacher_log_probs"](T.log(), sq, 2, "union")
    cat_ids = torch.cat([sq, st], dim=-1)
    union_reward = namespace["compute_top_k_reward"](
        q.log().gather(-1, cat_ids),
        T.log().gather(-1, cat_ids),
        scored["student_in_teacher"],
        scored["teacher_in_student"],
        "union",
        "student_p",
    )
    assert union_reward.shape[-1] == 4
    # This token is teacher-only; it appears only once both sides' scores are supplied.
    assert 2 not in results["union"]["candidate_ids"] and 2 in cat_ids[0][union_reward[0] != 0].tolist()
    # Actual weighted-PG signal differs from a normalized-support KL loss even at p=q.
    z = q.log().clone().requires_grad_()
    lp = z.log_softmax(-1)
    cfg_reward = namespace["compute_top_k_reward"](
        log_q,
        T.log().gather(-1, sq),
        scored["student_in_teacher"],
        scored["teacher_in_student"],
        "only_stu",
        "student_p",
    ).detach()
    pg = -(cfg_reward * (lp.gather(-1, sq) - log_q).exp()).sum()
    pg_gradient = torch.autograd.grad(pg, z, retain_graph=True)[0]
    log_p_s = lp.gather(-1, sq).log_softmax(-1)
    log_T_s = T.log().gather(-1, sq).log_softmax(-1)
    norm_kl = (log_p_s.exp() * (log_p_s - log_T_s)).sum()
    norm_gradient = torch.autograd.grad(norm_kl, z)[0]
    assert (pg_gradient - norm_gradient).abs().max() > 1e-3
    return dict(
        scope="Extracted helper functions fed with the K-wide tensors supplied by compute_rm_score; no GPU/Ray run",
        student_anchor=q[0].tolist(),
        teacher=T[0].tolist(),
        student_top2=sq[0].tolist(),
        teacher_top2=st[0].tolist(),
        expected_union=sorted(set(sq[0].tolist()) | set(st[0].tolist())),
        expected_symmetric_difference=sorted(set(sq[0].tolist()) ^ set(st[0].tolist())),
        pipeline_shaped_results=results,
        full_concatenated_helper=dict(ids=cat_ids[0].tolist(), reward=union_reward[0].tolist()),
        weighted_pg_vs_normalized_kl_max_gradient_difference=(pg_gradient - norm_gradient).abs().max().item(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oneshot-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = dict(full_kl=graph_audit(), oneshot=oneshot_audit(args.oneshot_root) if args.oneshot_root else None)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
