#!/usr/bin/env python3
"""Audit matched inputs, completed updates, finite metrics and support semantics."""

import argparse
import json
import math
from pathlib import Path

import torch
from kl_direction_probe import distributions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    a = p.parse_args()
    root = a.data
    runs = []
    for f in sorted(root.glob("*-fold?")):
        if not f.is_dir():
            continue
        config = json.load(open(f / "protocol.json"))
        train = json.load(open(f / "training.json"))
        ev = json.load(open(f / "evaluation.json"))
        assert [r["step"] for r in train] == list(range(1, 17))
        assert all(math.isfinite(r["loss_before"]) and math.isfinite(r["grad_norm"]) for r in train)
        assert all(r["parameter_step_norm"] > 0 for r in train if "parameter_step_norm" in r)
        assert sorted({r["step"] for r in ev}) == [0, 1, 8, 16]
        keys = None
        for step in [0, 1, 8, 16]:
            rs = [r for r in ev if r["step"] == step]
            assert len(rs) == 3376
            now = {(r["prompt_id"], r["source"], r["position"]) for r in rs}
            if keys is None:
                keys = now
            assert now == keys
            assert all(math.isfinite(r[k]) for r in rs for k in ["reverse", "forward", "js", "tv", "entropy"])
        runs.append(
            dict(
                arm=f.name,
                steps=len(train),
                clipped=sum(r["grad_norm"] > 1 for r in train),
                peak_gib=max(r["peak_gib"] for r in train),
            )
        )
    assert len(runs) == 12
    for tag, t, folds in [("initial", "justrl", [0, 1]), ("initial", "polaris", [0, 1]), ("warm64", "polaris", [0, 1])]:
        for fold in folds:
            x, y = [
                [r for r in json.load(open(root / f"{tag}-{t}-{d}-fold{fold}/evaluation.json")) if r["step"] == 0]
                for d in ["reverse", "forward"]
            ]
            assert x == y
    torch.manual_seed(42)
    z = torch.randn(5, 11, dtype=torch.float64, requires_grad=True)
    t = torch.randn(5, 14, dtype=torch.float64)
    lp, lq = distributions(z, t)
    p, q = lp.exp(), lq.exp()
    r = (p * (lp - lq)).sum(-1)
    gr = torch.autograd.grad(r.sum(), z, retain_graph=True)[0]
    gf = torch.autograd.grad((q * (lq - lp)).sum(), z, retain_graph=True)[0]
    er = float((gr - p * (lp - lq - r[:, None])).abs().max().detach())
    ef = float((gf - (p - q)).abs().max().detach())
    assert max(er, ef) < 1e-12
    qfull = t.softmax(-1)
    pfull = torch.cat([p, torch.zeros(5, 3, dtype=p.dtype)], dim=-1)
    assert torch.isinf((qfull * (qfull.log() - pfull.log())).sum())
    h = [json.loads(s) for s in (root / "run/history.jsonl").open()]
    u = [r for r in h if "distill/full_kl" in r]
    assert [r["step/optimizer"] for r in u] == list(range(1, 4097))
    out = dict(
        completed_arms=runs,
        optimizer_calls=sum(r["steps"] for r in runs),
        identical_initial_evaluations=True,
        fixed_inputs_across_updates=True,
        unequal_support_gradient_error={"reverse": er, "forward": ef},
        native_forward_kl_with_zero_extended_student_is_infinite=True,
        e48_optimizer_axis_continuous=True,
    )
    (root / "verification.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
