#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Full-run plateau and matched fixed-state KL-direction statistics."""

import argparse
import collections
import json
import math
import statistics as st
from pathlib import Path

from ..common.io import dump


def rows(p):
    return [json.loads(s) for s in p.open() if s.strip()]


def avg(rs, k):
    return st.mean(r[k] for r in rs)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--data", type=Path, required=True)
    x = a.parse_args()
    p = x.data
    h = rows(p / "run/history.jsonl")
    u = [r for r in h if "distill/full_kl" in r]
    o = [r for r in h if "rollout/response_tokens" in r]
    batches = collections.defaultdict(list)
    for r in u:
        batches[r["step/outer"]].append(r)
    windows = {}
    for lo, hi in [(1, 16), (49, 64), (113, 128), (177, 192), (241, 256)]:
        bs = [batches[i] for i in range(lo, hi + 1)]
        starts = [r[0]["distill/full_kl"] for r in bs]
        ends = [r[-1]["distill/full_kl"] for r in bs]
        os = [r for r in o if lo <= r["step/outer"] <= hi]
        w = dict(
            first_mean=st.mean(starts),
            first_std=st.stdev(starts),
            last_mean=st.mean(ends),
            last_std=st.stdev(ends),
            mean_relative_drop=st.mean(1 - b / a for a, b in zip(starts, ends, strict=False)),
            grad_mean=st.mean(r["optim/actor/grad_norm_preclip"] for b in bs for r in b),
            train_length=avg(os, "rollout/response_length/mean"),
            train_score=avg(os, "rollout/score/mean"),
            length_stop_fraction=sum(r["rollout/finish_reason/length_count"] for r in os)
            / sum(r["rollout/response_count"] for r in os),
        )
        for side, index in [("first", 0), ("last", -1)]:
            w["positions_" + side] = [
                sum(b[index][f"distill/full_kl_position/p{i:02d}_{i + 10:02d}/sum"] for b in bs)
                / sum(b[index][f"distill/full_kl_position/p{i:02d}_{i + 10:02d}/token_count"] for b in bs)
                for i in range(0, 100, 10)
            ]
        windows[f"{lo}-{hi}"] = w
    evals = [
        {k: v for k, v in r.items() if k.startswith(("step/", "eval/"))} for r in h if "eval/math500/acc/mean" in r
    ]
    b = list(batches.values())
    gain = sum(r[0]["distill/full_kl"] - r[-1]["distill/full_kl"] for r in b[:-1])
    jump = sum(y[0]["distill/full_kl"] - z[-1]["distill/full_kl"] for z, y in zip(b, b[1:], strict=False))
    audit = dict(
        history=len(h),
        updates=len(u),
        outers=len(o),
        nonfinite=sum(r["optim/actor/loss_nonfinite"] + r["optim/actor/grad_norm_preclip_nonfinite"] for r in u),
        skips=sum(r["optim/actor/update_skipped"] for r in u),
        clipped=sum(r["optim/actor/grad_norm_preclip"] > 1 for r in u),
        all_batches_decrease=all(r[-1]["distill/full_kl"] < r[0]["distill/full_kl"] for r in b),
        within_adjacent_increases=sum(
            y["distill/full_kl"] > z["distill/full_kl"] for r in b for z, y in zip(r, r[1:], strict=False)
        ),
        cross_batch_difference_over_within_batch_gain=jump / gain,
    )
    dump(p / "run-statistics.json", dict(audit=audit, windows=windows, evals=evals))
    summary = []
    for folder in sorted(p.glob("*-fold?")):
        if not (folder / "evaluation.json").exists():
            continue
        cfg = json.load(open(folder / "protocol.json"))
        ev = json.load(open(folder / "evaluation.json"))
        groups = collections.defaultdict(list)
        for r in ev:
            partition = (
                "train" if r["prompt_id"] in cfg["train_ids"] else ("heldout" if r["prompt_id"] >= 8 else "other_fold")
            )
            groups[(r["step"], partition, r["source"], r["domain"], r["region"])].append(r)
        for key, rs in groups.items():
            if key[0] == 0:
                continue
            initial = groups[(0, *key[1:])]
            row = dict(
                arm=folder.name,
                teacher=cfg["teacher"],
                direction=cfg["direction"],
                step=key[0],
                partition=key[1],
                source=key[2],
                domain=key[3],
                region=key[4],
                n=len(rs),
            )
            for metric in ["reverse", "forward", "js", "tv"]:
                before, after = avg(initial, metric), avg(rs, metric)
                row[metric + "_before"] = before
                row[metric + "_after"] = after
                row[metric + "_relative_drop"] = 1 - after / before
            row["entropy_before"] = avg(initial, "entropy")
            row["entropy_after"] = avg(rs, "entropy")
            summary.append(row)
    dump(p / "direction-summary.json", summary)
    diagnostics = json.load(open(p / "gradient-diagnostics.json"))
    g = collections.defaultdict(list)
    for r in diagnostics:
        g[(r["teacher"], r["source"], r["domain"], r["region"])].append(r)
    diag = []
    for key, rs in g.items():
        d = dict(zip(["teacher", "source", "domain", "region"], key, strict=False))
        d["count"] = len(rs)
        for k in rs[0]:
            if isinstance(rs[0][k], float):
                d[k] = avg(rs, k)
        d["max_extra_mass"] = max(r["extra_teacher_mass"] for r in rs)
        diag.append(d)
    dump(p / "gradient-summary.json", diag)
    tails = []
    for teacher in ["justrl", "polaris"]:
        for source in ["student", "polaris"]:
            rs = [r for r in diagnostics if r["teacher"] == teacher and r["source"] == source and r["domain"] == "math"]
            item = dict(teacher=teacher, source=source, count=len(rs))
            for k in ["reverse_grad_norm", "forward_grad_norm", "reverse", "forward"]:
                v = sorted(r[k] for r in rs)
                item[k] = dict(mean=st.mean(v), std=st.pstdev(v), p99=v[math.ceil(0.99 * len(v)) - 1], max=max(v))
            item["pooled_gradient_cosine"] = sum(
                r["gradient_cosine"] * r["reverse_grad_norm"] * r["forward_grad_norm"] for r in rs
            ) / math.sqrt(sum(r["reverse_grad_norm"] ** 2 for r in rs) * sum(r["forward_grad_norm"] ** 2 for r in rs))
            tails.append(item)
    dump(p / "gradient-tail-summary.json", tails)
    print(json.dumps(audit, indent=2))
    print("summarized arms", len({r["arm"] for r in summary}))


if __name__ == "__main__":
    main()
