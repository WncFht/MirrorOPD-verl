#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Freeze explicit local comparisons and summarize all finite scalar metrics.

Outer windows are descriptive, not independent training-seed replicates.
First/last reuse observations bracket 15, not 16, optimizer operations.
"""

import argparse
import collections
import csv
import json
import math
import shutil
import statistics as st
from pathlib import Path

from ..common.io import dump

RUNS = dict(E48="grcuotmi", E20="lwgb5eex", E17="cix5i2x2", E18="9hx3uegm", E19="d6jhh378", E35="ntzdb4o7")


def read(p):
    return [json.loads(s) for s in p.open() if s.strip()]


def finite(x):
    return isinstance(x, int | float) and not isinstance(x, bool) and math.isfinite(x)


def avg(rows, key):
    xs = [r[key] for r in rows if finite(r.get(key))]
    return st.mean(xs) if xs else None


def flat(x, stem=""):
    out = {}
    for k, v in x.items():
        name = stem + k
        if isinstance(v, dict):
            out.update(flat(v, name + "."))
        else:
            out[name] = v
    return out


def analyze(rows):
    updates = [r for r in rows if "optim/actor/loss" in r]
    outers = [r for r in rows if "rollout/response_tokens" in r]
    groups = collections.defaultdict(list)
    for r in updates:
        groups[r["step/outer"]].append(r)
    groups = {k: sorted(v, key=lambda r: r["step/update"]) for k, v in groups.items()}
    metrics = []
    for key in sorted({k for r in rows for k in r}):
        obs = [(r.get("step/outer"), r.get("step/update"), r[key]) for r in rows if finite(r.get(key))]
        if obs:
            vals = [v[2] for v in obs]
            metrics.append(
                dict(
                    metric=key,
                    count=len(vals),
                    first=vals[0],
                    last=vals[-1],
                    min=min(vals),
                    max=max(vals),
                    mean=st.mean(vals),
                    first_outer=obs[0][0],
                    last_outer=obs[-1][0],
                )
            )
    batch = []
    for o, rs in groups.items():
        a, b = rs[0], rs[-1]
        k = "distill/full_kl"
        if k not in a:
            continue
        item = dict(
            outer=o,
            update=b["step/update"],
            reuse=len(rs),
            first=a[k],
            last=b[k],
            absolute_drop=a[k] - b[k],
            relative_drop=1 - b[k] / a[k],
            first_grad=a["optim/actor/grad_norm_preclip"],
            last_grad=b["optim/actor/grad_norm_preclip"],
            rises=sum(y[k] > x[k] for x, y in zip(rs, rs[1:], strict=False)),
        )
        for pos in range(0, 100, 10):
            key = f"distill/full_kl_position/p{pos:02d}_{pos + 10:02d}/mean"
            if key in a:
                item[f"p{pos:02d}_first"] = a[key]
                item[f"p{pos:02d}_last"] = b[key]
        batch.append(item)
    windows = {}
    for lo, hi in [(1, 8), (9, 16), (17, 32), (33, 48), (49, 58), (1, 32), (1, 58)]:
        bs = [b for b in batch if lo <= b["outer"] <= hi]
        us = [r for r in updates if lo <= r["step/outer"] <= hi]
        os = [r for r in outers if lo <= r["step/outer"] <= hi]
        if not bs:
            continue
        w = {k: avg(bs, k) for k in bs[0] if k not in ("outer", "update")}
        w.update({k: avg(os, k) for k in outers[0] if not k.startswith("_")})
        w["grad_mean"] = avg(us, "optim/actor/grad_norm_preclip")
        w["clip_count"] = sum(r["optim/actor/grad_norm_preclip"] > 1 for r in us)
        w["batches"] = len(bs)
        w["length_stop_fraction"] = sum(r["rollout/finish_reason/length_count"] for r in os) / sum(
            r["rollout/response_count"] for r in os
        )
        for side, index in [("first", 0), ("last", -1)]:
            ends = [rs[index] for o, rs in groups.items() if lo <= o <= hi]
            for pos in range(0, 100, 10):
                stem = f"distill/full_kl_position/p{pos:02d}_{pos + 10:02d}/"
                valid = [r for r in ends if stem + "sum" in r]
                if valid:
                    w[f"position_{side}_{pos:02d}"] = sum(r[stem + "sum"] for r in valid) / sum(
                        r[stem + "token_count"] for r in valid
                    )
        windows[f"{lo}-{hi}"] = w
    evals = [
        {k: v for k, v in r.items() if k.startswith(("eval/", "step/"))} for r in rows if "eval/math500/acc/mean" in r
    ]
    audit = dict(
        rows=len(rows),
        updates=len(updates),
        outer=len(outers),
        final_optimizer=max(r.get("step/optimizer", 0) for r in rows),
        nonfinite=sum(
            r.get("optim/actor/loss_nonfinite", 0) + r.get("optim/actor/grad_norm_preclip_nonfinite", 0)
            for r in updates
        ),
        skipped=sum(r.get("optim/actor/update_skipped", 0) for r in updates),
        grad_clip_count=sum(r["optim/actor/grad_norm_preclip"] > 1 for r in updates),
        batch_decrease_fraction=st.mean(b["last"] < b["first"] for b in batch),
        adjacent_decrease_fraction=1 - sum(b["rises"] for b in batch) / sum(b["reuse"] - 1 for b in batch),
    )
    if all("distill/full_kl_position/p00_10/sum" in r for r in updates):
        audit["position_reconstruction_max_abs_error"] = max(
            abs(
                r["distill/full_kl"]
                - sum(r[f"distill/full_kl_position/p{p:02d}_{p + 10:02d}/sum"] for p in range(0, 100, 10))
                / sum(r[f"distill/full_kl_position/p{p:02d}_{p + 10:02d}/token_count"] for p in range(0, 100, 10))
            )
            for r in updates
        )
    return dict(audit=audit, windows=windows, batches=batch, evals=evals), metrics


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--logs-repo", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    args = a.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    result = {}
    configs = {}
    for label, rid in RUNS.items():
        dest = args.out / ("run" if label == "E48" else label)
        if not dest.exists():
            source = next((args.logs_repo / "derived/MirrorOPD-verl").glob("*-" + rid))
            dest.mkdir()
            for name in ["history.jsonl", "config.json", "run.json", "metric-definitions.jsonl"]:
                shutil.copyfile(source / name, dest / name)
        rows = read(dest / "history.jsonl")
        result[label], metrics = analyze(rows)
        configs[label] = flat(json.loads((dest / "config.json").read_text()))
        with (args.out / f"{label}-metrics.csv").open("w") as f:
            w = csv.DictWriter(f, fieldnames=list(metrics[0]))
            w.writeheader()
            w.writerows(metrics)
    dump(args.out / "statistics.json", result)
    dump(
        args.out / "config-differences.json",
        {
            label: {
                k: {"E48": configs["E48"].get(k), "reference": cfg.get(k)}
                for k in sorted(set(cfg) | set(configs["E48"]))
                if cfg.get(k) != configs["E48"].get(k)
            }
            for label, cfg in configs.items()
            if label != "E48"
        },
    )
    print(json.dumps({k: v["audit"] for k, v in result.items()}, indent=2))


if __name__ == "__main__":
    main()
