#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Summarize finite-pool gradient noise and fixed-LR batch-size experiments."""

import argparse
import collections
import random
import statistics as st
from pathlib import Path

import numpy as np

from ..common.io import dump, read
from ..common.statistics import mean_ci as ci


def groups(rows):
    return {
        "seen64": [r for r in rows if r["id"] < 64],
        "held_math": [r for r in rows if r["group"] in ["E", "F", "G", "H"]],
        "code": [r for r in rows if r["domain"] == "code"],
        "if": [r for r in rows if r["domain"] == "if"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    a = p.parse_args()
    root = a.data
    noise = []
    for f in sorted(root.glob("noise-*.json")):
        if f.name == "noise-summary.json":
            continue
        r = read(f)
        g = np.array(r["gram"])
        n = 32
        mean2 = float(g[:n, :n].mean())
        variance = float(np.diag(g[:n, :n]).mean() - mean2)
        held2 = g[n, n]
        dot = float(g[:n, n].mean())
        out = dict(
            teacher=r["teacher"],
            phase=r["phase"],
            pool_mean_gradient_norm=mean2**0.5,
            batch16_variance_trace=variance,
            noise_to_pool_signal=variance / mean2,
            pool_held_cosine=dot / (mean2 * held2) ** 0.5,
            held_gradient_norm=float(held2**0.5),
            seen_held_cosine=float(g[-1, n] / (g[-1, -1] * held2) ** 0.5),
            pool_risk=st.mean(r["losses"][:n]),
            held_risk=r["losses"][n],
            by_batch=[],
        )
        rng = random.Random(20261027)
        for batch in [16, 64, 128, 512]:
            m = batch // 16
            subsets = (
                [[i] for i in range(32)]
                if m == 1
                else ([list(range(32))] if m == 32 else [rng.sample(range(32), m) for _ in range(4000)])
            )
            cos = []
            dots = []
            for ids in subsets:
                norm2 = float(g[np.ix_(ids, ids)].mean())
                d = float(g[ids, n].mean())
                cos.append(d / (norm2 * held2) ** 0.5)
                dots.append(d)
            expected_var = (n - m) / (m * (n - 1)) * variance
            out["by_batch"].append(
                dict(
                    batch=batch,
                    subset_count=len(subsets),
                    mean_cosine=st.mean(cos),
                    cosine_quantiles=np.quantile(cos, [0.1, 0.5, 0.9]).tolist(),
                    negative_held_dot_fraction=st.mean(d < 0 for d in dots),
                    expected_noise_to_signal=expected_var / mean2,
                    interpretation=(
                        "Uniform aggregation of disjoint B16 blocks within this fixed 512-prompt bank; "
                        "not a population noise estimate."
                    ),
                )
            )
        out["frozen_first_moment"] = []
        for batch in [16, 64, 128, 512]:
            m = batch // 16
            weights = np.zeros(32)
            for step in range(1, 129):
                weights *= 0.9
                offset = ((step - 1) * m) % 32
                weights[offset : offset + m] += 0.1 / m
                if step in [1, 4, 8, 16, 32, 128]:
                    norm2 = float(weights @ g[:32, :32] @ weights)
                    dot = float(weights @ g[:32, 32])
                    out["frozen_first_moment"].append(
                        dict(
                            batch=batch,
                            step=step,
                            held_cosine=dot / (norm2 * held2) ** 0.5,
                            definition=(
                                "Beta1=0.9 EMA of frozen-parameter raw gradients in the actual cyclic block order; "
                                "no clipping or second-moment preconditioning, not the observed Adam update."
                            ),
                        )
                    )
        noise.append(out)
    dump(root / "noise-summary.json", noise)
    trajectories = []
    fresh = []
    for d in sorted(root.iterdir()):
        if not (d / "protocol.json").exists():
            continue
        cfg = read(d / "protocol.json")
        baseline = read(root / f"baseline-{cfg['teacher']}-{cfg['phase']}/fixed.json")
        bases = groups(baseline)
        for file in sorted(d.glob("fixed-*.json")):
            rows = read(file)
            step = rows[0]["step"]
            for group, rs in groups(rows).items():
                out = dict(
                    arm=d.name,
                    teacher=cfg["teacher"],
                    phase=cfg["phase"],
                    batch=cfg["batch_size"],
                    step=step,
                    examples=step * cfg["batch_size"],
                    group=group,
                    prompts=len(rs),
                )
                for metric in ["rkl", "fkl_shared", "js", "tv", "hp"]:
                    before = st.mean(r["stats"]["all"][metric] / 1024 for r in bases[group])
                    after = st.mean(r["stats"]["all"][metric] / 1024 for r in rs)
                    out.update(
                        {metric + "_before": before, metric: after, metric + "_relative_drop": 1 - after / before}
                    )
                trajectories.append(out)
        for file in sorted(d.glob("fresh-*.json")):
            rows = read(file)
            if len(rows) != 64:
                continue
            step = rows[0]["step"]
            old = {(r["id"], r["seed"]): r for r in read(root / f"baseline-{cfg['teacher']}-{cfg['phase']}/fresh.json")}
            changes = collections.defaultdict(list)
            for r in rows:
                changes[r["id"]].append(r["stats"]["all"]["rkl"] - old[r["id"], r["seed"]]["stats"]["all"]["rkl"])
            delta = [st.mean(v) for v in changes.values()]
            fresh.append(
                dict(
                    arm=d.name,
                    teacher=cfg["teacher"],
                    phase=cfg["phase"],
                    batch=cfg["batch_size"],
                    step=step,
                    examples=step * cfg["batch_size"],
                    sequence_rkl=st.mean(r["stats"]["all"]["rkl"] for r in rows),
                    mean_length=st.mean(len(r["response_ids"]) for r in rows),
                    token_rkl=sum(r["stats"]["all"]["rkl"] for r in rows) / sum(len(r["response_ids"]) for r in rows),
                    baseline_sequence_rkl=st.mean(r["stats"]["all"]["rkl"] for r in old.values()),
                    difference_from_baseline=st.mean(delta),
                    paired_prompt_ci=ci(delta),
                )
            )
    dump(root / "trajectory-summary.json", trajectories)
    dump(root / "fresh-summary.json", fresh)
    comparisons = []
    for teacher in ["justrl", "polaris"]:
        for phase in ["initial", "warm128"]:
            batches = [64, 128, 512] if (teacher, phase) == ("polaris", "warm128") else [64, 128]
            for batch in batches:
                for budget, large_step, small_step in [
                    ("equal_updates_8", 8, 8),
                    ("equal_updates", 16, 16),
                    ("equal_examples", 2048 // batch, 128),
                ]:
                    left = root / f"{teacher}-{phase}-b{batch}" / f"fixed-{large_step}.json"
                    right = root / f"{teacher}-{phase}-b16" / f"fixed-{small_step}.json"
                    if not (left.exists() and right.exists()):
                        continue
                    line_rows = groups(read(left))
                    r = groups(read(right))
                    for group in line_rows:
                        ri = {r["id"]: r for r in r[group]}
                        delta = [
                            (row["stats"]["all"]["rkl"] - ri[row["id"]]["stats"]["all"]["rkl"]) / 1024
                            for row in line_rows[group]
                        ]
                        comparisons.append(
                            dict(
                                teacher=teacher,
                                phase=phase,
                                batch=batch,
                                control_batch=16,
                                budget=budget,
                                large_step=large_step,
                                small_step=small_step,
                                group=group,
                                mean_difference=st.mean(delta),
                                paired_prompt_ci=ci(delta),
                            )
                        )
    dump(root / "paired-fixed.json", comparisons)
    fresh_comparisons = []
    for teacher in ["justrl", "polaris"]:
        for phase in ["initial", "warm128"]:
            for batch in [64, 128, 512] if (teacher, phase) == ("polaris", "warm128") else [64, 128]:
                left = root / f"{teacher}-{phase}-b{batch}" / f"fresh-{2048 // batch}.json"
                right = root / f"{teacher}-{phase}-b16" / "fresh-128.json"
                if not (left.exists() and right.exists()):
                    continue
                line_rows = {(r["id"], r["seed"]): r for r in read(left)}
                r = {(r["id"], r["seed"]): r for r in read(right)}
                if len(line_rows) != 64 or len(r) != 64:
                    continue
                assert line_rows.keys() == r.keys()
                delta = collections.defaultdict(list)
                lengths = []
                for key in line_rows:
                    delta[key[0]].append(line_rows[key]["stats"]["all"]["rkl"] - r[key]["stats"]["all"]["rkl"])
                    lengths.append(len(line_rows[key]["response_ids"]) - len(r[key]["response_ids"]))
                values = [st.mean(v) for v in delta.values()]
                fresh_comparisons.append(
                    dict(
                        teacher=teacher,
                        phase=phase,
                        batch=batch,
                        control_batch=16,
                        budget="equal_examples",
                        mean_difference=st.mean(values),
                        paired_prompt_ci=ci(values),
                        mean_length_difference=st.mean(lengths),
                        prompts=len(values),
                    )
                )
    dump(root / "paired-fresh.json", fresh_comparisons)
    cross = []
    for phase in ["initial", "warm128"]:
        for source in ["initial", "warm128"]:
            f = root / f"overlap-mass-{phase}-fresh-{source}.json"
            if not f.exists():
                continue
            r = read(f)
            m = r["means"]
            cross.append(
                dict(
                    parameters=phase,
                    states=source,
                    overlap=m["overlap16"],
                    common_p=m["p_mass_3"],
                    hp=m["hp"],
                    hq=m["hq"],
                    rkl_per_token=m["rkl"],
                    sequence_rkl=sum(x["sums"]["rkl"] for x in r["records"]) / len(r["records"]),
                    shared_gkl_fraction=m["gkl_3"] / m["rkl"],
                )
            )
    dump(root / "overlap-cross-summary.json", cross)
    print("noise cases", len(noise), "fixed observations", len(trajectories), "fresh arms", len(fresh))


if __name__ == "__main__":
    main()
