#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Evidence ledger for the consolidated transfer report.

S/I/B/N denote distinct protocols, never pooled as training replicates.
Transfer bars: relative fixed-bank native RKL decrease; teacher colors teal/orange.
Source heatmap: I1 math, 64 updates, one fixed reference source per column.
Gradient bars: full-parameter raw cosine, fixed banks, not Adam update cosine.
Overlap panels: rank-group mass versus unweighted membership; nonnegative KL
allocation p*log(p/q)-p+q over four disjoint sets on the union vocabulary.
Cross-state heatmaps: identical prompt/seed pairs and teacher-forcing scorer.
Batch bars: N warm Polaris, constant LR, separate equal-update/equal-example axes.
Fresh contrasts: paired prompt bootstrap, conditional on one trained model;
panels belong to different experiments and do not rank effects across studies.
Every plotted scalar and its source file are exported without file digests.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from ..common.plotting import load_style


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    fv = load_style()
    a.out.mkdir(parents=True, exist_ok=True)
    root = a.workspace.resolve()
    doc = root / "MirrorOPD-verl/examples/mirroropd/docs/experiments/polaris-transfer"
    paths = {
        "I": doc / "data/intersection-study-20261027",
        "B": root / "wandb-logs/MirrorOPD-verl/analysis/batch-transfer-20261027",
        "N": root / "wandb-logs/MirrorOPD-verl/analysis/batch-size-stage-20261027",
    }
    sources = []
    values = {}
    audit = {}

    def read(family, name):
        path = paths[family] / name
        sources.append(str(path.relative_to(root)))
        return json.loads(path.read_text())

    fv.apply_rc("paper", fontsize=10)
    teal, orange = fv.HUES["vteal"], fv.HUES["vorange"]
    ink = fv.NEUTRALS["vink"]
    gray = fv.NEUTRALS["baseline"]

    def save(fig, name, caption):
        fig.text(0.5, 0.02, caption, ha="center", fontsize=9, color=gray)
        fig.canvas.draw()
        audit[name] = [str(x) for x in fv.audit(fig)]
        for ext in ["png", "pdf"]:
            fig.savefig(a.out / f"{name}.{ext}", dpi=180)
        plt.close(fig)

    def labels(ax, bars, fmt="{:.1f}"):
        ax.bar_label(bars, labels=[fmt.format(v.get_height()) for v in bars], padding=4, fontsize=9)

    stats = read("B", "transfer-summary.json")
    groups = ["A", "BCD", "held_math", "Code", "IF"]
    v = []
    fig, ax = plt.subplots(figsize=(6.8, 4.7))
    fig.subplots_adjust(left=0.13, right=0.98, top=0.92, bottom=0.26)
    for j, (teacher, color, label) in enumerate([("justrl", teal, "JustRL"), ("polaris", orange, "Polaris")]):
        ys = []
        for group in groups:
            rs = [
                r
                for r in stats
                if r["arm"] == teacher + "-single"
                and r["step"] == 128
                and (r["group"] in ["B", "C", "D"] if group == "BCD" else r["group"] == group)
            ]
            ys.append(100 * (1 - sum(r["rkl_after"] for r in rs) / sum(r["rkl_before"] for r in rs)))
        labels(ax, ax.bar(np.arange(5) + (j - 0.5) * 0.36, ys, 0.36, color=color, label=label))
        v.append(dict(teacher=teacher, drop_percent=ys))
    ax.set(
        xticks=range(5),
        xticklabels=["Fitted A", "Unseen B–D", "Held math", "Code", "IF"],
        ylabel="Fixed RKL decrease (%)",
        xlabel="Evaluation states",
        ylim=(0, 110),
    )
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.075), ncol=2, frameon=False)
    values["single-batch-transfer"] = v
    save(fig, "single-batch-transfer", "B1: train only 16 math trajectories for 128 updates; H=1024.")

    paired = read("B", "paired-controls.json")
    contrast = next(
        r
        for r in paired
        if r["left"] == "polaris-rich-lr1" and r["right"] == "polaris-roundrobin" and r["measure"] == "fresh"
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.7))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.93, bottom=0.29, wspace=0.48)
    coverage_values = []
    for i, (arm, label, color) in enumerate(
        [
            ("polaris-roundrobin", "64 trajectories", gray),
            ("polaris-rich-lr1", "512 trajectories", orange),
        ]
    ):
        rs = sorted([r for r in stats if r["arm"] == arm and r["group"] == "held_math"], key=lambda r: r["step"])
        axes[0].plot(
            [0] + [r["step"] for r in rs],
            [rs[0]["rkl_before"]] + [r["rkl_after"] for r in rs],
            marker="o",
            color=color,
            label=label,
        )
        raw = read("B", arm + "/fresh-128.json")
        value = float(np.mean([r["stats"]["all"]["rkl"] for r in raw]))
        labels(axes[1], axes[1].bar([i], [value], 0.5, color=color, label=label), "{:.1f}")
        coverage_values.append(dict(arm=arm, fixed=rs, fresh=value))
    axes[0].set(xlabel="Optimizer updates", ylabel="Fixed math RKL / horizon", xticks=[0, 32, 64, 96, 128])
    axes[1].set(
        xlabel="Independent training trajectories",
        ylabel="Fresh sequence KL (nats)",
        xticks=[0, 1],
        xticklabels=["64", "512"],
        ylim=(0, 250),
    )
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.075),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    values["independent-coverage"] = dict(series=coverage_values, paired_fresh=contrast)
    save(
        fig,
        "independent-coverage",
        (
            f"B2: LR=1e-6, B16, 128 updates; fresh difference {contrast['mean_difference']:.2f} "
            f"[{contrast['paired_prompt_ci'][0]:.2f}, {contrast['paired_prompt_ci'][1]:.2f}]."
        ),
    )

    fixed = read("I", "summary-fixed.json")
    arms = [
        "justrl-student-k16",
        "polaris-student-k16",
        "polaris-teacher-k16",
        "polaris-intersection-k16",
        "polaris-topk-k16",
    ]
    matrix = np.array(
        [
            [
                next(
                    r["drop"] * 100
                    for r in fixed
                    if r["arm"] == arm
                    and r["step"] == 64
                    and r["domain"] == "math"
                    and r["source"] == s
                    and r["band"] == "all"
                )
                for s in ["student", "justrl", "polaris"]
            ]
            for arm in arms
        ]
    )
    cmap = LinearSegmentedColormap.from_list("paper_teal", [fv.NEUTRALS["paper"], teal])
    fig, ax = plt.subplots(figsize=(6.8, 4.7))
    fig.subplots_adjust(left=0.31, right=0.84, top=0.93, bottom=0.24)
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for i in range(5):
        for j in range(3):
            ax.text(j, i, f"{matrix[i, j]:.1f}%", ha="center", va="center", color=ink)
    ax.set(
        xticks=range(3),
        xticklabels=["Initial student", "JustRL", "Polaris"],
        yticks=range(5),
        yticklabels=[
            "JustRL / student",
            "Polaris / student",
            "Polaris / teacher",
            "Polaris / intersection",
            "Polaris / top-16",
        ],
        xlabel="Frozen reference source",
        ylabel="Teacher / training source",
    )
    cax = fig.add_axes([0.88, 0.27, 0.02, 0.62])
    fig.colorbar(im, cax=cax, label="RKL decrease (%)")
    values["source-transfer"] = dict(arms=arms, matrix=matrix.tolist())
    save(fig, "source-transfer", "I1: 64 updates; the same 64 held-out math prompts in every column.")

    cases = [read("B", f"fixed-gradient-case-{i}.json")[0] for i in range(5)]
    coverage = read("B", "coverage-gradient-summary.json")
    just = [cases[i]["cosine"][0][1] for i in [0, 2, 4]]
    pol = [cases[i]["cosine"][0][1] for i in [1, 3, 3]]
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 7.6))
    fig.subplots_adjust(left=0.14, right=0.97, top=0.94, bottom=0.15, hspace=0.55)
    for j, (ys, c, label) in enumerate([(just, teal, "JustRL target"), (pol, orange, "Polaris target")]):
        labels(axes[0], axes[0].bar(np.arange(3) + (j - 0.5) * 0.36, ys, 0.36, color=c, label=label), "{:.3f}")
    axes[0].set(
        xticks=range(3),
        xticklabels=["Initial student", "Each own128", "Same Polaris128"],
        xlabel="Student parameters",
        ylabel="Train–held gradient cosine",
        ylim=(0, 1.12),
    )
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=2, frameon=False, fontsize=9)
    ys = [
        coverage[0]["seen_held_cosine"],
        next(r["new_held_cosine"] for r in coverage if r["new_prompts"] == 64),
        next(r["new_held_cosine"] for r in coverage if r["new_prompts"] == 448),
    ]
    labels(axes[1], axes[1].bar(range(3), ys, 0.55, color=orange), "{:.3f}")
    axes[1].set(
        xticks=range(3),
        xticklabels=["Seen 64", "New 64", "New 448"],
        xlabel="Gradient data at the same Polaris128 checkpoint",
        ylabel="Train–held gradient cosine",
        ylim=(0, 1.12),
    )
    values["gradient-transfer"] = dict(justrl=just, polaris=pol, coverage=ys)
    save(fig, "gradient-transfer", "B4: full-parameter raw gradients; target and state controls shown separately.")

    warm = read("N", "overlap-mass-warm128.json")["means"]
    cross = read("N", "overlap-cross-summary.json")
    mass = [100 * warm[f"rank_{lo}_{hi}_p_mass"] for lo, hi in [(1, 1), (2, 4), (5, 8), (9, 16)]]
    membership = [100 * warm[f"rank_{lo}_{hi}_shared_fraction"] for lo, hi in [(1, 1), (2, 4), (5, 8), (9, 16)]]
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 7.5))
    fig.subplots_adjust(left=0.23, right=0.97, top=0.94, bottom=0.21, hspace=0.58)
    for j, (ys, c, label) in enumerate(
        [(mass, teal, "Student probability mass"), (membership, orange, "Candidates in teacher top-16")]
    ):
        labels(axes[0], axes[0].bar(np.arange(4) + (j - 0.5) * 0.36, ys, 0.36, color=c, label=label))
    axes[0].set(
        xticks=range(4),
        xticklabels=["1", "2–4", "5–8", "9–16"],
        xlabel="Student rank group",
        ylabel="Percentage (%)",
        ylim=(0, 115),
    )
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.24), ncol=1, frameon=False, fontsize=9)
    current = read("N", "overlap-mass-warm128-fresh-warm128.json")["means"]
    alloc = np.array([[r[f"gkl_{c}"] / r["rkl"] * 100 for c in range(4)] for r in [warm, current]])
    bottom = np.zeros(2)
    for c, color, label in [
        (3, teal, "Intersection"),
        (2, orange, "Student only"),
        (1, fv.HUES["vviolet"], "Teacher only"),
        (0, gray, "Neither"),
    ]:
        axes[1].barh([0, 1], alloc[:, c], left=bottom, color=color, label=label)
        bottom += alloc[:, c]
    axes[1].set(
        yticks=[0, 1],
        yticklabels=[f"Fixed prefixes\n({alloc[0, 3]:.1f}% shared)", f"Current fresh\n({alloc[1, 3]:.1f}% shared)"],
        xlabel="Nonnegative RKL allocation (%)",
        ylabel="Bank",
        xlim=(0, 101),
    )
    fig.legend(
        *axes[1].get_legend_handles_labels(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.065),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    values["overlap-residual"] = dict(rank_mass=mass, rank_membership=membership, kl_allocation=alloc.tolist())
    save(fig, "overlap-residual", "N4: Polaris128; common candidates carry most mass and most residual KL.")

    fig, axes = plt.subplots(1, 2, figsize=(7, 4.1))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.82, bottom=0.31, wspace=0.38)
    vals = {}
    for ax, key, title in zip(
        axes, ["overlap", "rkl_per_token"], ["Overlap@16 (%)", "Native RKL per token"], strict=False
    ):
        mat = np.array(
            [
                [
                    next(r[key] for r in cross if r["parameters"] == p and r["states"] == s)
                    for s in ["initial", "warm128"]
                ]
                for p in ["initial", "warm128"]
            ]
        )
        if key == "overlap":
            mat *= 100
        ax.imshow(mat, cmap=cmap, aspect="auto")
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    f"{mat[i, j]:.2f}" if key == "overlap" else f"{mat[i, j]:.4f}",
                    ha="center",
                    va="center",
                    color=ink,
                )
        ax.set(
            title=title,
            xticks=[0, 1],
            xticklabels=["Initial", "Polaris128"],
            yticks=[0, 1],
            yticklabels=["Initial", "Polaris128"],
            xlabel="Frozen state source",
        )
        vals[key] = mat.tolist()
    axes[0].set_ylabel("Student parameters")
    values["state-feedback"] = vals
    save(fig, "state-feedback", "N4: identical 32 prompts × 2 seeds; one common teacher-forcing scorer.")

    stats = read("N", "trajectory-summary.json")
    fig, axes = plt.subplots(1, 2, figsize=(7, 4.5))
    fig.subplots_adjust(left=0.11, right=0.97, top=0.87, bottom=0.29, wspace=0.28)
    vals = []
    for ax, batches, steps, title in [
        (axes[0], [16, 64, 128], [16] * 3, "Equal updates: 16"),
        (axes[1], [16, 64, 128, 512], [128, 32, 16, 4], "Equal examples: 2048"),
    ]:
        ys = [
            next(
                r["rkl_relative_drop"] * 100
                for r in stats
                if r["teacher"] == "polaris"
                and r["phase"] == "warm128"
                and r["batch"] == b
                and r["step"] == s
                and r["group"] == "held_math"
            )
            for b, s in zip(batches, steps, strict=False)
        ]
        labels(ax, ax.bar(range(len(batches)), ys, 0.6, color=orange), "{:.2f}")
        ax.set(
            title=title,
            xticks=range(len(batches)),
            xticklabels=[str(b) for b in batches],
            xlabel="Effective batch size",
            ylim=(0, 5),
        )
        vals.append(dict(batch=batches, steps=steps, drop_percent=ys))
    for ax in axes:
        ax.set_ylabel("Held-out RKL decrease (%)")
    values["batch-budgets"] = vals
    save(fig, "batch-budgets", "N1/N2: same Polaris128 start, LR=1e-6, same ordered 512-trajectory pool.")

    contrasts = read("I", "paired-method-contrasts.json")
    fresh = read("N", "fresh-summary.json")
    nrows = [
        next(
            r
            for r in fresh
            if r["teacher"] == "polaris" and r["phase"] == "warm128" and r["batch"] == b and r["step"] == s
        )
        for b, s in [(16, 128), (64, 32), (128, 16), (512, 4), (512, 8)]
    ]
    fig, axes = plt.subplots(2, 1, figsize=(7.3, 8.2))
    fig.subplots_adjust(left=0.37, right=0.96, top=0.94, bottom=0.19, hspace=0.48)
    for ax, rows, names, title, field in [
        (
            axes[0],
            contrasts,
            [
                "Intersection − student (64)",
                "Intersection − top-k (64)",
                "Teacher − student (64)",
                "High − random (16)",
                "Low − random (16)",
            ],
            "I1/I2: sampling and state selection",
            "mean_difference",
        ),
        (
            axes[1],
            nrows,
            ["B16 × 128", "B64 × 32", "B128 × 16", "B512 × 4", "B512 × 8"],
            "N1/N2: warm branches versus start",
            "difference_from_baseline",
        ),
    ]:
        means = np.array([r[field] for r in rows])
        lo = np.array([r["paired_prompt_ci"][0] for r in rows])
        hi = np.array([r["paired_prompt_ci"][1] for r in rows])
        ax.errorbar(
            means,
            np.arange(5),
            xerr=[means - lo, hi - means],
            fmt="o",
            color=orange,
            capsize=0,
            label="Paired mean / 95% CI",
        )
        ax.axvline(0, color=gray, lw=0.8, label="Zero difference")
        ax.set(
            title=title,
            yticks=range(5),
            yticklabels=names,
            xlabel="Fresh sequence KL difference (nats)",
            ylabel="Comparison",
            xlim=(-30, 35),
        )
        ax.invert_yaxis()
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.065),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    values["fresh-contrasts"] = dict(intersection=contrasts, batch=nrows)
    save(fig, "fresh-contrasts", "95% paired prompt intervals; negative favors intervention; one training seed.")
    (a.out / "plot-data.json").write_text(
        json.dumps(dict(sources=sorted(set(sources)), figures=values), indent=2) + "\n"
    )
    (a.out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
