# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Render observed MDPO pilot/sweep metrics without running models.

Semantic ledger: rollout plots use one batch per outer, colors encode eta;
optimization plots hold eta=.1 and facet kappa, colors encode outer; positions
are response-relative deciles of the MDPO objective, not teacher KL. Evaluation
heatmaps use 64 questions per dataset; untested combinations stay masked. Missing results
stay absent. All optimizer observations precede their corresponding update.
Geometry ledger: rollout facets share y scales, teacher-KL facets share y scales;
MDPO objective and position facets deliberately have distinct scales because
their targets/constants differ. Bar axes start at zero; no smoothing or CI
claims are made. Every drawn value is traceable to the frozen input JSON.
"""

import argparse
import json
from pathlib import Path

from examples.mirroropd.research.core.plotting import load_style


def save(fig, out, name, fv, *, different_objectives=False):
    fig.canvas.draw()
    # Only objective/position facets differ in mathematical target and scale.
    issues = fv.audit(fig, ignore=["facet-ylim-mismatch"] if different_objectives else [])
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"{name}.{suffix}", dpi=200)
    (out / f"{name}-audit.json").write_text(json.dumps(issues, indent=2) + "\n")
    print(name, issues)


def rollout_plot(data, out, fv, plt):
    fig, axes = plt.subplots(2, 3, figsize=(9, 6.7), sharex=True, sharey="row")
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.27, top=0.80, wspace=0.22, hspace=0.30)
    eta_colors = {0.05: fv.HUES["vteal"], 0.1: fv.HUES["vorange"], 0.2: fv.HUES["vviolet"], 0.3: fv.HUES["vblue"]}
    legend = {}
    for col, kappa in enumerate((0, 0.5, 1)):
        axes[0, col].set_title(f"kappa = {kappa:g}")
        for arm in data["arms"]:
            if arm["kappa"] != kappa or not arm["rollouts"]:
                continue
            rs = arm["rollouts"]
            x = [r["outer"] for r in rs]
            label = f"eta = {arm['eta']:g}"
            color = eta_colors[arm["eta"]]
            for ax, y in [
                (axes[0, col], [r["length"] / 1000 for r in rs]),
                (axes[1, col], [r["cap_count"] / 64 * 100 for r in rs]),
            ]:
                (line,) = ax.plot(
                    x,
                    y,
                    marker="o",
                    color=color,
                    label=label,
                    markerfacecolor=color if arm["training_complete"] else "white",
                    linewidth=1.8,
                )
                legend[label] = line
        axes[0, col].set(ylim=(0, 16.384))
        axes[1, col].set(xlabel="Outer step", xticks=[3, 4, 5], xlim=(2.8, 5.2), ylim=(0, 100))
        for ax in axes[:, col]:
            ax.grid(axis="y", alpha=0.25)
    axes[0, 0].set_ylabel("Mean response length (k)")
    axes[1, 0].set_ylabel("Responses at 16k cap (%)")
    fig.legend(legend.values(), legend.keys(), loc="lower center", bbox_to_anchor=(0.54, 0.10), ncol=4, frameon=False)
    fig.suptitle(data["models"] + "\nTraining rollouts across MDPO settings", y=0.95)
    fig.text(
        0.5,
        0.035,
        "64 new responses per outer; these are training batches, not validation.\n"
        "All ten arms completed; each restores the same outer-2 model and optimizer.",
        ha="center",
        fontsize=10,
    )
    save(fig, out, "training-length", fv)
    plt.close(fig)


def optimization_plot(data, out, fv, plt):
    fig, axes = plt.subplots(3, 3, figsize=(9, 9.3))
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.23, top=0.87, wspace=0.52, hspace=0.60)
    colors = {3: fv.HUES["vteal"], 4: fv.HUES["vorange"], 5: fv.HUES["vviolet"]}
    for col, kappa in enumerate((0, 0.5, 1)):
        arm = next(a for a in data["arms"] if a["kappa"] == kappa and a["eta"] == 0.1)
        assert arm["training_complete"] and len(arm["updates"]) == 48
        axes[0, col].set_title(f"kappa = {kappa:g}, eta = 0.1")
        for outer, color in colors.items():
            rows = [d for d in arm["updates"] if d["step/outer"] == outer]
            for ax, metric in [(axes[0, col], "distill/mdpo/objective"), (axes[1, col], "distill/teacher_kl")]:
                ax.plot(
                    [d["step/update"] for d in rows], [d[metric] for d in rows], color=color, label=f"Outer {outer}"
                )
                ax.set(xlabel="Optimizer update", xticks=[33, 48, 64, 80])
                ax.grid(alpha=0.2)
        axes[0, col].set_ylabel("MDPO objective")
        axes[1, col].set(ylabel="Teacher reverse KL", ylim=(0.10, 0.20))
        rows = [d for d in arm["updates"] if d["step/outer"] == 5]
        keys = [f"distill/mdpo/objective_position/p{i:02d}_{i + 10}/mean" for i in range(0, 100, 10)]
        for row, style, label in [(rows[0], "--", "First update"), (rows[-1], "-", "16th update")]:
            axes[2, col].plot(range(5, 100, 10), [row[k] for k in keys], color=colors[5], linestyle=style, label=label)
        axes[2, col].set(xlabel="Response position (%)", ylabel="MDPO by position", xticks=[0, 50, 100])
        axes[2, col].grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    position_handles, position_labels = axes[2, 0].get_legend_handles_labels()
    fig.legend(
        handles + position_handles,
        labels + position_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.095),
        ncol=5,
        frameon=False,
    )
    fig.suptitle(data["models"] + "\nOptimization on fixed batches: eta = 0.1", y=0.97)
    fig.text(
        0.5,
        0.018,
        "Top/bottom: different MDPO targets, so absolute levels are not directly comparable.\n"
        "Bottom: outer 5, first (dashed) vs 16th (solid) pre-update observation; not teacher KL.",
        ha="center",
        fontsize=10,
    )
    save(fig, out, "optimization", fv, different_objectives=True)
    plt.close(fig)


def evaluation_plot(data, out, fv, plt):
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap, to_rgb

    specs = [
        ("accuracy", 100, "Accuracy (%)", 100, "vteal", ".1f"),
        ("cap_count", 100 / 64, "Cap hits (%)", 100, "vorange", ".1f"),
        ("mean_length", 0.001, "Mean length (k)", 32, "vviolet", ".2f"),
    ]
    kappas, etas = [0, 0.5, 1], [0.05, 0.1, 0.2, 0.3]
    lookup = {(r["kappa"], r["eta"], r["dataset"]): r for r in data["evaluation"]}
    fig, axes = plt.subplots(2, 3, figsize=(9, 6.8))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.20, top=0.81, wspace=0.40, hspace=0.55)
    for row, dataset in enumerate(["math", "dapo"]):
        for col, (metric, scale, title, high, hue, number_format) in enumerate(specs):
            ax = axes[row, col]
            matrix = np.full((3, 4), np.nan)
            for i, kappa in enumerate(kappas):
                for j, eta in enumerate(etas):
                    if (kappa, eta, dataset) in lookup:
                        matrix[i, j] = lookup[(kappa, eta, dataset)][metric] * scale
            endpoint = tuple(0.65 * channel + 0.35 for channel in to_rgb(fv.HUES[hue]))
            cmap = LinearSegmentedColormap.from_list(metric, ["white", endpoint])
            cmap.set_bad((0.92, 0.92, 0.92, 1))
            ax.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=high, cmap=cmap, aspect="auto")
            for i in range(3):
                for j in range(4):
                    value = matrix[i, j]
                    ax.text(
                        j,
                        i,
                        "—" if np.isnan(value) else format(value, number_format),
                        ha="center",
                        va="center",
                        color=fv.NEUTRALS["vink"],
                    )
            ax.set(
                title=f"{dataset.upper()}: {title}",
                xlabel="eta",
                ylabel="kappa",
                xticks=range(4),
                xticklabels=[".05", ".1", ".2", ".3"],
                yticks=range(3),
                yticklabels=["0", ".5", "1"],
            )
    fig.suptitle(data["models"] + "\nCompleted grid: final-checkpoint evaluation", y=0.96)
    fig.text(
        0.5,
        0.045,
        "64 questions per dataset; one response each; cap 31,744. Gray cells were not tested.\n"
        "Color scales: accuracy/cap 0–100%; length 0–32k. Single-seed exploratory comparison.",
        ha="center",
        fontsize=10,
    )
    save(fig, out, "final-evaluation", fv, different_objectives=True)
    plt.close(fig)


def paired_plot(data, out, fv, plt):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.8))
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.25, top=0.83, wspace=0.85, hspace=0.50)
    labels = [
        "k1 - k0; eta=.1",
        "k.5 - k0; eta=.1",
        "k1 - k0; eta=.2",
        "k.5 - k0; eta=.2",
        "k1; eta .05 - .1",
        "k.5; eta .05 - .1",
    ]
    specs = [
        ("accuracy", "Accuracy difference (pp)", 100),
        ("correct_stopped", "Correct-and-stopped difference (pp)", 100),
        ("cap_rate", "Cap-rate difference (pp)", 100),
        ("mean_length", "Mean-length difference (k tokens)", 0.001),
    ]
    for ax, (metric, title, scale) in zip(axes.flat, specs, strict=True):
        for prefix, label, hue in [("k1-", "Candidate kappa=1", "vviolet"), ("k05-", "Candidate kappa=0.5", "vorange")]:
            positions, points, lows, highs = [], [], [], []
            for i, comparison in enumerate(data["paired_comparisons"]):
                if not comparison["candidate"].startswith(prefix):
                    continue
                v = comparison["metrics"][metric]
                point = v["difference"] * scale
                low, high = [x * scale for x in v["ci95"]]
                positions.append(i)
                points.append(point)
                lows.append(point - low)
                highs.append(high - point)
            ax.errorbar(points, positions, xerr=[lows, highs], fmt="o", color=fv.HUES[hue], capsize=3, label=label)
        ax.axvline(0, color=fv.NEUTRALS["baseline"], lw=1, ls=":")
        ax.set(xlabel=title, yticks=range(6), yticklabels=labels, ylim=(5.6, -0.6))
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Paired-question differences with 95% bootstrap intervals", y=0.95)
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.55, 0.10), ncol=2, frameon=False
    )
    fig.text(
        0.5,
        0.035,
        "Candidate minus control; 10,000 resamples, stratified by MATH/DAPO.\n"
        "Intervals exclude training-seed uncertainty and do not adjust for screening or multiple comparisons.",
        ha="center",
        fontsize=10,
    )
    save(fig, out, "paired-intervals", fv, different_objectives=True)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fv = load_style()
    fv.apply_rc("paper", fontsize=12)
    data = json.loads(args.data.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    for plot in (rollout_plot, optimization_plot, evaluation_plot, paired_plot):
        plot(data, args.out, fv, plt)


if __name__ == "__main__":
    main()
