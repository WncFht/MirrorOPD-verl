# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Render observed MDPO pilot/sweep metrics without running models.

Semantic ledger: rollout plots use one batch per outer, colors encode eta;
optimization plots hold eta=.1 and facet kappa, colors encode outer; positions
are response-relative deciles of the MDPO objective, not teacher KL. Evaluation
bars use 64 questions per dataset and colors encode dataset. Missing results
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
    fig.legend(legend.values(), legend.keys(), loc="lower center", bbox_to_anchor=(0.54, 0.10), ncol=3, frameon=False)
    fig.suptitle(data["models"] + "\nTraining rollouts across MDPO settings", y=0.95)
    fig.text(
        0.5,
        0.035,
        "64 new responses per outer; these are training batches, not validation.\n"
        "Open markers: unfinished arm; missing future points are not imputed.",
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
    order = ["k0-eta01", "k1-eta01", "k0-eta02", "k1-eta02"]
    lookup = {(r["arm"], r["dataset"]): r for r in data["evaluation"]}
    assert all((arm, dataset) in lookup for arm in order for dataset in ("math", "dapo"))
    specs = [
        ("Accuracy", "accuracy", 100, "Correct (%)", 100),
        ("Correct and stopped", "correct_and_stopped", 100 / 64, "Responses (%)", 100),
        ("Mean length", "mean_length", 0.001, "Response length (k)", 32),
        ("Median length", "median_length", 0.001, "Response length (k)", 32),
        ("Response cap hits", "cap_count", 100 / 64, "Responses (%)", 100),
        ("Repeated 8-grams", "repeat8_mean", 100, "Repeated windows (%)", 100),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(9, 6.7))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.29, top=0.82, wspace=0.42, hspace=0.55)
    for ax, (title, metric, scale, ylabel, upper) in zip(axes.flat, specs, strict=True):
        for offset, dataset, label, color in [
            (-0.18, "math", "MATH (64)", fv.HUES["vteal"]),
            (0.18, "dapo", "DAPO (64)", fv.HUES["vorange"]),
        ]:
            ys = [lookup[(arm, dataset)][metric] * scale for arm in order]
            ax.bar([i + offset for i in range(4)], ys, width=0.34, label=label, color=color)
        ax.set(
            title=title,
            ylabel=ylabel,
            ylim=(0, upper),
            xticks=range(4),
            xticklabels=["0/.1", "1/.1", "0/.2", "1/.2"],
            xlabel="kappa / eta",
        )
        ax.grid(axis="y", alpha=0.2)
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.11), ncol=2, frameon=False
    )
    fig.suptitle(data["models"] + "\nFinal-checkpoint evaluation: first four arms", y=0.95)
    fig.text(
        0.5,
        0.035,
        "Outer 5 / 80 updates; 64 questions per dataset, one response each; cap 31,744.\n"
        "Single-seed pilot; sweep-arm evaluation has not run yet.",
        ha="center",
        fontsize=10,
    )
    # These panels measure different outcomes, not replicated facets.
    save(fig, out, "final-evaluation", fv, different_objectives=True)
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
    for plot in (rollout_plot, optimization_plot, evaluation_plot):
        plot(data, args.out, fv, plt)


if __name__ == "__main__":
    main()
