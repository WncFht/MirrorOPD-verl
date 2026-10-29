# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Ledger: observed update counts versus fixed-panel risks, normalized per arm.

Panels are teacher/student pairs, not repeated seeds. Colors encode the same
train/future/heldout panels throughout. All markers are real score checkpoints;
connecting lines interpolate visually only. Baseline heatmap cells are avg@4
task accuracy of original models, never trained student endpoint accuracy.
"""

import csv

from examples.mirroropd.research.core.io import dump
from examples.mirroropd.research.core.paths import REPO
from examples.mirroropd.research.core.plotting import load_style
from examples.mirroropd.research.studies.boundary.common import ARMS, DATASETS, MODELS, OUT


def main():
    import argparse

    argparse.ArgumentParser(description=__doc__).parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    fv = load_style()
    fv.apply_rc("paper", fontsize=10)
    out = REPO / "examples/mirroropd/docs/experiments/teacher-transfer/transfer-mechanism/figures/boundary-20261028"
    out.mkdir(parents=True, exist_ok=True)
    with (OUT / "analysis/risk-curves.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    audits = {}
    colors = {"train_ad": fv.HUES["vteal"], "future_eh": fv.NEUTRALS["baseline"], "held_math": fv.HUES["vorange"]}
    labels = {"train_ad": "Train A-D", "future_eh": "Untrained E-H", "held_math": "Held-out math"}

    def save(fig, name, caption):
        fig.text(0.5, 0.025, caption, ha="center", color=fv.NEUTRALS["baseline"], fontsize=9)
        fig.canvas.draw()
        audits[name] = [str(p) for p in fv.audit(fig)]
        for ext in ["png", "pdf"]:
            fig.savefig(out / f"{name}.{ext}", dpi=180)
        plt.close(fig)

    for prefix, shape in [("TB1", (2, 3)), ("TB2", (2, 2))]:
        for metric in ["rkl", "js"]:
            arms = [a for a in ARMS if a.startswith(prefix)]
            fig, axes = plt.subplots(*shape, figsize=(9 if prefix == "TB1" else 8.5, 6), sharex=True, sharey=True)
            fig.subplots_adjust(left=0.09, right=0.97, bottom=0.2, top=0.92, wspace=0.25, hspace=0.4)
            active = axes.ravel()
            high = max(
                float(r["value"]) / float(r["initial"])
                for r in rows
                if r["arm"] in arms and r["panel"] in colors and r["metric"] == metric
            )
            for ax, arm in zip(active, arms, strict=False):
                student, teacher = ARMS[arm]
                for panel in colors:
                    selected = sorted(
                        [r for r in rows if r["arm"] == arm and r["panel"] == panel and r["metric"] == metric],
                        key=lambda r: int(r["step"]),
                    )
                    ax.plot(
                        [int(r["step"]) for r in selected],
                        [float(r["value"]) / float(r["initial"]) for r in selected],
                        marker="o",
                        markersize=4,
                        color=colors[panel],
                        label=labels[panel],
                    )
                ax.set(
                    title=f"{arm}: {student} <- {teacher}",
                    xlabel="Parameter updates",
                    ylabel=f"{metric.upper()} / initial",
                    xticks=[0, 16, 32, 48, 64],
                    ylim=(0, high * 1.08),
                )
                ax.tick_params(labelbottom=True)
            for ax in active[len(arms) :]:
                ax.set_visible(False)
            fig.legend(
                *active[0].get_legend_handles_labels(),
                loc="lower center",
                bbox_to_anchor=(0.5, 0.075),
                ncol=3,
                frameon=False,
            )
            save(
                fig,
                f"{prefix.lower()}-{metric}",
                "Fixed S0 prefixes; H=1024; one training seed. Train bank includes future batches early on.",
            )

    with (OUT / "analysis/baselines.csv").open() as stream:
        baseline = list(csv.DictReader(stream))
    names = list(MODELS)
    data = np.array(
        [
            [
                float(
                    next(
                        r["mean_percent"]
                        for r in baseline
                        if r["model"] == n and r["template"] == "shared" and r["dataset"] == d
                    )
                )
                for d in DATASETS
            ]
            for n in names
        ]
    )
    fig, ax = plt.subplots(figsize=(8, 5.6))
    fig.subplots_adjust(left=0.18, right=0.86, top=0.92, bottom=0.2)
    cmap = LinearSegmentedColormap.from_list("accuracy", [fv.NEUTRALS["paper"], fv.HUES["vteal"]])
    im = ax.imshow(data, vmin=0, vmax=100, cmap=cmap, aspect="auto")
    ax.set(
        xticks=range(5),
        xticklabels=DATASETS,
        yticks=range(len(names)),
        yticklabels=names,
        xlabel="Benchmark",
        ylabel="Original model",
    )
    for y in range(len(names)):
        for x in range(5):
            ax.text(x, y, f"{data[y, x]:.1f}", ha="center", va="center", color=fv.NEUTRALS["vink"])
    fig.colorbar(im, ax=ax, label="Mean accuracy (%)", fraction=0.04, pad=0.04)
    save(
        fig,
        "original-model-accuracy",
        "Common R1 prompt; avg@4; original models only. Light native-template results are reported separately.",
    )
    dump(out / "audit.json", audits)
    print(out)


if __name__ == "__main__":
    main()
