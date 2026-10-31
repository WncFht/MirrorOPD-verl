# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Plot observed batch-normalization factors using figure-viz.

Semantic ledger: one series is one resolved online full-RKL run. Each point
represents C / mean response length for one new rollout batch, where C is the
run's median batch mean. This is raw loss/gradient scaling at fixed parameters,
not an observed Adam update or an accuracy effect. Both panels share axes units
and run colors; the right panel only contains the first twelve batches.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--style", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.style.resolve()))
    import fvstyle as fv

    fv.apply_rc("paper")
    with (args.data / "outer-series.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    labels = ["E17", "E35", "E39", "E54"]
    colors = dict(zip(labels, fv.series_colors("paper", 4), strict=True))
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.9), sharey=True)
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.36, top=0.88, wspace=0.20)
    for index, ax in enumerate(axes):
        for label in labels:
            points = [r for r in rows if r["label"] == label and (index == 0 or int(r["outer"]) <= 12)]
            ax.plot(
                [int(r["outer"]) for r in points],
                [float(r["relative_scale"]) for r in points],
                color=colors[label],
                label=label,
                linewidth=0.9,
            )
        ax.axhline(1, color=fv.NEUTRALS["baseline"], linestyle="--", linewidth=0.7)
        ax.set_yscale("log")
        ax.set_ylim(0.7, 9)
        ax.set_yticks([1, 2, 4, 8], labels=["1", "2", "4", "8"])
        ax.set_xlabel("Rollout batch")
        ax.grid(axis="y", color=fv.NEUTRALS["vmist"], linewidth=0.5)
        ax.set_title("(a) All recorded batches" if index == 0 else "(b) First 12 batches")
    axes[0].set_xlim(0, 250)
    axes[0].set_xticks([0, 100, 200])
    axes[0].set_ylabel("Relative raw-gradient scale")
    axes[1].set_xlim(0, 12.5)
    axes[1].set_xticks([1, 4, 8, 12])
    fig.legend(
        *axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.54, 0.10), ncol=4, frameon=False
    )
    fig.text(
        0.5,
        0.02,
        "Run-median length / batch-mean length; raw scaling before gradient clipping.",
        ha="center",
        color=fv.NEUTRALS["baseline"],
        fontsize="small",
    )
    fig.canvas.draw()
    problems = fv.audit(fig)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "token-mean-scale.audit.json").write_text(json.dumps(problems, indent=2) + "\n")
    for problem in problems:
        print(problem)
    if any(fv.problem_severity(problem) == "error" for problem in problems):
        raise RuntimeError("Figure audit found errors")
    for extension in ("png", "pdf", "svg"):
        fig.savefig(args.output / f"token-mean-scale.{extension}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
