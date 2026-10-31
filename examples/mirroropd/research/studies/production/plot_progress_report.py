# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Render weekly report figures from frozen measured CSV rows.

Semantic ledger: each point is one recorded task mean, conditional on one run;
outer is a generation batch and updates is the applied optimizer count. Signal
panels compare four methods within each budget, and do not identify the effect
of changing budgets. Lines connect observed checkpoints without interpolation.
Geometric ledger: all accuracy axes use percentage points; matched task facets
share limits, while different tasks have independently labelled ranges. Method
colors remain constant within a figure. No synthetic data or fitted curves.
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--style", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.style))
    import fvstyle as fv
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fv.apply_rc("paper", font="sans", fontsize=10)
    args.output.mkdir(parents=True, exist_ok=True)
    colors = [fv.HUES[k] for k in ("vteal", "vorange", "vviolet", "vcoral")]
    tasks = [("math500", "MATH-500"), ("amc23", "AMC23"), ("aime25", "AIME25")]
    audits = {}

    def load(name):
        return list(csv.DictReader((args.data / name).open()))

    def finish(fig, name, source, caption):
        fig.text(0.5, 0.025, caption, ha="center", color=fv.NEUTRALS["baseline"], fontsize="small")
        fig.canvas.draw()
        # Different tasks have intentionally different accuracy scales.
        problems = fv.audit(fig, ignore=["facet-ylim-mismatch"])
        errors = [p for p in problems if fv.problem_severity(p) == "error"]
        audits[name] = {"problems": problems, "errors": errors}
        digest = hashlib.sha256((args.data / source).read_bytes()).hexdigest()
        fig.savefig(args.output / f"{name}.png", dpi=220, bbox_inches="tight", metadata={"SourceSHA256": digest})
        fig.savefig(args.output / f"{name}.pdf", bbox_inches="tight", metadata={"Subject": digest})
        plt.close(fig)

    signal = load("sampled-signal.csv")
    methods = ["rollout IS", "current PPO", "current IS", "current NoIS"]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6), sharey="col")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.87, bottom=0.23, hspace=0.5, wspace=0.3)
    for row, outer in enumerate([32, 200]):
        for col, (task, title) in enumerate(tasks):
            ax = axes[row, col]
            fv.apply_frame(ax, "open")
            fv.apply_grid(ax, "y-only")
            selected = [r for r in signal if int(r["outer"]) == outer]
            for i, record in enumerate(selected):
                ax.plot(i, float(record[task]), "o", color=colors[i], label=methods[i], markersize=7)
            ax.set_xticks(range(4), [r["experiment"] for r in selected])
            ax.set_xlim(-0.5, 3.5)
            ax.set_ylim((80, 93) if col < 2 else (28, 42))
            ax.set_xlabel("Experiment")
            ax.set_ylabel("Mean accuracy (%)")
            if row == 0:
                ax.set_title(title)
            ax.text(
                0.03, 0.93, "32 outer / reuse4" if row == 0 else "200 outer / reuse16", transform=ax.transAxes, va="top"
            )
    handles = [
        Line2D([], [], marker="o", linestyle="none", color=c, label=m) for c, m in zip(colors, methods, strict=True)
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, 0.09), ncol=4)
    finish(
        fig,
        "rq1-sampled-signals",
        "sampled-signal.csv",
        "Observed final means; n=16 answers/question; one training seed; budgets differ across rows.",
    )

    baseline = load("baseline-evaluations.csv")
    comparisons = [
        ("rq1-full-vs-sampled", [("E13", "Full RKL / reuse1"), ("E15", "Sampled vanilla / reuse1")]),
        ("rq2-reuse-fixed-outer", [("E13", "E13: reuse1"), ("E16", "E16: reuse4"), ("E17", "E17: reuse16")]),
        ("rq2-learning-rate", [("E13", "E13: LR 1e-6"), ("E14", "E14: LR 5e-7")]),
    ]
    for name, arms in comparisons:
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))
        fig.subplots_adjust(left=0.08, right=0.98, top=0.87, bottom=0.35, wspace=0.32)
        for ax, (task, title) in zip(axes, tasks, strict=True):
            fv.apply_frame(ax, "open")
            fv.apply_grid(ax, "y-only")
            for (experiment, label), color in zip(arms, colors, strict=False):
                rs = sorted(
                    [r for r in baseline if r["experiment"] == experiment and r["task"] == task],
                    key=lambda r: int(r["outer"]),
                )
                ax.plot(
                    [int(r["outer"]) for r in rs],
                    [float(r["accuracy_percent"]) for r in rs],
                    color=color,
                    marker="o",
                    markersize=3,
                    label=label,
                )
            ax.set_title(title)
            ax.set_xlabel("Generation batches (outer)")
            ax.set_ylabel("Mean accuracy (%)")
            ax.set_xticks([20, 80, 140, 200])
        fig.legend(
            *axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.53, 0.11), ncol=len(arms)
        )
        finish(
            fig,
            name,
            "baseline-evaluations.csv",
            "Recorded checkpoints through outer200; n=16 answers/question; one training seed per recipe.",
        )
    (args.output / "new-figures-audit.json").write_text(json.dumps(audits, indent=2) + "\n")
    errors = {k: v["errors"] for k, v in audits.items() if v["errors"]}
    if errors:
        raise RuntimeError(errors)
    print(f"Rendered {len(audits)} figures; audit errors = 0")


if __name__ == "__main__":
    main()
