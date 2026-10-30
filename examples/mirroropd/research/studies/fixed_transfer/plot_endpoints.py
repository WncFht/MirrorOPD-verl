# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Ledger: one row per completed teacher arm; latest available evaluated KL endpoint.

Each horizontal bar encodes percent reduction from that teacher's initial risk.
Panels are RKL and JS, colors are fixed A-D versus held-out math. All rows use
one frozen student bank. Labels state update budgets; no intermediate dynamics
or task accuracy is plotted. Input analysis CSV and plotted rows remain auditable.
"""

import argparse
import csv
from pathlib import Path

from examples.mirroropd.research.core.io import dump, write_csv
from examples.mirroropd.research.core.plotting import load_style
from examples.mirroropd.research.studies.fixed_transfer.display_policy import MAX_DISPLAY_UPDATES, in_main_comparison
from examples.mirroropd.research.studies.fixed_transfer.spec import Study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    study = Study.load(args.plan)
    with (study.directory / "analysis/endpoints.csv").open() as stream:
        rows = [r for r in csv.DictReader(stream) if int(r["step"]) <= MAX_DISPLAY_UPDATES]
    arms = [a for a in study.arms if in_main_comparison(a.teacher) and any(r["arm"] == a.name for r in rows)]
    if not arms:
        raise ValueError("No completed KL endpoints")
    steps = {a.name: max(int(r["step"]) for r in rows if r["arm"] == a.name) for a in arms}
    selected = [r for r in rows if int(r["step"]) == steps[r["arm"]]]
    index = {(r["arm"], r["panel"], r["metric"]): r for r in selected}
    arms.sort(key=lambda arm: float(index[arm.name, "held_math", "rkl"]["drop_percent"]), reverse=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fv = load_style()
    fv.apply_rc("paper", fontsize=10)
    fig, axes = plt.subplots(1, 2, figsize=(10, max(5, len(arms) * 0.65 + 2)), sharey=True)
    fig.subplots_adjust(left=0.24, right=0.97, bottom=0.24, top=0.85, wspace=0.17)
    plotted = []
    for ax, metric in zip(axes, ["rkl", "js"], strict=True):
        for offset, panel, label, color in [
            (-0.18, "train_ad", "Fixed A-D (64)", fv.HUES["vteal"]),
            (0.18, "held_math", "Held-out math (64)", fv.HUES["vorange"]),
        ]:
            chosen = [index[(a.name, panel, metric)] for a in arms]
            values = [float(r["drop_percent"]) for r in chosen]
            y = np.arange(len(arms)) + offset
            ax.barh(y, values, height=0.30, color=color, label=label)
            for pos, value in zip(y, values, strict=True):
                ax.text(value + 1.2, pos, f"{value:.1f}", va="center", color=fv.NEUTRALS["vink"])
            plotted.extend(chosen)
        minimum = min(float(r["drop_percent"]) for r in plotted if r["metric"] == metric)
        ax.set(title=metric.upper(), xlabel="Reduction from S0 (%)", xlim=(min(0, minimum - 8), 112))
        ax.set_yticks(np.arange(len(arms)), [f"{a.teacher} ({steps[a.name]})" for a in arms])
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].invert_yaxis()
    fig.suptitle(f"Student: {study.student} | completed teacher endpoints", y=0.96)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.58, 0.10), ncol=2)
    fig.text(
        0.5,
        0.015,
        "Frozen S0; H=1024; one training seed. Row labels show updates.\n"
        "Distribution fit, not task accuracy; pending teachers excluded.",
        ha="center",
        fontsize=9,
        color=fv.NEUTRALS["baseline"],
    )
    fig.canvas.draw()
    # The right panel shares teacher rows and their visible labels with the left panel.
    problems = fv.audit(fig, scoped_ignore=[("missing-axis-label", 1)])
    audit = [str(p) for p in problems]
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "plotted-data.csv", plotted)
    dump(args.output / "audit.json", dict(problems=audit, study=study.identity))
    if any(fv.problem_severity(problem) == "error" for problem in problems):
        plt.close(fig)
        raise RuntimeError("Endpoint figure audit failed; see audit.json")
    for ext in ["png", "pdf"]:
        fig.savefig(args.output / f"latest-endpoints.{ext}", dpi=180)
    plt.close(fig)
    print(audit)


if __name__ == "__main__":
    main()
