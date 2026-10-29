# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Ledger: final risk reductions after 64 updates, one figure per student.

Each teacher row is one independent training arm. The two panels measure RKL
and JS on the same fixed train A-D bank and held-out math bank. Each bar is
100*(1-final/initial) for its own teacher/panel, not accuracy or recovered task
capability. Bars show point estimates; prompt-bootstrap intervals remain in the
source CSV. Teacher order follows lineage groups rather than measured success.
"""

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ...common.plotting import load_style
from .common import OUT, REPO, dump

ORDERS = {
    "r1": ["TB1-J", "TB1-N", "TB2-M1", "TB1-R", "TB1-P", "TB1-S", "TB2-L1"],
    "r7": ["TB2-P7", "TB2-L7"],
}
LABELS = {
    "TB1-J": "JustRL-1.5B",
    "TB1-N": "Nemotron v1",
    "TB2-M1": "MiniMath-1.5B",
    "TB1-R": "R1-Distill-7B",
    "TB1-P": "Polaris-7B",
    "TB1-S": "Skywork-7B",
    "TB2-L1": "Light-R1-7B",
    "TB2-P7": "Polaris-7B",
    "TB2-L7": "Light-R1-7B",
}


def main():
    source = OUT / "analysis/endpoints.csv"
    with source.open() as stream:
        rows = list(csv.DictReader(stream))
    index = {(r["arm"], r["panel"], r["metric"]): r for r in rows}
    output = (
        REPO
        / "examples/mirroropd/docs/experiments/teacher-transfer/transfer-mechanism/figures/boundary-endpoints-20261029"
    )
    output.mkdir(parents=True, exist_ok=True)
    fv = load_style()
    fv.apply_rc("paper", fontsize=10)
    audits, plotted = {}, []
    panels = [("train_ad", "Train A-D (64)", fv.HUES["vteal"]), ("held_math", "Held-out math (64)", fv.HUES["vorange"])]
    for student, arms in ORDERS.items():
        fig, axes = plt.subplots(1, 2, figsize=(8, 6 if student == "r1" else 4), sharey=True)
        fig.subplots_adjust(left=0.23, right=0.97, top=0.86, bottom=0.34 if student == "r7" else 0.27, wspace=0.17)
        for ax, metric in zip(axes, ["rkl", "js"], strict=True):
            for offset, (panel, label, color) in zip([-0.18, 0.18], panels, strict=True):
                selected = [index[(arm, panel, metric)] for arm in arms]
                values = np.array([float(r["drop_percent"]) for r in selected])
                lows = np.array([float(r["ci_low"]) for r in selected])
                highs = np.array([float(r["ci_high"]) for r in selected])
                assert np.all(lows <= values) and np.all(highs >= values)
                positions = np.arange(len(arms)) + offset
                ax.barh(positions, values, height=0.30, color=color, label=label)
                for y, value, row in zip(positions, values, selected, strict=True):
                    ax.text(value + 1.5, y, f"{value:.1f}", va="center", color=fv.NEUTRALS["vink"])
                    plotted.append(dict(student=student, **row))
            ax.set(
                title=metric.upper(),
                xlabel="Reduction from initial (%)",
                xlim=(-2, 108),
                xticks=[0, 25, 50, 75, 100],
                yticks=np.arange(len(arms)),
                yticklabels=[LABELS[arm] for arm in arms],
            )
            ax.axvline(0, lw=0.7, color=fv.NEUTRALS["baseline"])
        axes[0].invert_yaxis()
        axes[0].set_ylabel("Teacher")
        size = "1.5B" if student == "r1" else "7B"
        fig.suptitle(f"Student: DeepSeek-R1-Distill-Qwen-{size}", y=0.97)
        fig.legend(
            *axes[0].get_legend_handles_labels(),
            loc="lower center",
            bbox_to_anchor=(0.60, 0.105),
            ncol=2,
            frameon=False,
        )
        fig.text(
            0.5,
            0.018,
            "64 updates; frozen S0; H=1024. Final distribution fit, not task accuracy.\n"
            "Point estimates; one training seed. Prompt-bootstrap intervals are in the source CSV.",
            ha="center",
            color=fv.NEUTRALS["baseline"],
            fontsize=9,
        )
        fig.canvas.draw()
        audits[student] = [str(p) for p in fv.audit(fig)]
        for ext in ["png", "pdf"]:
            fig.savefig(output / f"{student}-endpoints.{ext}", dpi=180)
        plt.close(fig)
    with (output / "plotted-data.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(plotted[0]))
        writer.writeheader()
        writer.writerows(plotted)
    assert len(plotted) == 9 * 2 * 2
    dump(output / "audit.json", dict(source=str(source), expected_arms=9, plotted_rows=len(plotted), figures=audits))
    print(output)
    print(audits)


if __name__ == "__main__":
    main()
