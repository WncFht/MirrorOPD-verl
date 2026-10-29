# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Ledger: one row per teacher, latest endpoint, separate figure per student.

Bars encode 100*(1-final/initial) on fixed A-D or held-out math, with each
teacher's own S0 risk as denominator. The row label reports actual updates;
128-update continuations are not equal-budget controls of 64-update arms.
No line connects intermediate checkpoints. CSV retains all plotted values.
"""

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ...common.plotting import load_style
from .analyze import csv_write
from .round2_common import OLD_OUT, OUT, REPO, dump, read

ORDERS = {
    "r1": ["CONT-J", "CONT-N", "CONT-M", "NEW-DS", "NEW-BRO", "NEW-DC", "TB1-R", "TB1-P", "TB1-S", "TB2-L1"],
    "r7": ["TB2-P7", "TB2-L7"],
}
LABELS = {
    "CONT-J": "JustRL (128)",
    "CONT-N": "Nemotron v1 (128)",
    "CONT-M": "MiniMath (128)",
    "NEW-DS": "DeepScaleR (64)",
    "NEW-BRO": "BroRL (64)",
    "NEW-DC": "DeepCoder (64)",
    "TB1-R": "R1-Distill-7B (64)",
    "TB1-P": "Polaris-7B (64)",
    "TB1-S": "Skywork-7B (64)",
    "TB2-L1": "Light-R1-7B (64)",
    "TB2-P7": "Polaris-7B (64)",
    "TB2-L7": "Light-R1-7B (64)",
}


def main():
    with (OLD_OUT / "analysis/endpoints.csv").open() as stream:
        old = list(csv.DictReader(stream))
    new = read(OUT / "analysis/summary.json")["endpoints"]
    index = {(r["arm"], r["panel"], r["metric"]): r for r in old + new}
    output = REPO / "examples/mirroropd/docs/experiments/teacher-transfer/transfer-mechanism/figures/boundary-round2"
    output.mkdir(parents=True, exist_ok=True)
    fv = load_style()
    fv.apply_rc("paper", fontsize=10)
    plotted, audits = [], {}
    for student, arms in ORDERS.items():
        large = student == "r1"
        fig, axes = plt.subplots(1, 2, figsize=(9, 7.6 if large else 4.2), sharey=True)
        fig.subplots_adjust(
            left=0.26, right=0.97, top=0.89 if large else 0.83, bottom=0.19 if large else 0.35, wspace=0.17
        )
        for ax, metric in zip(axes, ["rkl", "js"], strict=True):
            for offset, panel, label, color in [
                (-0.18, "train_ad", "Fixed A-D (64)", fv.HUES["vteal"]),
                (0.18, "held_math", "Held-out math (64)", fv.HUES["vorange"]),
            ]:
                values = [float(index[(arm, panel, metric)]["drop_percent"]) for arm in arms]
                y = np.arange(len(arms)) + offset
                ax.barh(y, values, height=0.30, color=color, label=label)
                for arm, pos, value in zip(arms, y, values, strict=True):
                    ax.text(
                        value + 1.5 if value >= 0 else value - 1.5,
                        pos,
                        f"{value:.1f}",
                        va="center",
                        ha="left" if value >= 0 else "right",
                        color=fv.NEUTRALS["vink"],
                    )
                    row = index[(arm, panel, metric)]
                    plotted.append(
                        dict(
                            student=student,
                            arm=arm,
                            label=LABELS[arm],
                            panel=panel,
                            metric=metric,
                            initial=float(row["initial"]),
                            final=float(row["final"]),
                            drop_percent=value,
                            ci_low=float(row["ci_low"]),
                            ci_high=float(row["ci_high"]),
                        )
                    )
            minimum = min(r["drop_percent"] for r in plotted if r["student"] == student and r["metric"] == metric)
            ax.set(
                title=metric.upper(),
                xlabel="Reduction from S0 (%)",
                xlim=(min(-2, minimum - 15), 108),
                yticks=np.arange(len(arms)),
                yticklabels=[LABELS[a] for a in arms],
            )
            ax.axvline(0, lw=0.7, color=fv.NEUTRALS["baseline"])
        axes[0].invert_yaxis()
        axes[0].set_ylabel("Teacher (updates)")
        size = "1.5B" if large else "7B"
        fig.suptitle(f"Student: DeepSeek-R1-Distill-Qwen-{size}", y=0.97)
        fig.legend(
            *axes[0].get_legend_handles_labels(),
            loc="lower center",
            bbox_to_anchor=(0.60, 0.075),
            ncol=2,
            frameon=False,
        )
        fig.text(
            0.5,
            0.013,
            "Latest endpoints; frozen S0; H=1024; one training seed.\n"
            "Update budgets are shown per row. Distribution fit, not task accuracy.",
            ha="center",
            fontsize=9,
            color=fv.NEUTRALS["baseline"],
        )
        fig.canvas.draw()
        audits[student] = [str(p) for p in fv.audit(fig)]
        for ext in ["png", "pdf"]:
            fig.savefig(output / f"{student}-latest-endpoints.{ext}", dpi=180)
        plt.close(fig)
    csv_write(output / "plotted-data.csv", plotted)
    assert len(plotted) == 12 * 4
    dump(output / "audit.json", dict(figures=audits, plotted_rows=len(plotted)))
    print(audits)


if __name__ == "__main__":
    main()
