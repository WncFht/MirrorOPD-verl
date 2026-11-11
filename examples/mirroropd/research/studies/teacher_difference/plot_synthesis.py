# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Synthesis figure for the teacher/student cross-batch report."""

import csv
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
STYLE = ROOT.parent / "dotfiles/.agents/skills/figure-viz/assets/mpl"
sys.path.insert(0, str(STYLE))
import fvstyle as fv  # noqa: E402


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    os.environ["FIGURE_VIZ_MPL"] = str(STYLE)
    fv.apply_rc("paper")
    out = Path("runs/teacher-difference-synthesis-20261110")
    out.mkdir(parents=True, exist_ok=True)
    pac = read_rows(Path("runs/teacher-difference-stages-20261109/analysis/r1_pac_factors.csv"))
    crossed = read_rows(Path("runs/teacher-difference-crossed-20261109/update-analysis/test-summary.csv"))
    qwen = [
        r
        for r in read_rows(Path("runs/teacher-difference-qwen-20261109/cross-analysis/cross-batch.csv"))
        if r["role"] == "all-train" and r["field"] == "live"
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.8))
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.16, top=0.91, wspace=0.34, hspace=0.42)
    colors = {"justrl": fv.HUES["vteal"], "polaris": fv.HUES["vorange"]}
    ax = axes[0, 0]
    for teacher in ("justrl", "polaris"):
        values = [r for r in pac if r["target"] == teacher and r["split"] == "heldout/math"]
        ax.plot(
            [int(r["stage"]) for r in values],
            [float(r["PXi_over_L"]) for r in values],
            "o-",
            label=teacher,
            color=colors[teacher],
        )
    ax.set_yscale("log")
    ax.set_xlabel("Updates")
    ax.set_ylabel("$P\\Xi/L$")
    ax.set_title("R1 heldout efficiency")
    ax.legend(frameon=False, fontsize=7)
    ax = axes[0, 1]
    branches = ["justrl-on-justrl", "polaris-on-justrl", "justrl-on-polaris", "polaris-on-polaris"]
    positions = np.arange(4)
    update_colors = fv.series_colors("paper", 2)
    for j, field in enumerate(("native", "restored-adam")):
        scale = 0.003 if field == "native" else 1.0
        values = [
            next(
                r
                for r in crossed
                if r["branch"] == branch and r["field"] == field and float(r["displacement"]) == scale
            )
            for branch in branches
        ]
        ax.plot(
            positions + (j - 0.5) * 0.16,
            [100 * float(r["value"]) for r in values],
            "o-",
            label=field,
            color=update_colors[j],
        )
    ax.set_xticks(positions, ["J→J", "P→J", "J→P", "P→P"])
    ax.set_ylabel("Test KL reduction (%)")
    ax.set_title("Teacher × student update")
    ax.legend(frameon=False, fontsize=7)
    ax = axes[1, 0]
    values = [r for r in crossed if r["field"] == "restored-adam" and float(r["displacement"]) == 1.0]
    x = np.array([float(r["predicted"]) for r in values]) * 100
    y = np.array([float(r["value"]) for r in values]) * 100
    ax.scatter(
        x,
        y,
        c=[colors["justrl"] if "justrl" in r["branch"].split("-on-")[0] else colors["polaris"] for r in values],
        s=22,
    )
    extent = max(np.max(np.abs(x)), np.max(np.abs(y))) * 1.08
    ax.plot(
        [-extent, extent],
        [-extent, extent],
        "--",
        color=fv.NEUTRALS["baseline"],
        linewidth=0.8,
        label="Prediction = observed",
    )
    ax.set_xlabel("Linear prediction (%)")
    ax.set_ylabel("Observed (%)")
    ax.set_title("Historical Adam: prediction vs test")
    ax.legend(frameon=False, fontsize=7)
    ax = axes[1, 1]
    bank_colors = {
        bank: color
        for bank, color in zip(
            sorted({r["bank"] for r in qwen}), fv.series_colors("paper", len({r["bank"] for r in qwen})), strict=True
        )
    }
    ax.scatter(
        [float(r["remaining_fraction"]) for r in qwen],
        [float(r["D_over_L"]) for r in qwen],
        c=[bank_colors[r["bank"]] for r in qwen],
        s=18,
        alpha=0.85,
    )
    ax.axvline(1, color=fv.NEUTRALS["baseline"], linewidth=0.6)
    ax.axhline(0, color=fv.NEUTRALS["baseline"], linewidth=0.6)
    ax.set_xlabel("Remaining KL / initial KL")
    ax.set_ylabel("Train → heldout $D/L$")
    ax.set_title("Qwen: endpoint risk vs propagation")
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=color, label=bank) for bank, color in bank_colors.items()
    ]
    ax.legend(handles=handles, frameon=False, fontsize=6.5)
    fig.text(
        0.5,
        0.025,
        "Fixed banks; R1 step240 cross cells; Qwen 65 conditions. Source CSVs retain intervals.",
        ha="center",
        va="bottom",
        fontsize=7,
        color=fv.NEUTRALS["baseline"],
    )
    fig.canvas.draw()
    problems = fv.audit(fig, print_slot=fig.get_figwidth())
    (out / "teacher-difference-synthesis.audit.txt").write_text(
        "\n".join(fv.format_audit_lines(problems)) + "\n" if problems else "PASS: fv.audit(fig) returned no issues.\n"
    )
    if any(fv.problem_severity(p) == "error" for p in problems):
        raise RuntimeError("Figure audit failed")
    fig.savefig(out / "teacher-difference-synthesis.png", dpi=220)
    fig.savefig(out / "teacher-difference-synthesis.pdf")
    (out / "sources.json").write_text(
        json.dumps(
            {
                "r1_pac": str(Path("runs/teacher-difference-stages-20261109/analysis/r1_pac_factors.csv")),
                "crossed": str(Path("runs/teacher-difference-crossed-20261109/update-analysis/test-summary.csv")),
                "qwen": str(Path("runs/teacher-difference-qwen-20261109/cross-analysis/cross-batch.csv")),
                "audits": [fv.format_audit_lines(problems)],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
