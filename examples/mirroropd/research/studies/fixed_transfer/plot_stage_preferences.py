# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Ledger: one point is a paired student projection contrast on one fixed state panel.

Columns encode teacher-stage directions; intervals are paired prompt bootstrap
95% intervals. Projection contrasts are signed, unbounded coefficients, not percentages.
"""

import argparse
import csv
import shutil
from pathlib import Path

from examples.mirroropd.research.core.io import dump, read
from examples.mirroropd.research.core.plotting import load_style
from examples.mirroropd.research.studies.fixed_transfer.spec import Study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Study.load(args.plan).directory / "analysis/stage-preferences"
    assert read(root / "complete.json")["scientific_updates"] == 0
    with (root / "paired-stage-projection.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ["paired-stage-projection.csv", "geometry.csv", "prompt-moments.csv", "inputs.json", "complete.json"]:
        shutil.copy2(root / name, args.output / name)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fv = load_style()
    fv.apply_rc("paper", fontsize=10)
    panels = [
        ("train_all", "Training pool A-T (320)"),
        ("train_ad", "Old training A-D (64)"),
        ("held_math", "Held-out Math (64)"),
        ("held_code", "Held-out Code (32)"),
        ("held_if", "Held-out IF (32)"),
        ("held_all", "All held-out (128)"),
    ]
    plotted = [r for r in rows if r["panel"] in {key for key, _ in panels}]
    limits = (
        min(-0.05, min(float(r["ci_low"]) for r in plotted) - 0.05),
        max(1.05, max(float(r["ci_high"]) for r in plotted) + 0.05),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    fig.subplots_adjust(left=0.21, right=0.97, top=0.80, bottom=0.23, wspace=0.25)
    for ax, (a, b, title) in zip(
        axes,
        [("L1", "L2", "Teacher direction: SFT to DPO"), ("L2", "L3", "Teacher direction: DPO to RLVR")],
        strict=True,
    ):
        for y, (panel, _) in enumerate(panels):
            row = next(r for r in rows if r["a"] == a and r["b"] == b and r["panel"] == panel)
            assert row["identifiable"] == "True" and row["ci_low"] and row["ci_high"]
            value, lo, hi = (float(row[k]) for k in ["later_minus_earlier_projection", "ci_low", "ci_high"])
            color = fv.HUES["vblue" if panel.startswith("train") else "vteal"]
            ax.errorbar(value, y, xerr=[[value - lo], [hi - value]], fmt="o", color=color, capsize=3)
        ax.axvline(0, color=fv.NEUTRALS["baseline"], linewidth=0.8)
        ax.set(title=title, xlabel="Later-student minus earlier-student projection", xlim=limits)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(range(len(panels)), [label for _, label in panels])
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Fixed state panel")
    fig.suptitle(
        "Llama-3.2-3B-Instruct student | Tulu-3-8B teacher stages\n"
        "Equal 320-update endpoints; fixed original-student weighting",
        y=0.95,
    )
    fig.legend(
        [Line2D([], [], color=fv.HUES["vblue"], marker="o"), Line2D([], [], color=fv.HUES["vteal"], marker="o")],
        ["Training panels", "Held-out panels"],
        loc="lower center",
        bbox_to_anchor=(0.59, 0.10),
        ncol=2,
    )
    fig.text(
        0.5,
        0.025,
        "Signed projection coefficients, NOT absorption percentages. "
        "Positive means greater alignment with the later-stage direction.\n"
        "95% paired-prompt bootstrap intervals; 5000 draws; seed42; one training seed; frozen S0; H1024.",
        ha="center",
        fontsize=8.5,
        color=fv.NEUTRALS["baseline"],
    )
    fig.canvas.draw()
    issues = fv.audit(fig, print_slot=fig.get_figwidth(), scoped_ignore=[("missing-axis-label", 1)])
    dump(args.output / "audit.json", [str(i) for i in issues])
    if any(fv.problem_severity(i) == "error" for i in issues):
        raise RuntimeError(f"Figure audit failed: {issues}")
    for extension in ["png", "pdf"]:
        fig.savefig(args.output / f"paired-stage-preferences.{extension}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
