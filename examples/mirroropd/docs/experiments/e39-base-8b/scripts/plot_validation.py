#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Real E39 validation trends through outer step 140.
Semantic ledger: each series is one fixed validation dataset; each marker is
one complete n=16 evaluation of a checkpoint. Bands are the existing 95%
question-bootstrap intervals for mean accuracy, not variation across seeds.
Geometric ledger: three equal-width panels share outer-step coordinates;
y axes represent accuracy %, mean token length in thousands, and cap-hit %.
"""

import argparse
import csv
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
skill = Path("/mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/fht_data/dotfiles/.agents/skills/figure-viz")
sys.path.insert(0, str(skill / "assets/mpl"))
import fvstyle as fv  # noqa: E402, I001 - runtime source/style path is configured before import
import matplotlib  # noqa: E402, I001 - runtime source/style path is configured before import

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402, I001 - runtime source/style path is configured before import

fv.apply_rc("paper")
plt.rcParams.update(
    {
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 7,
        "lines.linewidth": 1.2,
        "lines.markersize": 2.5,
    }
)
rows = list(csv.DictReader(args.data.read_text().splitlines()))
fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.25))
fig.subplots_adjust(left=0.075, right=0.985, bottom=0.26, top=0.80, wspace=0.45)
colors = fv.series_colors("paper", 3)
for (slug, name), color in zip(
    [("math500", "MATH-500"), ("amc23", "AMC23"), ("aime25", "AIME25")], colors, strict=False
):
    rr = [r for r in rows if r["dataset"] == slug]
    x = [int(r["outer_step"]) for r in rr]
    for ax, field, scale in zip(axes, ["acc", "length_mean", "cap_fraction"], [100, 0.001, 100], strict=False):
        ax.plot(x, [float(r[field]) * scale for r in rr], color=color, marker="o", label=name)
    axes[0].fill_between(
        x,
        [float(r["acc_low"]) * 100 for r in rr],
        [float(r["acc_high"]) * 100 for r in rr],
        color=color,
        alpha=0.12,
        linewidth=0,
    )
for ax, label, ylim in zip(
    axes, ["Accuracy (%)", "Mean length (k tokens)", "Length-cap hits (%)"], [(-2, 100), (0, 16), (0, 50)], strict=False
):
    ax.set(xlabel="Outer step", ylabel=label, xlim=(-5, 150), ylim=ylim)
    ax.set_xticks([0, 50, 100, 150])
    ax.grid(axis="y", alpha=0.2)
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.52, 0.99), ncol=3, frameon=False)
fig.text(
    0.5,
    0.045,
    "Qwen3-1.7B-Base student; n=16 per question; bands: 95% question-bootstrap CI",
    ha="center",
    fontsize=6.2,
    color=fv.NEUTRALS["baseline"],
)
fig.canvas.draw()
problems = fv.audit(fig)
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.with_suffix(".audit.txt").write_text("\n".join(map(str, problems)) + "\n")
for ext in ("png", "pdf", "svg"):
    fig.savefig(args.out.with_suffix("." + ext), dpi=260)
print("audit", problems)
