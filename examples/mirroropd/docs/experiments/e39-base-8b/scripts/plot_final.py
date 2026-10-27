#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Render audited figures from frozen E39 analysis tables.

Semantic ledger: validation series identify datasets, training series identify
first/last pre-update observations, position series identify training windows.
Repetition panels use one fixed text sample per question, not all 16 samples.
Geometry: equal-width panels; outer-step axes have common 0..200 support.
Position curves use count-weighted relative-position bins, never raw sums.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
sys.path.insert(
    0, "/mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/fht_data/dotfiles/.agents/skills/figure-viz/assets/mpl"
)
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
        "lines.linewidth": 1.1,
        "lines.markersize": 2.3,
    }
)
args.out.mkdir(parents=True, exist_ok=True)
colors = fv.series_colors("paper", 3)


def csvrows(name):
    with (args.data / name).open() as f:
        return list(csv.DictReader(f))


def figure(n=3):
    fig, axes = plt.subplots(1, n, figsize=(5.5, 2.35))
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.27, top=0.80, wspace=0.45)
    return fig, axes


def outer_axis(ax, label, ylim=None):
    ax.set(xlabel="Outer step", ylabel=label, xlim=(-5, 205))
    ax.set_xticks([0, 100, 200])
    ax.grid(axis="y", alpha=0.2)
    if ylim is not None:
        ax.set_ylim(*ylim)


def finish(fig, axes, name, caption, legend=True):
    if legend:
        fig.legend(
            *axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.52, 0.99), ncol=3, frameon=False
        )
    fig.text(0.5, 0.04, caption, ha="center", fontsize=6.1, color=fv.NEUTRALS["baseline"])
    fig.canvas.draw()
    problems = fv.audit(fig)
    (args.out / (name + ".audit.txt")).write_text("\n".join(map(str, problems)) + "\n")
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(args.out / (name + "." + ext), dpi=260)
    print(name, problems)
    plt.close(fig)


vals = csvrows("validation.csv")
fig, axes = figure()
for (slug, name), color in zip(
    [("math500", "MATH-500"), ("amc23", "AMC23"), ("aime25", "AIME25")], colors, strict=False
):
    rows = [r for r in vals if r["slug"] == slug]
    x = [int(r["outer"]) for r in rows]
    for ax, field, scale in zip(axes, ["accuracy", "mean_length", "cap_fraction"], [100, 0.001, 100], strict=False):
        ax.plot(x, [float(r[field]) * scale for r in rows], color=color, marker="o", label=name)
    axes[0].fill_between(
        x,
        [float(r["accuracy_ci_low"]) * 100 for r in rows],
        [float(r["accuracy_ci_high"]) * 100 for r in rows],
        color=color,
        alpha=0.12,
        linewidth=0,
    )
for ax, label, ylim in zip(
    axes, ["Accuracy (%)", "Mean length (k tokens)", "Length-cap hits (%)"], [(-2, 100), (0, 16), (0, 50)], strict=False
):
    outer_axis(ax, label, ylim)
finish(fig, axes, "final-validation", "Full validation: 570 questions x 16; bands: 95% question-bootstrap CI")

train = csvrows("training.csv")
x = [int(r["outer"]) for r in train]
fig, axes = figure()
for end, color, label in [("first", colors[0], "First inner update"), ("last", colors[1], "Last inner update")]:
    for ax, key in zip(axes, ["kl", "identity", "grad"], strict=False):
        ax.plot(x, [float(r[f"{key}_{end}"]) for r in train], color=color, label=label)
for ax, label in zip(axes, ["Optimized KL (nats)", "EOS identity KL (nats)", "Gradient norm"], strict=False):
    outer_axis(ax, label)
axes[0].set_ylim(bottom=0)
axes[1].set_ylim(bottom=0)
axes[2].set_yscale("log")
finish(fig, axes, "final-optimization", "First/last are pre-update measurements within the same 16-update batch")

fig, axes = figure()
for ax, key, label, scale in zip(
    axes,
    ["mean_length", "cap_fraction", "train_score"],
    ["Train length (k tokens)", "Train length-cap hits (%)", "Training task score (%)"],
    [0.001, 100, 100],
    strict=False,
):
    ax.plot([int(r["outer"]) for r in train], [float(r[key]) * scale for r in train], color=colors[0])
    outer_axis(ax, label)
    ax.set_ylim(bottom=-0.5 if key == "cap_fraction" else 0)
finish(
    fig,
    axes,
    "final-training-behavior",
    "DAPO batches change each step; task score is diagnostic and not part of the loss",
    legend=False,
)

pos = csvrows("position-kl.csv")
fig, axes = figure(2)
for (lo, hi), color, label in [((1, 10), colors[0], "Steps 1-10"), ((181, 200), colors[1], "Steps 181-200")]:
    for ax, endpoint in zip(axes, ["first", "last"], strict=False):
        means = []
        for i in range(0, 100, 10):
            rows = [
                r
                for r in pos
                if lo <= int(r["outer"]) <= hi and r["endpoint"] == endpoint and r["bin"] == f"p{i:02d}_{i + 10:02d}"
            ]
            means.append(sum(float(r["sum"]) for r in rows) / sum(float(r["token_count"]) for r in rows))
        ax.plot(np.arange(5, 100, 10), means, color=color, marker="o", label=label)
for ax, title in zip(axes, ["First inner update", "Last inner update"], strict=False):
    ax.set(xlabel="Relative response position (%)", ylabel="Raw vocabulary KL (nats)", xlim=(0, 100), ylim=(0, 0.40))
    ax.set_xticks([0, 50, 100])
    ax.grid(axis="y", alpha=0.2)
    ax.set_title(title, fontsize=7.5)
finish(
    fig,
    axes,
    "final-position-kl",
    "Token-weighted bins; raw KL includes stop-token identity; lengths change across windows",
)

panel = csvrows("sample-panel-summary.csv")
fig, axes = figure()
for name, color in zip(["MATH-500", "AMC23", "AIME25"], colors, strict=False):
    rows = [r for r in panel if r["dataset"] == name]
    x = [int(r["step"]) for r in rows]
    for ax, key in zip(axes, ["loop_fraction", "repeat8", "short_fraction"], strict=False):
        ax.plot(x, [float(r[key]) * 100 for r in rows], color=color, marker="o", label=name)
for ax, label in zip(
    axes, ["Long-loop responses (%)", "Repeated 8-grams (%)", "Responses under 32 tokens (%)"], strict=False
):
    outer_axis(ax, label, (-2, 60))
finish(
    fig, axes, "final-repetition", "Fixed sample_index=0: 570 responses/checkpoint; repetition uses re-tokenized text"
)

resources = json.loads((args.data / "resources.json").read_text())
fig, axes = figure()
fig.subplots_adjust(left=0.115, wspace=0.5)
x = [r["outer"] for r in resources]
for ax, key, label in zip(
    axes,
    ["minutes", "mean_gpu_util_pct", "max_mem_gib"],
    ["Validation time (min)", "Mean GPU utilization (%)", "Sampled peak memory (GiB)"],
    strict=False,
):
    ax.plot(x, [r[key] for r in resources], color=colors[0], marker="o")
    outer_axis(ax, label)
axes[0].set_yscale("log")
axes[1].set_ylim(0, 100)
axes[2].set_ylim(0, 80)
finish(
    fig,
    axes,
    "final-resources",
    "Validation windows only; GPU samples every ~30 s; utilization includes tail waiting",
    legend=False,
)
