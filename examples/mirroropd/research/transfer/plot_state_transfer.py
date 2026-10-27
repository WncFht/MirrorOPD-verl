#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Semantic ledger: color is target teacher; line style is rollout source.
Geometry: fixed categorical source slots, linear KL; equal-width domain slots
for relative held-out changes. Error ranges are two training prompt folds,
not confidence intervals. All numeric inputs come from frozen summaries.
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..common.plotting import load_style


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--data", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    args = a.parse_args()
    fv = load_style()
    args.out.mkdir(parents=True, exist_ok=True)
    initial = json.loads((args.data / "initial-summary.json").read_text())
    transfer = json.loads((args.data / "transfer-summary.json").read_text())
    fv.apply_rc("paper", fontsize=10)
    cs = fv.series_colors("paper", 2)
    colors = dict(zip(["justrl", "polaris"], cs, strict=False))
    audit = {}

    def save(fig, name, caption):
        fig.text(0.5, 0.025, caption, ha="center", fontsize=8, color=fv.NEUTRALS["baseline"])
        fig.canvas.draw()
        # Caps and vertical ranges belong to four semantic errorbar series;
        # the audit's artist count otherwise treats them as sixteen series.
        ignored = ["palette-overflow"] if name == "heldout-transfer" else []
        audit[name] = [str(x) for x in fv.audit(fig, ignore=ignored)]
        for ext in ["png", "pdf"]:
            fig.savefig(args.out / f"{name}.{ext}", dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(6.3, 7.4))
    fig.subplots_adjust(left=0.15, right=0.96, bottom=0.16, top=0.94, hspace=0.47)
    sources = ["student", "justrl", "polaris"]
    for ax, region in zip(axes, ["early", "late"], strict=False):
        for t in colors:
            vals = [
                next(
                    r["kl"]
                    for r in initial
                    if r["domain"] == "math" and r["teacher"] == t and r["source"] == s and r["region"] == region
                )
                for s in sources
            ]
            ax.plot(
                range(3),
                vals,
                marker="o",
                label={"justrl": "JustRL", "polaris": "Polaris"}[t] + " target",
                color=colors[t],
            )
        ax.set(
            xticks=range(3),
            xticklabels=["Initial student", "JustRL", "Polaris"],
            xlabel="Rollout source",
            ylabel="Initial KL (nats/state)",
            ylim=(0, 0.6),
            title="Positions " + ("0–248" if region == "early" else "512–752"),
        )
    fig.legend(
        *axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.055), ncol=2, frameon=False
    )
    save(fig, "source-mismatch", "Same 16 math prompts; 32 early and 16 later states per trace.")
    fig, axes = plt.subplots(2, 1, figsize=(6.3, 7.6))
    fig.subplots_adjust(left=0.15, right=0.96, bottom=0.19, top=0.94, hspace=0.48)
    domains = ["math", "code", "if"]
    for ax, region in zip(axes, ["early", "late"], strict=False):
        for t in colors:
            for train_source, style in [("student", "-"), (t, "--")]:
                means = []
                low = []
                high = []
                for domain in domains:
                    rs = [
                        r
                        for r in transfer
                        if r["teacher"] == t
                        and r["training_source"] == train_source
                        and r["temperature"] == 1
                        and r["step"] == 16
                        and r["partition"] == "heldout"
                        and r["source"] == "student"
                        and r["domain"] == domain
                        and r["region"] == region
                    ]
                    ys = [100 * r["relative_drop"] for r in rs]
                    assert len(ys) == 2, (t, train_source, domain, region, len(ys))
                    means.append(np.mean(ys))
                    low.append(min(ys))
                    high.append(max(ys))
                ax.errorbar(
                    range(3),
                    means,
                    yerr=[np.array(means) - low, np.array(high) - means],
                    linestyle=style,
                    marker="o",
                    capsize=3,
                    color=colors[t],
                    label={"justrl": "JustRL", "polaris": "Polaris"}[t]
                    + (" / student states" if train_source == "student" else " / teacher states"),
                )
        ax.axhline(0, color=fv.NEUTRALS["baseline"], lw=0.8, zorder=0)
        ax.set(
            xticks=range(3),
            xticklabels=["Math (8)", "Code (4)", "IF (4)"],
            xlabel="Held-out domain (prompts)",
            ylabel="KL decrease on student states (%)",
            title=region.capitalize() + " states",
        )
    # Equal scales expose comparisons without rescaling the two state regions.
    lo = min(ax.get_ylim()[0] for ax in axes)
    hi = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(lo, hi)
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.06),
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    save(fig, "heldout-transfer", "16 full-parameter updates; ranges span two training prompt folds.")
    (args.out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
