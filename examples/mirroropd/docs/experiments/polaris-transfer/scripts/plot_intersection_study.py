#!/usr/bin/env python3
"""Ledger: each method has one color across risk panels; axes are real updates.
Fixed risk uses a frozen, source-balanced math panel. Fresh risk is sampled
without support constraints at H=1024. Contrast intervals are paired prompt
bootstrap, not training-seed uncertainty. No smoothed or synthetic curves.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.environ["FIGURE_VIZ_MPL"])
import fvstyle as fv


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--data", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    args = a.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fv.apply_rc("paper", fontsize=10)
    fixed = json.loads((args.data / "summary-fixed.json").read_text())
    fresh = json.loads((args.data / "summary-fresh.json").read_text())
    methods = [
        ("justrl-student-k16", "JustRL / student", fv.NEUTRALS["baseline"]),
        ("polaris-student-k16", "Polaris / student", fv.HUES["vteal"]),
        ("polaris-teacher-k16", "Polaris / teacher", fv.HUES["vviolet"]),
        ("polaris-intersection-k16", "Polaris / intersection", fv.HUES["vorange"]),
        ("polaris-topk-k16", "Polaris / student top-k", fv.HUES["vblue"]),
    ]
    audits = {}

    def save(fig, name, caption):
        fig.text(0.5, 0.022, caption, ha="center", fontsize=8, color=fv.NEUTRALS["baseline"])
        fig.canvas.draw()
        audits[name] = [str(x) for x in fv.audit(fig)]
        for ext in ["png", "pdf"]:
            fig.savefig(args.out / f"{name}.{ext}", dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(6.5, 8))
    fig.subplots_adjust(left=0.16, right=0.96, bottom=0.22, top=0.96, hspace=0.38)
    for arm, label, color in methods:
        r = sorted(
            [
                r
                for r in fixed
                if r["arm"] == arm and r["domain"] == "math" and r["source"] == "balanced" and r["band"] == "all"
            ],
            key=lambda r: r["step"],
        )
        axes[0].plot(
            [0] + [x["step"] for x in r],
            [1] + [x["after"] / x["before"] for x in r],
            marker="o",
            color=color,
            label=label,
        )
        r = sorted([r for r in fresh if r["arm"] == arm], key=lambda r: r["step"])
        axes[1].plot(
            [0] + [x["step"] for x in r],
            [1] + [x["sequence_kl_after"] / x["sequence_kl_before"] for x in r],
            marker="o",
            color=color,
            label=label,
        )
    for ax in axes:
        ax.set(xlabel="Optimizer updates", ylim=(0, 1.05), xticks=[0, 16, 32, 48, 64])
    axes[0].set_ylabel("Fixed-panel reverse KL / initial")
    axes[1].set_ylabel("Fresh sequence KL / initial")
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    save(fig, "fixed-and-fresh", "H=1024; unrestricted fresh evaluation; one training seed per arm.")

    rows = json.loads((args.data / "paired-method-contrasts.json").read_text())
    labels = [
        "Intersection − student (64)",
        "Intersection − top-k (64)",
        "Teacher − student (64)",
        "High − random states (16)",
        "Low − random states (16)",
    ]
    means = [r["mean_difference"] for r in rows]
    lo = [m - r["paired_prompt_ci"][0] for m, r in zip(means, rows)]
    hi = [r["paired_prompt_ci"][1] - m for m, r in zip(means, rows)]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.subplots_adjust(left=0.39, right=0.96, bottom=0.22, top=0.83)
    ax.errorbar(means, range(5), xerr=[lo, hi], fmt="o", color=fv.HUES["vteal"], capsize=0, label="Paired mean and CI")
    ax.axvline(0, color=fv.NEUTRALS["baseline"], lw=0.8, label="No difference")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.24), ncol=2, frameon=False, fontsize=8)
    ax.set(
        yticks=range(5),
        yticklabels=labels,
        xlabel="Sequence KL difference (nats)\nNegative favors the first method",
        ylabel="Paired comparison",
    )
    ax.invert_yaxis()
    save(fig, "method-contrasts", "95% paired prompt bootstrap; numbers in labels are training updates.")
    (args.out / "audit.json").write_text(json.dumps(audits, indent=2) + "\n")
    print(json.dumps(audits, indent=2))


if __name__ == "__main__":
    main()
