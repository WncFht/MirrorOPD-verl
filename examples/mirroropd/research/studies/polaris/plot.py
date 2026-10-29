#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Ledger: colors identify optimizers; evaluation view is a panel/x category.
The ranges span two training prompt folds, not confidence intervals. Fixed
held-out student prefixes and teacher logits are common to all plotted arms.
"""


def main():
    import argparse
    import json
    from pathlib import Path

    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from examples.mirroropd.research.core.plotting import load_style

    fv = load_style()

    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows = json.loads((a.data / "summary.json").read_text())
    fv.apply_rc("paper", fontsize=10)
    colors = dict(zip(["adamw", "adam0", "sgd"], fv.series_colors("paper", 3), strict=False))
    labels = {"adamw": "AdamW", "adam0": "AdamW beta1=0", "sgd": "SGD"}
    audit = {}

    def select(opt, pre, view, calibrated=True):
        return [
            r
            for r in rows
            if r["initial"] == "initial"
            and r["optimizer"] == opt
            and r["training_precision"] == pre
            and r["view"] == view
            and r["step"] == 32
            and r["partition"] == "test"
            and r["source"] == "student"
            and r["domain"] == "math"
            and r["region"] == "all"
            and r["calibrated"] == calibrated
            and not r["accum_fp32"]
        ]

    def save(fig, name, caption):
        fig.text(0.5, 0.025, caption, ha="center", fontsize=8, color=fv.NEUTRALS["baseline"])
        fig.canvas.draw()
        # Errorbar cap artists belong to the same three optimizer series.
        audit[name] = [str(x) for x in fv.audit(fig, ignore=["palette-overflow"] if name == "precision-matrix" else [])]
        for ext in ["png", "pdf"]:
            fig.savefig(a.out / f"{name}.{ext}", dpi=180)
        plt.close(fig)

    fig, axs = plt.subplots(2, 1, figsize=(6.3, 7.6))
    fig.subplots_adjust(left=0.16, right=0.96, bottom=0.24, top=0.94, hspace=0.48)
    for ax, view in zip(axs, ["fp32", "bf16"], strict=False):
        for opt in colors:
            groups = [[100 * r["relative_drop"] for r in select(opt, pr, view)] for pr in ["bf16", "fp32"]]
            assert all(len(g) == 2 for g in groups)
            y = np.array([np.mean(g) for g in groups])
            lo = np.array([min(g) for g in groups])
            hi = np.array([max(g) for g in groups])
            ax.errorbar([0, 1], y, yerr=[y - lo, hi - y], color=colors[opt], marker="o", capsize=4, label=labels[opt])
        ax.set(
            xticks=[0, 1],
            xticklabels=["BF16 training", "FP32 training"],
            xlabel="Student training precision",
            ylabel="Held-out math KL decrease (%)",
            title=view.upper() + " evaluation",
            ylim=(0, 22),
            xlim=(-0.15, 1.15),
        )
    fig.legend(
        *axs[0].get_legend_handles_labels(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.06),
        ncol=1,
        frameon=False,
        fontsize=8,
    )
    save(fig, "precision-matrix", "32 updates; first calibration-step function drift matched; ranges: two folds.")

    fig, ax = plt.subplots(figsize=(6.3, 4.8))
    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.3, top=0.95)
    for prec, ls, marker in [("bf16", "-", "o"), ("fp32", "--", "s")]:
        y = [np.mean([r["rkl_after"] for r in select("adamw", prec, v)]) for v in ["fp32", "rounded_fp32", "bf16"]]
        ax.plot([0, 1, 2], y, ls, marker=marker, color=colors["adamw"], label="AdamW: train " + prec.upper())
    ax.set(
        xticks=[0, 1, 2],
        xticklabels=[
            "FP32 weights\nFP32 arithmetic",
            "BF16-rounded weights\nFP32 arithmetic",
            "BF16 weights\nBF16 arithmetic",
        ],
        ylabel="Held-out math teacher KL",
        xlabel="Realization of the SAME endpoint weights",
        xlim=(-0.25, 2.25),
    )
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.07),
        ncol=1,
        frameon=False,
        fontsize=8,
    )
    save(fig, "weight-rounding", "Only weight rounding / arithmetic changes; no additional optimization.")

    fig, ax = plt.subplots(figsize=(6.3, 5.3))
    fig.subplots_adjust(left=0.15, right=0.96, bottom=0.32, top=0.95)
    for opt in colors:
        for prec, marker in [("bf16", "o"), ("fp32", "^")]:
            rs = select(opt, prec, "fp32")
            ax.scatter(
                [r["drift"] for r in rs],
                [100 * r["relative_drop"] for r in rs],
                color=colors[opt],
                marker=marker,
                s=45,
                label=labels[opt] + " / " + prec.upper(),
            )
    rs = select("adamw", "bf16", "fp32", False)
    if rs:
        ax.scatter(
            [r["drift"] for r in rs],
            [100 * r["relative_drop"] for r in rs],
            color=colors["adamw"],
            marker="D",
            s=45,
            label="AdamW BF16 / LR 3e-6",
        )
    ax.set(xlabel="KL to initial student (FP32 evaluation)", ylabel="Held-out math teacher-KL decrease (%)")
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.07),
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    save(fig, "function-drift", "Each point is one training fold; a larger function change is not a free gain.")

    fig, ax = plt.subplots(figsize=(6.3, 5.1))
    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.29, top=0.95)
    settings = [
        ("adamw", "bf16", False, "-", "AdamW BF16 / matched LR"),
        ("adam0", "bf16", False, "-", "beta1=0 BF16 / matched LR"),
        ("adamw", "fp32", False, ":", "AdamW FP32 / matched LR"),
        ("adamw", "bf16", True, "--", "AdamW BF16 / LR 3e-6"),
    ]
    for opt, prec, fixed, style, label in settings:
        ys = []
        for fold in [0, 1]:
            arm = f"initial-{opt}-{prec}-fold{fold}" + ("-fixedlr" if fixed else "")
            history = json.loads((a.data / arm / "training.json").read_text())
            k = np.array([r["pre_update_kl"] for r in history[:16]])
            ys.append(k / k[0])
        ax.plot(range(1, 17), np.mean(ys, axis=0), linestyle=style, color=colors[opt], label=label)
    ax.set(xlabel="First-batch reuse observation", ylabel="Training KL / first observation", ylim=(0, 1.04))
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.065),
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    save(fig, "first-batch-delay", "Fresh optimizer: no previous-batch momentum. Curves average two folds.")
    (a.out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
