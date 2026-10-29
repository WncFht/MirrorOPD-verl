#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""B1 evidence ledger: all curves are fixed-input evaluations after real updates.

Each block trains 16 updates on A/B/C/D and the sequence repeats twice. Curve
colors identify evaluation batches, not teachers. Every curve is normalized by
its own initial RKL. Shading and letter labels identify the current training
block. Within/other-block bars telescope observed reductions along one AdamW
path; optimizer memory remains, so these are not independent causal effects.
Schedule comparisons keep exposures and per-token normalization identical.
"""

import argparse
import json
from pathlib import Path

from examples.mirroropd.research.core.plotting import load_style


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fv = load_style()
    a.out.mkdir(exist_ok=True, parents=True)
    read = lambda name: json.loads((a.data / name).read_text())
    rows = read("transfer-summary.json")
    influence = read("block-influence.json")
    fv.apply_rc("paper", fontsize=10)
    colors = [fv.HUES[k] for k in ["vteal", "vorange", "vblue", "vviolet"]]
    gray = fv.NEUTRALS["baseline"]
    steps = list(range(0, 129, 16))
    audits, tables, ledger = {}, {}, {}

    def get(teacher, schedule, group, step):
        return next(
            r for r in rows if r["arm"] == f"{teacher}-{schedule}" and r["group"] == group and r["step"] == step
        )

    def curve(teacher, schedule, group):
        initial = get(teacher, schedule, group, 16)["rkl_before"]
        return [initial] + [get(teacher, schedule, group, s)["rkl_after"] for s in steps[1:]]

    def save(fig, name, caption):
        fig.text(0.5, 0.02, caption, ha="center", fontsize=9, color=gray)
        fig.canvas.draw()
        audits[name] = [str(w) for w in fv.audit(fig)]
        for ext in ["png", "pdf"]:
            fig.savefig(a.out / f"{name}.{ext}", dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(9, 8.3))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.94, bottom=0.20, hspace=0.62, wspace=0.28)
    for col, teacher in enumerate(["justrl", "polaris"]):
        tables[teacher] = {
            g: curve(teacher, "blocked", g) for g in list("ABCDEFGH") + ["train_pool", "held_math", "Code", "IF"]
        }
        for row, groups in enumerate(["ABCD", "EFGH"]):
            ax = axes[row, col]
            for i in range(8):
                if i % 2 == 0:
                    ax.axvspan(16 * i, 16 * (i + 1), color=fv.NEUTRALS["vmist"], alpha=0.45, zorder=0)
                ax.text(16 * i + 8, 1.055, "ABCD"[i % 4], ha="center", va="center", fontsize=9, color=gray)
            for color, g in zip(colors, groups, strict=False):
                vals = tables[teacher][g]
                ax.plot(steps, np.array(vals) / vals[0], color=color, marker="o", markersize=3, label=g)
            ax.set(
                title=("JustRL" if teacher == "justrl" else "Polaris")
                + (" / fitted batches" if row == 0 else " / never-trained batches"),
                xlabel="Optimizer updates",
                ylabel="Fixed RKL / initial",
                ylim=(-0.025, 1.12),
                xticks=[0, 32, 64, 96, 128],
            )
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="center",
        bbox_to_anchor=(0.5, 0.53),
        ncol=4,
        frameon=False,
        title="Training-bank evaluation batches",
    )
    fig.legend(
        *axes[1, 0].get_legend_handles_labels(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.07),
        ncol=4,
        frameon=False,
        title="Independent held-out batches",
    )
    save(fig, "blocked-batch-curves", "B1: train A16 → B16 → C16 → D16 twice. Letters show the active training batch.")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.6))
    fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.35, wspace=0.38)
    for ax, teacher in zip(axes, ["justrl", "polaris"], strict=False):
        records = []
        for g in "ABCD":
            rs = [r for r in influence if r["arm"] == teacher + "-blocked" and r["evaluated_group"] == g]
            initial = tables[teacher][g][0]
            own = sum(r["absolute_drop"] for r in rs if r["trained_group"] == g)
            other = sum(r["absolute_drop"] for r in rs if r["trained_group"] != g)
            first_start = 16 * "ABCD".index(g)
            unseen = sum(r["absolute_drop"] for r in rs if r["trained_group"] != g and r["step_after"] <= first_start)
            seen = other - unseen
            assert abs(own + other - (initial - tables[teacher][g][-1])) < 1e-8
            records.append(
                dict(
                    group=g,
                    own_absolute_drop=own,
                    other_absolute_drop=other,
                    own_percent=100 * own / initial,
                    other_percent=100 * other / initial,
                    before_first_training_percent=100 * unseen / initial,
                    after_first_training_other_percent=100 * seen / initial,
                )
            )
        ledger[teacher] = records
        own = np.array([r["own_percent"] for r in records])
        other = np.array([r["other_percent"] for r in records])
        ax.bar(range(4), own, 0.65, color=colors[1], label="During this batch's blocks")
        unseen = np.array([r["before_first_training_percent"] for r in records])
        seen = np.array([r["after_first_training_other_percent"] for r in records])
        ax.bar(range(4), unseen, 0.65, bottom=own, color=colors[2], label="Other batches, before first training")
        bars = ax.bar(
            range(4), seen, 0.65, bottom=own + unseen, color=colors[0], label="Other batches, after first training"
        )
        ax.bar_label(bars, labels=[f"{v:.1f}" for v in own + other], padding=4, fontsize=9)
        ax.set(
            title="JustRL" if teacher == "justrl" else "Polaris",
            xticks=range(4),
            xticklabels=list("ABCD"),
            xlabel="Fixed evaluation batch",
            ylabel="Cumulative RKL decrease (%)",
            ylim=(0, 110),
        )
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.065),
        ncol=1,
        frameon=False,
        fontsize=9,
    )
    save(
        fig,
        "within-cross-block-gains",
        "Observed path decomposition; AdamW memory is retained. Not a causal attribution.",
    )

    fig, axes = plt.subplots(2, 2, figsize=(8.3, 7.5))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.94, bottom=0.18, hspace=0.45, wspace=0.36)
    schedule_colors = {"blocked": gray, "roundrobin": colors[0], "mixed": colors[1]}
    schedules = {}
    for col, teacher in enumerate(["justrl", "polaris"]):
        base = np.array(curve(teacher, "blocked", "held_math"))
        schedules[teacher] = {}
        for schedule, color in schedule_colors.items():
            risk = np.array(curve(teacher, schedule, "held_math"))
            schedules[teacher][schedule] = risk.tolist()
            label = {"blocked": "Blocked", "roundrobin": "Round-robin", "mixed": "Mixed"}[schedule]
            axes[0, col].plot(steps, risk, color=color, marker="o", markersize=3, label=label)
            axes[1, col].plot(steps, (risk / base - 1) * 100, color=color, marker="o", markersize=3, label=label)
        axes[0, col].set(
            title="JustRL" if teacher == "justrl" else "Polaris", ylabel="Held-out math RKL / horizon", ylim=(0, 0.27)
        )
        axes[1, col].set(ylabel="Difference from blocked (%)", ylim=(-6, 24))
        for row in range(2):
            axes[row, col].set(xlabel="Optimizer updates", xticks=[0, 32, 64, 96, 128])
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.065),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    save(
        fig,
        "matched-schedules",
        "B1: identical 64 trajectories, 128 updates, 32 visits per trajectory, fixed token weights.",
    )
    output = dict(
        steps=steps,
        curves=tables,
        path_decomposition=ledger,
        schedules=schedules,
        sources=["transfer-summary.json", "block-influence.json"],
        note="Series evaluated at 16-step boundaries; repeated blocks share optimizer history.",
    )
    (a.out / "plot-data.json").write_text(json.dumps(output, indent=2) + "\n")
    (a.out / "audit.json").write_text(json.dumps(audits, indent=2) + "\n")
    print(json.dumps(audits, indent=2))


if __name__ == "__main__":
    main()
