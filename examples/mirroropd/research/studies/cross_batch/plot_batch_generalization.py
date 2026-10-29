#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Ledger: curves are observed fixed-bank risks; colors identify evaluation groups.
Influence cells are 16-update changes normalized by each column's initial KL;
they are not independent one-step interventions from a common parameter point.
Coverage plots compare exact update budgets and report actual token exposure
separately. All observations are one training seed per recipe.
"""

import argparse
import json
from pathlib import Path

from examples.mirroropd.research.core.plotting import load_style


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    fv = load_style()
    a.out.mkdir(exist_ok=True, parents=True)
    stats = json.loads((a.data / "transfer-summary.json").read_text())
    influence = json.loads((a.data / "block-influence.json").read_text())
    prod = json.loads((a.data / "latest-run-summary.json").read_text())
    fv.apply_rc("paper", fontsize=10)
    audit = {}

    def save(fig, name, caption):
        fig.text(0.5, 0.022, caption, ha="center", fontsize=8, color=fv.NEUTRALS["baseline"])
        fig.canvas.draw()
        audit[name] = [str(x) for x in fv.audit(fig)]
        for ext in ["png", "pdf"]:
            fig.savefig(a.out / f"{name}.{ext}", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    fig.subplots_adjust(left=0.16, right=0.96, bottom=0.25, top=0.95)
    for name, color in [("E20", fv.HUES["vteal"]), ("E48", fv.HUES["vorange"])]:
        rows = prod[name]["batch_curve"]
        ax.plot([r["outer"] for r in rows], [r["first"] for r in rows], label=name, color=color)
    ax.set(xlabel="Outer batch", ylabel="First-observation KL (nats/token)", yscale="log")
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.07), ncol=2, frameon=False)
    save(fig, "e48-completed", "Completed runs: 256 outer batches and 4096 updates each.")

    series = [
        ("A", "Fitted batch A", fv.HUES["vteal"]),
        ("BCD", "Untrained batches B–D", fv.HUES["vorange"]),
        ("held_math", "Held-out math E–H", fv.HUES["vviolet"]),
        ("Code", "Held-out code", fv.HUES["vblue"]),
        ("IF", "Held-out IF", fv.NEUTRALS["baseline"]),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(6.6, 8.2))
    fig.subplots_adjust(left=0.14, right=0.96, bottom=0.22, top=0.94, hspace=0.43)
    for ax, teacher in zip(axes, ["justrl", "polaris"], strict=False):
        for group, label, color in series:
            ys = []
            for step in range(16, 129, 16):
                rows = [
                    r
                    for r in stats
                    if r["arm"] == teacher + "-single"
                    and r["step"] == step
                    and (r["group"] in ["B", "C", "D"] if group == "BCD" else r["group"] == group)
                ]
                ys.append(sum(r["rkl_after"] for r in rows) / sum(r["rkl_before"] for r in rows))
            ax.plot([0] + list(range(16, 129, 16)), [1] + ys, marker="o", color=color, label=label)
        ax.set(
            title="JustRL" if teacher == "justrl" else "Polaris",
            xlabel="Updates on batch A only",
            ylabel="Fixed-bank KL / initial",
            ylim=(-0.035, 1.05),
            xticks=[0, 32, 64, 96, 128],
        )
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    save(fig, "single-batch-transfer", "16 prompts in A; B–H, code and IF never receive training gradients.")

    columns = list("ABCDEFGH") + ["Code", "IF"]
    cmap = LinearSegmentedColormap.from_list("gain", [fv.HUES["vcoral"], fv.NEUTRALS["paper"], fv.HUES["vteal"]])
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 8.2))
    fig.subplots_adjust(left=0.18, right=0.86, bottom=0.14, top=0.95, hspace=0.37)
    for ax, teacher in zip(axes, ["justrl", "polaris"], strict=False):
        initial = {
            g: next(r["rkl_before"] for r in stats if r["arm"] == teacher + "-blocked" and r["group"] == g)
            for g in columns
        }
        matrix = np.array(
            [
                [
                    next(
                        r["absolute_drop"]
                        for r in influence
                        if r["arm"] == teacher + "-blocked" and r["step_after"] == step and r["evaluated_group"] == g
                    )
                    / initial[g]
                    * 100
                    for g in columns
                ]
                for step in range(16, 129, 16)
            ]
        )
        im = ax.imshow(matrix, cmap=cmap, norm=TwoSlopeNorm(vmin=-5, vcenter=0, vmax=50), aspect="auto")
        for i in range(8):
            for j in range(10):
                ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=8, color=fv.NEUTRALS["vink"])
        ax.set(
            title="JustRL" if teacher == "justrl" else "Polaris",
            xticks=range(10),
            xticklabels=columns,
            yticks=range(8),
            yticklabels=[f"{s - 15}–{s}: " + "ABCD"[(s // 16 - 1) % 4] for s in range(16, 129, 16)],
            xlabel="Fixed evaluation batch",
            ylabel="Updates: trained batch",
        )
    cax = fig.add_axes([0.89, 0.17, 0.025, 0.73])
    fig.colorbar(im, cax=cax, label="KL decrease (% of initial batch KL)")
    save(fig, "batch-influence", "Successive blocks; E–H/code/IF are always held out. Positive means improvement.")

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    fig.subplots_adjust(left=0.16, right=0.96, bottom=0.29, top=0.95)
    choices = [
        ("polaris-roundrobin", "64 prompts, LR 1e-6", fv.NEUTRALS["baseline"]),
        ("polaris-rr-lr3", "64 prompts, LR 3e-6", fv.HUES["vorange"]),
        ("polaris-rich-lr1", "512 prompts, LR 1e-6", fv.HUES["vteal"]),
        ("polaris-rich-lr3", "512 prompts, LR 3e-6", fv.HUES["vviolet"]),
    ]
    for arm, label, color in choices:
        rows = sorted([r for r in stats if r["arm"] == arm and r["group"] == "held_math"], key=lambda r: r["step"])
        ax.plot(
            [0] + [r["step"] for r in rows],
            [rows[0]["rkl_before"]] + [r["rkl_after"] for r in rows],
            marker="o",
            label=label,
            color=color,
        )
    ax.set(xlabel="Optimizer updates", ylabel="Held-out math KL (per fixed horizon token)", xticks=[0, 32, 64, 96, 128])
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    save(fig, "coverage-and-lr", "Same 128 updates and horizon-normalized budget; validation inputs stay fixed.")
    (a.out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
