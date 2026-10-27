#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Data figures: finite-pool noise and matched-stream batch trajectories."""

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(exist_ok=True, parents=True)
    fv.apply_rc("paper", fontsize=10)
    audit = {}
    noise = json.loads((a.data / "noise-summary.json").read_text())

    def save(fig, name, caption):
        fig.text(0.5, 0.018, caption, ha="center", fontsize=10, color=fv.NEUTRALS["baseline"])
        fig.canvas.draw()
        audit[name] = [str(x) for x in fv.audit(fig)]
        for ext in ["png", "pdf"]:
            fig.savefig(a.out / f"{name}.{ext}", dpi=180)
        plt.close(fig)

    colors = {"justrl": fv.HUES["vteal"], "polaris": fv.HUES["vorange"]}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.9))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.94, bottom=0.28, wspace=0.32)
    for r in noise:
        label = ("JustRL" if r["teacher"] == "justrl" else "Polaris") + (
            " initial" if r["phase"] == "initial" else " after 128 updates"
        )
        linestyle = "--" if r["phase"] == "initial" else "-"
        rows = r["by_batch"]
        axes[0].plot(
            [x["batch"] for x in rows[:-1]],
            [x["expected_noise_to_signal"] for x in rows[:-1]],
            marker="o",
            linestyle=linestyle,
            color=colors[r["teacher"]],
            label=label,
        )
        axes[1].plot(
            [x["batch"] for x in rows],
            [x["mean_cosine"] for x in rows],
            marker="o",
            linestyle=linestyle,
            color=colors[r["teacher"]],
            label=label,
        )
    axes[0].set(yscale="log", ylabel="Variance / squared pool-mean gradient")
    axes[1].set(ylabel="Mean cosine with held-out gradient", ylim=(0, 1.04))
    for ax, ticks in zip(axes, [[16, 64, 128], [16, 64, 128, 512]], strict=False):
        ax.set_xscale("log", base=2)
        ax.set_xticks(ticks, labels=[str(x) for x in ticks])
        ax.set_xlabel("Effective batch size")
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.07),
        ncol=2,
        frameon=False,
        fontsize=10,
    )
    save(fig, "gradient-noise", "Exact full-parameter Gram; B512 is the full fixed pool, not the population.")
    path = a.data / "trajectory-summary.json"
    if path.exists():
        stats = json.loads(path.read_text())
        palette = {16: fv.NEUTRALS["baseline"], 64: fv.HUES["vteal"], 128: fv.HUES["vorange"], 512: fv.HUES["vviolet"]}
        for axis in ["step", "examples"]:
            fig, axes = plt.subplots(2, 2, figsize=(9.4, 8))
            fig.subplots_adjust(left=0.1, right=0.97, top=0.94, bottom=0.19, wspace=0.3, hspace=0.38)
            for ax, (teacher, phase) in zip(
                axes.flat,
                [("justrl", "initial"), ("polaris", "initial"), ("justrl", "warm128"), ("polaris", "warm128")],
                strict=False,
            ):
                for b in [16, 64, 128, 512]:
                    rows = sorted(
                        [
                            r
                            for r in stats
                            if r["teacher"] == teacher
                            and r["phase"] == phase
                            and r["batch"] == b
                            and r["group"] == "held_math"
                        ],
                        key=lambda r: r[axis],
                    )
                    if not rows:
                        continue
                    ax.plot(
                        [0] + [r[axis] for r in rows],
                        [1] + [r["rkl"] / r["rkl_before"] for r in rows],
                        marker="o",
                        markersize=3,
                        color=palette[b],
                        label=f"Batch {b}",
                    )
                ax.set(
                    title=("JustRL" if teacher == "justrl" else "Polaris")
                    + (" / initial" if phase == "initial" else " / warm128"),
                    xlabel="Optimizer updates" if axis == "step" else "Processed trajectories",
                    ylabel="Held-out KL / starting KL",
                    ylim=(0, 1.04),
                )
                if axis == "examples":
                    ax.set_xlim(-100, 4300)
            handles, labels = axes[1, 1].get_legend_handles_labels()
            fig.legend(
                handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.065), ncol=4, frameon=False, fontsize=10
            )
            save(
                fig,
                f"batch-{axis}",
                "LR fixed at 1e-6. Same cyclic 512-trajectory stream; every arm starts with fresh AdamW.",
            )
    (a.out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(audit)


if __name__ == "__main__":
    main()
