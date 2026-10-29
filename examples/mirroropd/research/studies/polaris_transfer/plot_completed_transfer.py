#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Ledger: teacher/run colors stay fixed. Curves are raw observed batches;
reuse curves pool the final 32 batches. Position profiles pool bucket sum/count.
All axes are explicit; no interpolation to unobserved checkpoints.
"""

import argparse
import json
from pathlib import Path

from examples.mirroropd.research.core.plotting import load_style


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--data", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    args = a.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fv = load_style()
    args.out.mkdir(parents=True, exist_ok=True)
    s = json.loads((args.data / "statistics.json").read_text())
    fv.apply_rc("paper", fontsize=10)
    colors = dict(zip(["E48", "E20"], fv.series_colors("paper", 2), strict=False))
    audits = {}

    def save(fig, name, caption):
        fig.text(0.5, 0.025, caption, ha="center", fontsize=8, color=fv.NEUTRALS["baseline"])
        fig.canvas.draw()
        audits[name] = [str(x) for x in fv.audit(fig)]
        for ext in ["png", "pdf"]:
            fig.savefig(args.out / f"{name}.{ext}", dpi=180)
        plt.close(fig)

    def panels():
        fig, axs = plt.subplots(2, 1, figsize=(6.3, 7.5))
        fig.subplots_adjust(left=0.16, right=0.96, bottom=0.16, top=0.96, hspace=0.43)
        return fig, axs

    def legend(fig, ax):
        fig.legend(
            *ax.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.06), ncol=2, frameon=False
        )

    fig, axs = panels()
    for name in ["E48", "E20"]:
        bs = s[name]["batches"]
        w = s[name]["windows"][-1]
        axs[0].plot([b["outer"] for b in bs], [b["first"] for b in bs], color=colors[name], label=name)
        k = np.array(w["reuse_kl"])
        fraction = (k[0] - k) / (k[0] - k[-1])
        axs[1].plot(range(1, 17), fraction, marker="o", markersize=3, color=colors[name], label=name)
    axs[0].set(xlabel="Outer step", ylabel="New-batch KL (nats/token)", yscale="log")
    axs[1].set(
        xlabel="Reuse observation (outer 225–256)", ylabel="Fraction of observed KL decrease", ylim=(-0.04, 1.04)
    )
    legend(fig, axs[0])
    save(fig, "plateau-and-reuse", "Observations 1–16 bracket 15 updates; fraction normalizes to their decrease.")
    fig, axs = panels()
    for ax, index, title in zip(axs, [0, -1], ["Outer 1–32", "Outer 225–256"], strict=False):
        w = s["E48"]["windows"][index]
        for side, ls in [("first", "-"), ("last", "--")]:
            ax.plot(
                np.arange(5, 100, 10),
                [v["mean"] for v in w["positions_" + side]],
                ls,
                color=colors["E48"],
                label=side + " observation",
            )
        ax.set(xlabel="Relative response position (%)", ylabel="E48 KL (nats/token)", ylim=(0.14, 0.22), title=title)
    legend(fig, axs[0])
    save(fig, "position-plateau", "Pooled bucket sums/counts; position is relative to each actual response.")
    fig, axs = panels()
    for name in ["E48", "E20"]:
        ws = s[name]["windows"]
        es = s[name]["evals"]
        axs[0].plot(
            [w["hi"] for w in ws],
            [100 * w["outer_metrics"]["rollout/finish_reason/length_count"] / 64 for w in ws],
            marker="o",
            color=colors[name],
            label=name,
        )
        axs[1].plot(
            [e["step/outer"] for e in es],
            [100 * e["eval/aime25/generation/finish_reason/length_count"] / 480 for e in es],
            marker="o",
            color=colors[name],
            label=name,
        )
    axs[0].set(xlabel="Outer step (end of 32-batch window)", ylabel="Training length stops (%)", ylim=(-2, 75))
    axs[1].set(xlabel="Evaluation outer step", ylabel="AIME25 length stops (%)", ylim=(-2, 75))
    legend(fig, axs[0])
    save(fig, "length-stops", "Training cap 7,168; validation cap 31,744. No E20 step-0 evaluation.")
    (args.out / "audit.json").write_text(json.dumps(audits, indent=2) + "\n")
    print(json.dumps(audits, indent=2))


if __name__ == "__main__":
    main()
