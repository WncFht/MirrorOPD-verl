#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Plot frozen E48 observations and explicit controls.

Semantic ledger: color identifies run throughout; solid/dashed identify first
and last pre-update observations. Position axes are relative response deciles.
Geometry ledger: linear outer/update axes; logarithmic KL only when comparing
orders of magnitude. Panels with different units keep independent y scales.
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
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fv = load_style()
    args.out.mkdir(parents=True, exist_ok=True)
    s = json.loads((args.data / "statistics.json").read_text())
    fv.apply_rc("paper", fontsize=10)
    colors = dict(zip(["E48", "E20", "E17", "E35"], fv.series_colors("paper", 4), strict=False))
    audits = {}

    def save(fig, name, caption):
        fig.text(0.5, 0.025, caption, ha="center", fontsize=8, color=fv.NEUTRALS["baseline"])
        fig.canvas.draw()
        problems = fv.audit(fig)
        audits[name] = [str(x) for x in problems]
        for ext in ["png", "pdf"]:
            fig.savefig(args.out / f"{name}.{ext}", dpi=180)
        plt.close(fig)

    fig, axs = plt.subplots(2, 1, figsize=(6.3, 7.4))
    fig.subplots_adjust(left=0.16, right=0.96, bottom=0.16, top=0.96, hspace=0.42)
    for label in ["E48", "E20", "E17", "E35"]:
        bs = [b for b in s[label]["batches"] if b["outer"] <= 58]
        axs[0].plot([b["outer"] for b in bs], [b["first"] for b in bs], label=label, color=colors[label])
        axs[1].plot([b["outer"] for b in bs], [100 * b["relative_drop"] for b in bs], label=label, color=colors[label])
    axs[0].set(yscale="log", xlabel="Outer step", ylabel="KL at first reuse observation (nats/token)")
    axs[1].set(xlabel="Outer step", ylabel="Within-batch KL decrease (%)")
    fig.legend(
        *axs[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.055), ncol=4, frameon=False
    )
    save(fig, "convergence", "15 updates between observations; E35 uses a longer cap.")

    fig, axs = plt.subplots(2, 1, figsize=(6.3, 7.4))
    fig.subplots_adjust(left=0.16, right=0.96, bottom=0.16, top=0.94, hspace=0.45)
    for ax, window in zip(axs, ["1-8", "49-58"], strict=False):
        w = s["E48"]["windows"][window]
        for side, ls in [("first", "-"), ("last", "--")]:
            ax.plot(
                np.arange(5, 100, 10),
                [w[f"position_{side}_{p:02d}"] for p in range(0, 100, 10)],
                ls,
                label=side + " reuse observation",
                color=colors["E48"],
            )
        ax.set(xlabel="Relative response position (%)", ylim=(0.14, 0.23), title="Outer " + window)
    for ax in axs:
        ax.set_ylabel("E48 conditional KL (nats/token)")
    fig.legend(
        *axs[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.055), ncol=2, frameon=False
    )
    save(fig, "positions", "Pooled sums/counts; deciles use actual response lengths.")

    fig, axs = plt.subplots(3, 1, figsize=(6.3, 9.8))
    fig.subplots_adjust(left=0.16, right=0.96, bottom=0.13, top=0.96, hspace=0.55)
    for ax, ds in zip(axs, ["math500", "amc23", "aime25"], strict=False):
        for label in ["E48", "E20"]:
            rs = [r for r in s[label]["evals"] if r["step/optimizer"] <= 1024]
            ax.plot(
                [r["step/optimizer"] for r in rs],
                [
                    100 * r[f"eval/{ds}/generation/finish_reason/length_count"] / r[f"eval/{ds}/response_count"]
                    for r in rs
                ],
                marker="o",
                label=label,
                color=colors[label],
            )
        ax.set(xlabel="Optimizer updates", title=ds.upper(), ylim=(-1, 32))
    for ax in axs:
        ax.set_ylabel("Length-limit fraction (%)")
    fig.legend(
        *axs[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.055), ncol=2, frameon=False
    )
    save(fig, "validation-length", "Validation cap: 31,744. E20 has no step-0 evaluation.")
    (args.out / "audit.json").write_text(json.dumps(audits, indent=2))
    print(json.dumps(audits, indent=2))


if __name__ == "__main__":
    main()
