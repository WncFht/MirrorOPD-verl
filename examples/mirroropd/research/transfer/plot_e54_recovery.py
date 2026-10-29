# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""E54 continuation and reuse controls, with original measured coordinates.

Semantic ledger: reuse16 is violet; reuse4 is orange. Each training point is a
64-prompt batch; x counts completed optimizer updates BEFORE its generation.
Evaluation bars are the same 64 held-out questions with one draw each, at the
saved checkpoint's update count. No smoothing or imputation. Geometry uses two
equal panels per figure; axes with different units retain independent scales.
"""

import argparse
import json
from pathlib import Path

from ..common.plotting import load_style


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", type=Path, required=True, help="Initial five-step run directory")
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fv = load_style()
    fv.apply_rc("paper", fontsize=8)
    colors = {16: fv.HUES["vviolet"], 4: fv.HUES["vorange"]}
    args.out.mkdir(parents=True, exist_ok=True)
    audits = {}

    def save(fig, stem, caption):
        fig.text(0.5, 0.018, caption, ha="center", fontsize=7, color=fv.NEUTRALS["baseline"])
        audits[stem] = fv.audit(fig)
        for suffix in ("png", "pdf"):
            fig.savefig(args.out / f"{stem}.{suffix}", dpi=240)
        plt.close(fig)

    baseline = read(args.initial / "debug/rollout-summary.jsonl") + read(
        args.controls / "continue-r16/debug/rollout-summary.jsonl"
    )
    branch = read(args.controls / "branch-r4/debug/rollout-summary.jsonl")
    assert [r["outer"] for r in baseline] == list(range(1, 16))
    assert [r["outer"] for r in branch] == list(range(3, 15))
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.8))
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.29, top=0.94, wspace=0.4)
    for reuse, rows in [(16, baseline), (4, branch)]:
        x = [(r["outer"] - 1) * 16 if reuse == 16 else 32 + (r["outer"] - 3) * 4 for r in rows]
        for ax, key, scale in [(axes[0], "mean_length", 0.001), (axes[1], "cap_count", 100 / 64)]:
            ax.plot(
                x,
                [r[key] * scale for r in rows],
                marker="o",
                markersize=2.5,
                color=colors[reuse],
                label=f"reuse{reuse}",
            )
    for ax, ylabel, limits in zip(
        axes, ["Mean length (k tokens)", "Length-capped (%)"], [(0, 7.5), (0, 100)], strict=True
    ):
        ax.set(xlabel="Updates before rollout", ylabel=ylabel, ylim=limits, xlim=(-5, 240), xticks=[0, 80, 160, 240])
        ax.grid(axis="y", alpha=0.4)
    fig.legend(
        *axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.06), ncol=2, frameon=False
    )
    save(fig, "training-recovery", "Different prompt batches at each point; training response cap 7,168.")

    labels = ["outer2", "outer5", "r16-outer10", "r16-outer15", "r4-outer14"]
    scores = [json.loads((args.controls / "evaluation" / label / "metrics.json").read_text()) for label in labels]
    assert all(row["responses"] == 64 for row in scores)
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.8))
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.29, top=0.94, wspace=0.4)
    for ax, key, scale, ylabel, upper in [
        (axes[0], "accuracy", 100, "Accuracy (%)", 100),
        (axes[1], "mean_length", 0.001, "Mean length (k tokens)", 34),
    ]:
        ax.bar(range(5), [r[key] * scale for r in scores], color=[colors[16]] * 4 + [colors[4]], width=0.65)
        ax.set(
            xticks=range(5),
            xticklabels=["32", "80", "160", "240", "80"],
            xlabel="Checkpoint updates",
            ylabel=ylabel,
            ylim=(0, upper),
        )
        ax.grid(axis="y", alpha=0.4)
        ax.set_axisbelow(True)
    fig.legend(
        handles=[Patch(color=color, label=f"reuse{reuse}") for reuse, color in colors.items()],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.06),
        ncol=2,
        frameon=False,
    )
    save(fig, "fixed-evaluation", "Same 64 MATH-500 questions; one sample each; response cap 31,744.")
    (args.out / "audit.json").write_text(json.dumps(audits, indent=2) + "\n")
    print(json.dumps(audits))


if __name__ == "__main__":
    main()
