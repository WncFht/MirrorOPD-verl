# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""EOS coverage on matched training questions, with explicit censoring.

Semantic ledger: violet is teacher generation from the question; orange is
the original student trajectory plus frozen continuation when it hit 7168.
Each panel contains the same 64 questions under the two policies. Unfinished
paths at 31744 never count as EOS events. X is total response tokens, excluding
question tokens; Y is the fraction naturally stopped by that budget. Equal
panels and shared Y scale; no smoothing or imputation of censored end times.
"""

import argparse
import json
from pathlib import Path

from examples.mirroropd.research.core.io import read_rows as read
from examples.mirroropd.research.core.plotting import load_style


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fv = load_style()
    fv.apply_rc("paper", fontsize=8)
    args.out.mkdir(parents=True, exist_ok=True)
    teacher = read(args.data / "teacher/generated.jsonl")
    observed = json.loads((args.data / "observed-student.json").read_text())
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(5.5, 2.9))
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.31, top=0.87, wspace=0.15)
    for ax, label, outer, updates in zip(axes, ["student5", "student10"], [6, 11], [80, 160], strict=True):
        cohort = f"observed_outer{outer}"
        originals = [r for r in observed if r["cohort"] == cohort]
        teachers = [r for r in teacher if r["cohort"] == cohort]
        continued = read(args.data / label / "generated.jsonl")
        assert len(originals) == len(teachers) == 64
        student_ends = [r["original_length"] for r in originals if r["original_stopped"]]
        student_ends += [r["response_length"] for r in continued if r["finish_reason"] == "stop"]
        teacher_ends = [r["response_length"] for r in teachers if r["finish_reason"] == "stop"]
        for name, ends, hue in [
            ("Teacher from question", teacher_ends, "vviolet"),
            ("Student continued", student_ends, "vorange"),
        ]:
            x = [0] + [v / 1000 for v in sorted(ends)] + [31.744]
            y = [0] + [100 * i / 64 for i in range(1, len(ends) + 1)] + [100 * len(ends) / 64]
            ax.step(x, y, where="post", color=fv.HUES[hue], label=name)
        for i, cap in enumerate([7.168, 16.384]):
            ax.axvline(
                cap,
                color=fv.NEUTRALS["baseline"],
                linestyle=":",
                linewidth=0.9,
                label="7.2k / 16.4k caps" if i == 0 else None,
            )
        ax.set(
            title=f"Student after {updates} updates",
            xlabel="Response budget (k tokens)",
            xlim=(0, 32),
            ylim=(0, 105),
            xticks=[0, 8, 16, 24, 32],
            yticks=[0, 25, 50, 75, 100],
        )
        ax.grid(axis="y", alpha=0.4)
    axes[0].set_ylabel("EOS observed (%)")
    fig.legend(
        *axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.07), ncol=3, frameon=False
    )
    fig.text(
        0.5,
        0.018,
        "Matched questions; student keeps its original prefix. Right-censored at 31,744.",
        ha="center",
        fontsize=7,
        color=fv.NEUTRALS["baseline"],
    )
    audit = fv.audit(fig)
    for suffix in ["png", "pdf"]:
        fig.savefig(args.out / f"eos-coverage.{suffix}", dpi=240)
    (args.out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
