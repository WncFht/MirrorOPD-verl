#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Plot measured checkpoints against explicitly historical mean-accuracy references."""

# ruff: noqa: E402 -- Resolve the shared plotting style before importing it.
import argparse
import json
from pathlib import Path

import fvlocate

fvlocate.ensure()
import fvstyle as fv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    runs = json.loads((args.data / "analysis.json").read_text())["runs"]
    refs = json.loads(args.references.read_text())["historical"]
    fv.apply_rc("paper")
    palette = fv.series_colors("paper", 4)
    colors = dict(zip(runs, [palette[0], palette[1], fv.NEUTRALS["baseline"], *palette[2:]], strict=True))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
    for ax, slug, title in zip(axes, ("math500", "amc23", "aime25"), ("MATH-500", "AMC23", "AIME25"), strict=True):
        for label, run in runs.items():
            rows = run["evaluation"]
            ax.plot(
                [r["outer"] for r in rows],
                [100 * r["metrics"][f"eval/{slug}/acc/mean"] for r in rows],
                color=colors[label],
                marker="o",
                markersize=3,
                lw=1.5,
            )
        for ref in refs:
            teacher = ref["kind"] == "teacher"
            ax.axhline(
                100 * ref["mean_accuracy"][slug],
                color=fv.NEUTRALS["vink"] if teacher else fv.NEUTRALS["baseline"],
                ls="--" if teacher else ":",
                lw=1.4,
            )
        ax.set(title=title, xlabel="Outer step", ylabel="Mean accuracy (%)", xlim=(15, 205))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=fv.NEUTRALS["vmist"], lw=0.6)
        ax.set_axisbelow(True)
    handles = [Line2D([], [], color=colors[k], label=k, lw=2) for k in runs]
    handles += [
        Line2D([], [], color=fv.NEUTRALS["vink"], ls="--", label="Historical teacher"),
        Line2D([], [], color=fv.NEUTRALS["baseline"], ls=":", label="Historical student"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=7, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0.12, 1, 0.91), w_pad=2)
    fig.text(
        0.5,
        0.025,
        "References: One-Shot-OPD val-only, matching sampling budget, different pipeline; "
        "not measured step zero. E17 snapshot ends at eval 160.",
        ha="center",
        fontsize=9,
    )
    fig.canvas.draw()
    issues = fv.audit(fig, print_slot=fig.get_figwidth())
    lines = fv.format_audit_lines(issues)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "accuracy-reference.audit.txt").write_text("\n".join(lines) + "\n" if issues else "PASS\n")
    if any(fv.problem_severity(i) == "error" for i in issues):
        raise ValueError(lines)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(args.out / f"accuracy-reference.{ext}", dpi=150, facecolor=fv.NEUTRALS["paper"])
    svg = args.out / "accuracy-reference.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    main()
