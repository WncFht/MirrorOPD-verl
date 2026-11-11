# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Generic, auditable stage plots for any compiled teacher-difference matrix.

Semantic ledger: color identifies teacher; line style identifies trained
branch. Axes show actual optimizer updates and the named scalar statistic.
Each panel uses one model, frozen bank and subset; no bank mixtures are pooled.
"""

import argparse
import itertools
import json
import os
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from examples.mirroropd.research.core.plotting import load_style, save_figure

from .results import load_results, summary_rows, write_csv
from .runtime import write_json

METRICS = [
    ("loss_mean", "Reverse KL / fixed H"),
    ("P", "Pressure P"),
    ("A", "Trajectory accessibility A"),
    ("C", "Finite-bank coherence"),
    ("cross_record_normalized", "Distinct-record coherence"),
    ("Xi", "Collective gain Xi"),
    ("PXi", "Gradient energy P Xi"),
    ("PXi_over_L", "Relative gradient rate P Xi / L"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--style", type=Path)
    parser.add_argument("--subsets", nargs="+", default=["all", "heldout/math"])
    args = parser.parse_args()
    out = args.out or args.manifest.parent / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    style = args.style or json.loads(args.manifest.read_text()).get("execution", {}).get("figure_style")
    if style:
        os.environ["FIGURE_VIZ_MPL"] = str(style)
    fv = load_style()
    fv.apply_rc("paper")
    results = list(load_results(args.manifest))
    rows = [row for task, data, _ in results for row in summary_rows(task, data)]
    if not rows:
        print(json.dumps(dict(completed=0, out=str(out))))
        return
    write_csv(out / "pac.csv", rows)
    targets = sorted({r["teacher"] for r in rows})
    colors = {name: fv.HUES[hue] for name, hue in zip(targets, itertools.cycle(fv.HUES), strict=False)}
    figures = []
    for model, bank, subset, teacher in sorted(
        {(r["model"], r["bank"], r["subset"], r["teacher"]) for r in rows if r["subset"] in args.subsets}
    ):
        selected = [
            r for r in rows if (r["model"], r["bank"], r["subset"], r["teacher"]) == (model, bank, subset, teacher)
        ]
        fig, axes = plt.subplots(2, 4, figsize=(11, 5.5))
        fig.subplots_adjust(bottom=0.29, top=0.92, hspace=0.65, wspace=0.65)
        baseline = [r for r in selected if r["stage"] == 0 and r["branch"] == "initial"]
        branch_pairs = sorted({(r["teacher"], r["branch"]) for r in selected if r["branch"] != "initial"})
        if not branch_pairs:
            branch_pairs = [(teacher, "initial")]
        appearances = itertools.cycle(itertools.product(["o", "s", "^", "D"], ["-", "--", ":", "-."]))
        for (target, branch), (marker, linestyle) in zip(branch_pairs, appearances, strict=False):
            values = sorted(
                [r for r in selected if r["teacher"] == target and r["branch"] == branch], key=lambda r: r["stage"]
            )
            if len(baseline) == 1 and values[0]["stage"] != 0:
                values = baseline + values
            for ax, (metric, label) in zip(axes.flat, METRICS, strict=True):
                ax.plot(
                    [r["stage"] for r in values],
                    [r.get(metric, float("nan")) for r in values],
                    marker=marker,
                    markersize=3,
                    linestyle=linestyle,
                    color=colors[target],
                    label=values[-1]["label"],
                )
                ax.set_xlabel("Optimizer updates")
                ax.set_ylabel(label)
                ax.spines[["top", "right"]].set_visible(False)
                if metric in ["loss_mean", "P", "A", "Xi", "PXi", "PXi_over_L"]:
                    ax.set_yscale("log")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        labels = [textwrap.fill(label, width=42, break_long_words=False, break_on_hyphens=False) for label in labels]
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.075), ncol=2, fontsize=7, frameon=False)
        path = out / f"{model}-{bank}-{teacher}-{subset.replace('/', '-')}-pac"
        save_figure(
            fig,
            path,
            f"{model}; target={teacher}; bank={bank}; subset={subset}.\n"
            "Exact native-RKL trajectory gradients; initial baseline included only when uniquely measured.",
        )
        figures.append(
            dict(
                model=model,
                bank=bank,
                teacher=teacher,
                subset=subset,
                files=[path.with_suffix(extension).name for extension in (".png", ".pdf", ".audit.txt")],
                conditions=len(selected),
            )
        )
    write_json(out / "figure-index.json", figures)
    write_json(
        out / "sources.json",
        dict(
            manifest=str(args.manifest.resolve()),
            completed=len(results),
            sources=[str(path.resolve()) for _, _, path in results],
        ),
    )
    print(json.dumps(dict(completed=len(results), out=str(out))))


if __name__ == "__main__":
    main()
