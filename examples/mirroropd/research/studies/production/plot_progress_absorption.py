# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Plot measured held-out risks from the weekly report's frozen JSON.

Semantic ledger: each line is one trained pair on one fixed bank/domain;
colors identify pairs throughout, and points are recorded outer boundaries.
Geometric ledger: x is optimizer updates, y is absolute question-mean RKL
or remaining fraction. Four panels share their bank/domain arrangement.
"""

import argparse
import json
from pathlib import Path

from examples.mirroropd.research.core.plotting import save_figure
from examples.mirroropd.research.studies.absorption_dynamics.geometry_plotting import grid, setup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.data.read_text())
    _, fv = setup()
    colors = fv.series_colors("paper", 4)
    args.output.mkdir(parents=True, exist_ok=True)
    for relative in (False, True):
        fig, axes = grid(2, 2, width=11, height=7)
        for row, domain in enumerate(("math", "all")):
            for col, bank in enumerate(("initial", "teacher")):
                ax = axes[row, col]
                for color, name in zip(colors, ("E94", "E95", "E96", "E97"), strict=True):
                    events = data["pairs"][name]
                    key = f"eval/rkl/public_{bank}/{domain}/question_mean"
                    y = [e["metrics"][key] for e in events]
                    if relative:
                        y = [v / y[0] for v in y]
                    ax.plot([e["optimizer_steps"] for e in events], y, color=color, label=name)
                ax.set_title(f"{bank} bank / {'Math' if domain == 'math' else 'Math+Code+IF mean'}")
                ax.set_xlabel("Optimizer updates")
                ax.set_ylabel("RKL / initial RKL" if relative else "Absolute held-out RKL")
                ax.set_yscale("log")
        lower = min(ax.get_ylim()[0] for ax in axes.flat)
        upper = max(ax.get_ylim()[1] for ax in axes.flat)
        for ax in axes.flat:
            ax.set_ylim(lower, upper)
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=4)
        save_figure(
            fig,
            args.output / ("r1-rkl-relative" if relative else "r1-rkl-absolute"),
            "Fixed banks; valid-token mean per question; three domains equally weighted; one seed per pair.",
        )


if __name__ == "__main__":
    main()
