# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Plot measured validation wall times without mixing profiler runs.

Ledger: each x category is one fixed model/input pair; two colors encode serial
and fixed 2+2 placement. Bar height is the median of three full-window seconds;
points are the actual repeats. Input token counts differ across model families,
so comparisons are within each pair. No synthetic or interpolated observations.
"""

import argparse
import json
import statistics
from pathlib import Path

from examples.mirroropd.research.core.plotting import load_style, save_figure

RUNS = [
    ("qwen4-n768-batched", "Qwen3-1.7B student\nQwen3-4B teacher"),
    ("qwen8-n768", "Qwen3-1.7B student\nQwen3-8B teacher"),
    ("r1_14-n768", "DeepSeek-R1-Distill-Qwen-7B student\nDeepSeek-R1-Distill-Qwen-14B teacher"),
]


def main():
    import matplotlib.pyplot as plt
    import numpy as np

    p = argparse.ArgumentParser()
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    fv = load_style()
    fv.apply_rc(font="sans", fontsize=10)
    colors = fv.palette("paper")[:2]
    summaries = {r["run"]: r for r in json.loads(args.summary.read_text())["runs"]}
    data = []
    for run, label in RUNS:
        record = summaries[run]
        assert record["complete"] and record["numerical_check_pass"] and record["rows"] == 768
        samples = {m: record["placements"][m]["samples_seconds"] for m in ["serial", "parallel"]}
        assert all(len(v) == 3 for v in samples.values())
        data.append((label, samples))
    fig, ax = plt.subplots(figsize=(10.8, 4.5))
    top = 0
    for j, mode in enumerate(["serial", "parallel"]):
        xs = np.arange(len(data)) + (j - 0.5) * 0.34
        values = [statistics.median(d[mode]) for _, d in data]
        ax.bar(xs, values, width=0.30, color=colors[j], label=["Serial, 4 GPUs", "Pipeline, 2 + 2 GPUs"][j])
        for x, y, (_, d) in zip(xs, values, data, strict=True):
            points = d[mode]
            samples_artist = ax.scatter(
                x + np.array([-0.035, 0, 0.035]), points, color=fv.NEUTRALS["vink"], s=12, zorder=3
            )
            fv.mark_neutral(samples_artist)
            ax.annotate(
                f"{y:.1f}", (x, max(points)), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=10
            )
            top = max(top, max(points))
    ax.set_xticks(range(len(data)), [label for label, _ in data])
    ax.set_ylabel("Full validation window (seconds)")
    ax.set_ylim(0, top * 1.22)
    ax.legend(loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=fv.NEUTRALS["vmist"], linewidth=0.6)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.95, bottom=0.25)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_figure(
        fig,
        args.out,
        "768 real math trajectories (thinking); token workloads differ. Bars: median; dots: 3 repeats.\n"
        "GPU loading through four-GPU student restore; CPU layouts prebuilt; no optimizer.",
        caption_size=8,
    )


if __name__ == "__main__":
    main()
