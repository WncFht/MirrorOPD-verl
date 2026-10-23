#!/usr/bin/env python3
"""System telemetry plot; all points are sampled W&B observations.

Semantic ledger: teal = physical GPUs 0-3, orange = GPUs 4-7; solid = pool
sample mean, bands = per-device range at that sample. Gray span = approximate
validation interval inferred from the recorded testing timer. No role mapping
or continuous-time utilization is inferred by the plotting routine.
Geometric ledger: two aligned time panels, common x in hours from first GPU
sample, percent and GiB on separate axes; legends outside data rectangles.
"""

# ruff: noqa: E402
import argparse
import json
from pathlib import Path

import fvlocate

fvlocate.ensure()
import fvstyle as fv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analyze_run import is_number, read_rows
from matplotlib.patches import Patch
from plot_run import save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    system_path = args.data / "system.jsonl.gz"
    if not system_path.is_file():
        system_path = args.data / "system-sampled.jsonl.gz"
    rows = [r for r in read_rows(system_path) if is_number(r.get("system.gpu.0.gpu"))]
    analysis = json.loads((args.data / "system-analysis.json").read_text())
    source = json.loads((args.data / "source.json").read_text())
    t0 = rows[0]["_timestamp"]
    x = [(r["_timestamp"] - t0) / 3600 for r in rows]
    fv.apply_rc("paper")
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "legend.fontsize": 9})
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.13, top=0.86, hspace=0.32)
    for ax, metric, unit, scale in [
        (axes[0], "gpu", "GPU busy (%)", 1),
        (axes[1], "memoryAllocatedBytes", "Whole-device memory (GiB)", 1 / 1024**3),
    ]:
        for indices, color, label in [
            (range(4), fv.HUES["vteal"], "Physical GPUs 0-3"),
            (range(4, 8), fv.HUES["vorange"], "Physical GPUs 4-7"),
        ]:
            values = [[r[f"system.gpu.{i}.{metric}"] * scale for i in indices] for r in rows]
            ax.fill_between(x, [min(v) for v in values], [max(v) for v in values], color=color, alpha=0.16)
            ax.plot(x, [sum(v) / len(v) for v in values], color=color, linewidth=0.8, label=label)
        for a, b in analysis["validation_timestamp_intervals"]:
            ax.axvspan((a - t0) / 3600, (b - t0) / 3600, color=fv.NEUTRALS["vmist"], alpha=0.35, zorder=-1)
        ax.set_ylabel(unit)
        ax.set_ylim(0, 105 if metric == "gpu" else 80)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=fv.NEUTRALS["vmist"], linewidth=0.5)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Patch(facecolor=fv.NEUTRALS["vmist"], alpha=0.35))
    labels.append("Validation (approx. interval)")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=3, frameon=False)
    axes[1].set_xlabel("Hours from first GPU sample")
    save(
        fig,
        args.out / "system-gpu",
        f"{source['run'].split('/')[-1]} | {len(rows):,} sampled GPU records; "
        "lines = group means, bands = device range",
    )


if __name__ == "__main__":
    main()
