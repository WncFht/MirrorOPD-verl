#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Pipeline audit figures built only from recorded metrics.

Semantic ledger: teal = physical GPUs 0-3; orange = GPUs 4-7. Bar heights are
means of sparse interior probes, NOT time integrals. Replay-tail thresholds
overlap and must not be added. End-of-validation dots are individual probes;
white space is not interpolated into idle or busy periods.
Geometric ledger: paper-width panels, zero-based bars, separate units per
panel, external legends, captions and actual figure-viz audits.
"""

# ruff: noqa: E402
import argparse
import json
from pathlib import Path

from examples.mirroropd.research.core.io import read_rows
from examples.mirroropd.research.core.plotting import load_style
from examples.mirroropd.research.core.plotting import save_figure as save


def plain(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=fv.NEUTRALS["vmist"], linewidth=0.5)
    ax.set_axisbelow(True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--api-data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    global LinearSegmentedColormap, matplotlib, plt
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    global fv
    fv = load_style()
    args.out.mkdir(parents=True, exist_ok=True)
    j = json.loads((args.data / "pipeline-analysis.json").read_text())
    fv.apply_rc("paper")
    plt.rcParams.update(
        {
            "font.size": 7.6,
            "axes.labelsize": 7.6,
            "legend.fontsize": 6.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
        }
    )
    teal, orange = fv.HUES["vteal"], fv.HUES["vorange"]

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.8))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.8, bottom=0.25, wspace=0.4)
    phases = ["gen", "update_actor", "validation"]
    for ax, metric, ylabel, upper in [
        (axes[0], "busy_percent", "Sampled GPU busy (%)", 100),
        (axes[1], "memory_gib", "Device memory (GiB)", 80),
    ]:
        for offset, group, label, color in [
            (-0.18, "physical_0_3", "GPUs 0-3", teal),
            (0.18, "physical_4_7", "GPUs 4-7", orange),
        ]:
            values = [j["gpu_phase_probes"][p]["groups"][group][metric]["mean"] for p in phases]
            ax.bar([i + offset for i in range(3)], values, width=0.34, label=label, color=color)
        ax.set(
            xticks=range(3),
            xticklabels=["Gen\nN=298", "Update\nN=21", "Val\nN=1413"],
            ylabel=ylabel,
            xlabel="Inferred phase / probe rows",
            ylim=(0, upper),
        )
        plain(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    save(
        fig,
        args.out / "phase-gpu",
        "Inferred phase interiors, 2 s guard; sparse probe means, not GPU-time integrals",
        6.2,
    )

    val = j["completed_validation"][0]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.8))
    fig.subplots_adjust(left=0.085, right=0.98, top=0.91, bottom=0.25, wspace=0.4)
    rr = [r for w in val["windows"] for r in w]
    axes[0].scatter(
        [(r["timestamp"] - val["start"]) / 3600 for r in rr],
        [r["running"] for r in rr],
        s=5,
        color=teal,
        label="Replay status",
    )
    axes[0].set(xlabel="Validation elapsed (h)", ylabel="Unfinished prompt groups", ylim=(0, 17))
    axes[0].set_yticks([0, 4, 8, 12, 16])
    limits = [1, 2, 4, 7]
    spans = [val["nested_tail_spans"][str(k)]["consecutive_observed_span_seconds"] / 60 for k in limits]
    axes[1].bar([str(k) for k in limits], spans, color=teal)
    axes[1].set(xlabel="At most this many groups", ylabel="Observed tail span (min)", ylim=(0, 210))
    for ax in axes:
        plain(ax)
    save(
        fig,
        args.out / "validation-tail",
        "36 fixed windows; tail thresholds overlap; spans assume monotone completion",
        6.2,
    )

    rows = [
        r
        for r in read_rows(args.api_data / "system-sampled.jsonl.gz")
        if "system.gpu.0.gpu" in r and val["end"] - 180 <= r["_timestamp"] <= val["end"]
    ]
    x, y, values = [], [], []
    for r in rows:
        for gpu in range(8):
            x.append(r["_timestamp"] - val["end"])
            y.append(gpu)
            values.append(r[f"system.gpu.{gpu}.gpu"])
    fig, ax = plt.subplots(figsize=(5.5, 3))
    fig.subplots_adjust(left=0.1, right=0.95, top=0.94, bottom=0.23)
    cmap = LinearSegmentedColormap.from_list("busy", [fv.NEUTRALS["vmist"], teal])
    dots = ax.scatter(x, y, c=values, cmap=cmap, vmin=0, vmax=100, s=16)
    ax.set(xlabel="Seconds before validation history timestamp", ylabel="Physical GPU", xlim=(-185, 0))
    ax.set_yticks(range(8))
    ax.invert_yaxis()
    fig.colorbar(dots, ax=ax, pad=0.03, label="GPU busy (%)")
    save(
        fig,
        args.out / "validation-end",
        "Dots are observations; last seven samples show zero busy on all student GPUs",
        6.2,
    )


if __name__ == "__main__":
    main()
