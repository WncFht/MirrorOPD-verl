#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Plot frozen, matched-step pipeline comparisons.

Semantic ledger: orange=4+4 standalone; teal=8-GPU shared in all comparison
panels. Every point is an observed global step. Stage delta bars are shared
minus standalone; the teacher bar is newly serialized cost, not a claim that
standalone teacher computation was free. System bands are observed eight-GPU
ranges, not confidence intervals. Memory panels explicitly use different scopes.
Geometric ledger: equal bar widths/category spacing, zero-origin bars, true
step/time coordinates; line-only speed panels zoom their labeled y ranges.
No interpolation, smoothed surrogate, future validation or inferred speedup is
added to the data. Aggregation is performed by compare_pipeline_runs.py.
"""

# ruff: noqa: E402
import argparse
import json
from pathlib import Path

from examples.mirroropd.research.core.io import read_rows
from examples.mirroropd.research.core.plotting import load_style
from examples.mirroropd.research.core.plotting import save_figure as save
from examples.mirroropd.research.studies.pipeline.analyze_run import is_number

LABELS = {"standalone": "4+4 standalone", "shared": "8-GPU shared"}


def style(ax):
    """Keep axes and grids consistent across experimental panels."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=fv.NEUTRALS["vmist"], linewidth=0.5)
    ax.set_axisbelow(True)


def speed(result, out):
    """Show all paired steps, without claiming later unmatched baseline steps are comparable."""
    pairs = result["paired_steps"]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.8))
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.25, top=0.79, wspace=0.47)
    for label, prefix in [("standalone", "baseline"), ("shared", "shared")]:
        x = [r["step"] for r in pairs]
        y = [r[prefix + "_seconds"] for r in pairs]
        rate = [r[prefix + "_tokens"] / r[prefix + "_seconds"] for r in pairs]
        for ax, values in zip(axes, (y, rate), strict=True):
            ax.plot(x, values, marker="o", markersize=2.5, linewidth=1, color=COLORS[label], label=LABELS[label])
    for ax, title, ylabel in zip(
        axes,
        ("(a) Training step", "(b) Effective throughput"),
        ("Seconds / step", "Valid tokens / second"),
        strict=True,
    ):
        style(ax)
        ax.set(title=title, xlabel="Global step", ylabel=ylabel, xticks=[2, 4, 6, 8, 10, 12])
    axes[0].set_ylim(138, 154)
    axes[1].set_ylim(2450, 2950)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    save(
        fig,
        out / "matched-speed",
        "Matched steps 2–12; warm-up step 1 excluded.\nLine panels use zoomed axes; raw observations only.",
        caption_size=6.6,
    )


def stages(result, out):
    """Account for the net slowdown using disjoint timer components."""
    a, b = (result["runs"][k]["matched"] for k in ("standalone", "shared"))
    sa, sb = a["stage_mean_seconds"], b["stage_mean_seconds"]
    names = [
        "Gen stage",
        "Teacher: new serial stage",
        "Old log-prob",
        "Actor update",
        "Weight sync",
        "Adv + residual",
        "Net step delta",
    ]
    delta = [
        sb["gen"] - sa["gen"],
        sb["teacher_forward"] + sb["teacher_transfer"],
        sb["old_log_prob"] - sa["old_log_prob"],
        sb["update_actor"] - sa["update_actor"],
        sb["update_weights"] - sa["update_weights"],
        sb["adv"] - sa["adv"] + b["residual_seconds"]["mean"] - a["residual_seconds"]["mean"],
        b["metrics"]["timing_s/step"]["mean"] - a["metrics"]["timing_s/step"]["mean"],
    ]
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    fig.subplots_adjust(left=0.34, right=0.97, bottom=0.25, top=0.96)
    ax.barh(range(len(names)), delta, height=0.55, color=fv.HUES["vgray"])
    ax.set(yticks=range(len(names)), yticklabels=names, xlabel="Shared − standalone (seconds / step)", xlim=(-5, 5.5))
    ax.invert_yaxis()
    ax.axvline(0, color=fv.NEUTRALS["vink"], linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    for i, value in enumerate(delta):
        ax.text(
            value + (0.14 if value >= 0 else -0.14),
            i,
            f"{value:+.2f}",
            ha="left" if value >= 0 else "right",
            va="center",
            fontsize=7,
        )
    save(
        fig,
        out / "stage-delta",
        "Standalone teacher work is inside gen; it was not timed separately.\n"
        "The new serial teacher stage includes forward and transfer.",
        caption_size=6.6,
    )


def memory(result, out):
    """Separate process high-water, whole-device capacity and sampled busy."""
    labels = ("standalone", "shared")
    values = [
        [result["runs"][k]["matched"]["metrics"]["actor/perf/max_memory_allocated_gb"]["max"] for k in labels],
        [
            result["runs"][k]["matched_phases"]["gen"]["groups"]["all_eight"]["sum_device_memory_gib"]["mean"]
            for k in labels
        ],
        [result["runs"][k]["matched_system"]["groups"]["all_eight"]["busy_percent"]["mean"] for k in labels],
    ]
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.9))
    fig.subplots_adjust(left=0.095, right=0.99, bottom=0.28, top=0.78, wspace=0.65)
    for ax, yy, title, ylabel, ymax in zip(
        axes,
        values,
        ("(a) Process counter", "(b) Generation phase", "(c) Matched steps"),
        ("Peak allocated (GiB)", "8-GPU total (GiB)", "Mean GPU busy (%)"),
        (25, 400, 100),
        strict=True,
    ):
        for i, (label, value) in enumerate(zip(labels, yy, strict=True)):
            ax.bar(i, value, width=0.6, color=COLORS[label], label=LABELS[label])
            ax.text(i, value + ymax * 0.035, f"{value:.1f}", ha="center", va="bottom", fontsize=7)
        style(ax)
        ax.set(title=title, ylabel=ylabel, xlabel="Mode", xticks=[0, 1], xticklabels=["4+4", "Shared"], ylim=(0, ymax))
        ax.set_title(title, pad=12)
    fig.legend(
        [Patch(facecolor=COLORS[k]) for k in labels],
        [LABELS[k] for k in labels],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
    )
    save(
        fig,
        out / "memory-utilization",
        "Process counter: observed max; generation: sampled whole-device means.\n"
        "Busy: sampled eight-GPU means. Memory scopes differ; bars start at zero.",
        caption_size=6.6,
    )


def system(shared_dir, result, out):
    """Draw observed shared-mode GPU means and ranges in real timestamp coordinates."""
    rows = [r for r in read_rows(shared_dir / "system-sampled.jsonl.gz") if is_number(r.get("system.gpu.0.gpu"))]
    rows.sort(key=lambda r: r["_timestamp"])
    t0 = rows[0]["_timestamp"]
    x = [(r["_timestamp"] - t0) / 60 for r in rows]
    fig, axes = plt.subplots(2, 1, figsize=(5.5, 4.1), sharex=True)
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.19, top=0.89, hspace=0.35)
    for ax, metric, scale, label, limit in [
        (axes[0], "gpu", 1, "GPU busy (%)", 105),
        (axes[1], "memoryAllocatedBytes", 1 / 1024**3, "Device memory (GiB)", 50),
    ]:
        vv = [[r[f"system.gpu.{i}.{metric}"] * scale for i in range(8)] for r in rows]
        ax.fill_between(x, [min(v) for v in vv], [max(v) for v in vv], color=COLORS["shared"], alpha=0.18)
        ax.plot(x, [sum(v) / 8 for v in vv], color=COLORS["shared"], linewidth=0.9, label="Mean of eight GPUs")
        ax.set(ylabel=label, ylim=(0, limit))
        style(ax)
    axes[1].set_xlabel("Minutes since first GPU sample")
    fig.legend(
        [axes[0].lines[0], Patch(facecolor=COLORS["shared"], alpha=0.18)],
        ["Mean of eight GPUs", "Range across eight GPUs"],
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1),
        frameon=False,
    )
    save(
        fig,
        out / "shared-system",
        f"rcxtujbi | {len(rows)} sampled GPU records; line = eight-GPU mean.\n"
        "Sampling is about 15 s; short stage peaks can be missed.",
        caption_size=6.6,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    global COLORS, Patch, matplotlib, plt
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    global fv
    fv = load_style()
    COLORS = {"standalone": fv.HUES["vorange"], "shared": fv.HUES["vteal"]}
    args.out.mkdir(parents=True, exist_ok=True)
    result = json.loads((args.data / "comparison.json").read_text())
    fv.apply_rc("paper")
    speed(result, args.out)
    stages(result, args.out)
    memory(result, args.out)
    system(args.shared, result, args.out)


if __name__ == "__main__":
    main()
