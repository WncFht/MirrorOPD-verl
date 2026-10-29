#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Plot one policy-v3 snapshot; no comparison run or external baseline is read.

Semantic ledger: lines connect observations, quantile bands summarize tokens,
eval CI summarizes prompts, system bands summarize eight devices. None denotes
variation across training seeds. System history is sampled, not a GPU trace.
Geometry: independent units get separate panels, fractions become percent only
when labeled, cost bars start at zero. Every figure has PNG, PDF and audit output.
"""

# ruff: noqa: E402 -- Resolve the shared figure-viz style before importing it.
import argparse
import gzip
import json
from pathlib import Path

from examples.mirroropd.research.core.io import read_rows
from examples.mirroropd.research.core.logs import finite
from examples.mirroropd.research.core.plotting import load_style


def make_figure(rows, columns, title, width=10, height=None):
    fig, axes = plt.subplots(rows, columns, figsize=(width, height or (3 * rows + 0.8)), squeeze=False)
    fig.suptitle(title, y=0.98, fontsize=13, fontweight="bold")
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=fv.NEUTRALS["vmist"], linewidth=0.5)
        ax.set_axisbelow(True)
    return fig, axes.flatten()


def save(fig, output, name, caption):
    fig.tight_layout(rect=(0, 0.075, 1, 0.94), h_pad=2.8, w_pad=2.5)
    fig.text(0.5, 0.018, caption, ha="center", va="bottom", fontsize=8, color=fv.NEUTRALS["baseline"])
    fig.canvas.draw()
    problems = fv.audit(fig, print_slot=fig.get_figwidth())
    lines = fv.format_audit_lines(problems)
    (output / f"{name}.audit.txt").write_text("\n".join(lines) + "\n" if problems else "PASS: no audit issues.\n")
    if any(fv.problem_severity(p) == "error" for p in problems):
        raise RuntimeError(f"Figure audit failed: {name}: {lines}")
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{name}.{extension}", dpi=180, facecolor=fv.NEUTRALS["paper"])
    plt.close(fig)


def plot_metric(ax, series, key, label, color, scale=1):
    x, y = np.array(series[key]["points"]).T
    ax.plot(x, y * scale, label=label, color=color, linewidth=1.4)
    ax.set_xlabel(series[key]["axis"])


def legend(ax, columns=2):
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.015), ncol=columns, frameon=False, fontsize=8)


def gpu_observations(rows, suffix):
    import numpy as np

    """Skip incomplete samples; never substitute zero for a missing device."""
    result = []
    for row in rows:
        vals = [row.get(f"system.gpu.{device}.{suffix}") for device in range(8)]
        if finite(row.get("_timestamp")) and all(finite(v) for v in vals):
            result.append((row["_timestamp"], min(vals), float(np.median(vals)), max(vals)))
    return np.array(sorted(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    global matplotlib, np, plt
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    global fv
    fv = load_style()
    args.out.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.data / "series.json.gz", "rt") as stream:
        series = json.load(stream)
    analysis = json.loads((args.data / "analysis.json").read_text())
    source = json.loads((args.data / "source.json").read_text())
    run = source["run"].split("/")[-1]
    fv.apply_rc("paper")
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9})
    colors = fv.series_colors("paper", 3)
    neutral = fv.NEUTRALS["baseline"]

    fig, axes = make_figure(2, 2, f"E15 / {run}: optimization and rollout")
    for ax, keys, ylabel in zip(
        axes,
        [
            [("optim/actor/loss", "Loss"), ("optim/actor/grad_norm_preclip", "Gradient norm (preclip)")],
            [("rollout/score/mean", "Training task score (diagnostic)")],
            [("rollout/response_length/mean", "Mean response length")],
            [("policy/entropy/pre_update", "Pre-update entropy")],
        ],
        ["Recorded scalar", "Mean score", "Tokens", "Nats / token"],
        strict=True,
    ):
        for i, (key, label) in enumerate(keys):
            plot_metric(ax, series, key, label, colors[i])
        ax.set_ylabel(ylabel)
        legend(ax)
    axes[0].axhline(1, color=neutral, linestyle="--", linewidth=0.8)
    axes[1].set_ylim(0, 1)
    axes[2].axhline(7168, color=neutral, linestyle="--", linewidth=0.8)
    save(
        fig,
        args.out,
        "training",
        "200 observed updates; no smoothing. Task score is not included in the loss. "
        "Dashed: clip threshold / length cap.",
    )

    fig, axes = make_figure(2, 2, "E15: distillation signal and response truncation")
    for key, label, color in [
        ("std", "Advantage std", colors[0]),
        ("abs_mean", "Mean |advantage|", colors[1]),
    ]:
        plot_metric(axes[0], series, "advantage/distillation/" + key, label, color)
    axes[0].set_ylabel("Advantage magnitude")
    for key, label, color in [
        ("distill/advantage/sign_flip_fraction", "Current vs behavior signal flip", colors[0]),
        ("advantage/distillation/zero_fraction", "Exactly zero advantage", colors[1]),
    ]:
        plot_metric(axes[1], series, key, label, color, 100)
    axes[1].set_ylabel("Token fraction (%)")
    for q, color in [("p01", colors[0]), ("p50", neutral), ("p99", colors[1])]:
        plot_metric(axes[2], series, f"distill/student_teacher/sampled_log_gap_{q}", q, color)
    axes[2].set_ylabel("Sampled student - teacher log-prob")
    x, y = np.array(analysis["training"]["length_stopped_fraction_by_step"]).T
    axes[3].plot(x, y * 100, color=colors[0], label="finish_reason = length")
    axes[3].set(xlabel="step/outer", ylabel="Responses (%)", ylim=(0, 100))
    for ax in axes:
        legend(ax)
    save(
        fig,
        args.out,
        "signal",
        "Quantiles: 26 recorded batches, not confidence intervals. "
        "Flip compares two signals on the same sampled tokens.",
    )

    fig, axes = make_figure(2, 2, "E15: training recomputation versus rollout probabilities")
    for ax, suffix, label, ylabel in zip(
        axes[:3],
        ["k3", "ess_fraction", "log_ratio_abs_mean"],
        ["Anchor / behavior k3", "Token ESS fraction", "Mean |token log-ratio|"],
        ["k3 estimator", "ESS / token count", "Nats / token"],
        strict=True,
    ):
        plot_metric(ax, series, "policy/anchor_behavior/" + suffix, label, colors[0])
        ax.set_ylabel(ylabel)
    axes[1].set_ylim(0.65, 1.02)
    for step, value in analysis["training"]["low_ess_outer_steps"]:
        axes[1].annotate(str(int(step)), (step, value), xytext=(8, -3), textcoords="offset points", fontsize=9)
    for suffix, label, color in [
        ("min", "Batch min", colors[1]),
        ("mean", "Batch mean", colors[0]),
        ("max", "Batch max", neutral),
    ]:
        plot_metric(axes[3], series, "policy/anchor_behavior/sequence_log_ratio_" + suffix, label, color)
    axes[3].set_ylabel("Sum of token log-ratios / response")
    for ax in axes:
        legend(ax, 3)
    save(
        fig,
        args.out,
        "behavior",
        "Ratios and ESS are diagnostics; behavior importance sampling is disabled. "
        "Sequence sums depend on response length.",
    )

    datasets = [
        ("math500", "MATH-500 (500 prompts)"),
        ("amc23", "AMC23 (40 prompts)"),
        ("aime25", "AIME25 (30 prompts)"),
    ]
    fig, axes = make_figure(1, 3, "E15: validation mean accuracy", width=12, height=4.2)
    for ax, (dataset, label), color in zip(axes, datasets, colors, strict=True):
        rows = analysis["evaluation"][dataset]
        steps = [r["step"] for r in rows]
        ax.plot(steps, [r["acc/mean"] * 100 for r in rows], marker="o", color=color, label=label)
        ax.fill_between(
            steps,
            [r["acc/mean_ci_low"] * 100 for r in rows],
            [r["acc/mean_ci_high"] * 100 for r in rows],
            color=color,
            alpha=0.15,
        )
        ax.set(xlabel="step/optimizer", ylabel="Accuracy (%)", ylim=(0, 100))
        legend(ax, 1)
    save(
        fig,
        args.out,
        "evaluation",
        "16 responses per prompt. Shading: stored 95% prompt-bootstrap CI (256 resamples); "
        "not training-seed uncertainty.",
    )

    fig, axes = make_figure(1, 3, "E15: sampling budget at optimizer step 200", width=12, height=4.2)
    for ax, (dataset, label) in zip(axes, datasets, strict=True):
        for family, color in [("pass", colors[0]), ("majority", colors[1])]:
            y = [dict(series[f"eval/{dataset}/acc/{family}@{k}"]["points"])[200] * 100 for k in range(1, 17)]
            ax.plot(range(1, 17), y, color=color, marker=".", label=f"{family}@k")
        mean = analysis["evaluation"][dataset][-1]["acc/mean"] * 100
        ax.axhline(mean, color=neutral, linestyle="--", label="Mean accuracy")
        ax.set(xlabel=f"k: {label}", ylabel="Accuracy (%)", ylim=(0, 100), xticks=[1, 4, 8, 12, 16])
        legend(ax, 2)
    save(
        fig,
        args.out,
        "budget",
        "pass@k uses a correctness oracle. Majority: parsed answers, abstentions, averaged ties. "
        "Intermediate k can use MC subsets.",
    )

    fig, axes = make_figure(1, 2, "E15: training-stage cost and validation overhead", width=11, height=4.8)
    times = analysis["timing"]["regular_mean_seconds"]
    names = [
        "rollout_pipeline",
        "update_actor",
        "old_log_prob",
        "update_weights",
        "teacher_forward",
        "teacher_transfer",
        "adv",
    ]
    axes[0].barh(range(len(names)), [times[n] for n in names], color=colors[0])
    axes[0].set(
        yticks=range(len(names)), yticklabels=[n.replace("_", " ") for n in names], xlabel="Mean seconds / regular step"
    )
    axes[0].invert_yaxis()
    for i, name in enumerate(names):
        axes[0].text(times[name] + 0.25, i, f"{times[name]:.2f}", va="center", fontsize=9)
    axes[0].set_xlim(0, max(times.values()) * 0.75)
    outer = dict(series["perf/timing/outer_step_seconds"]["points"])
    validation = dict(series["perf/timing/validation_seconds"]["points"])
    steps = sorted(outer)
    outer_h = np.cumsum([outer[s] for s in steps]) / 3600
    validation_h = np.cumsum([validation.get(s, 0) for s in steps]) / 3600
    axes[1].plot(steps, outer_h, label="Training (incl. checkpoint)", color=colors[0])
    axes[1].plot(steps, validation_h, label="Validation", color=colors[1])
    axes[1].plot(steps, outer_h + validation_h, label="Accounted total", color=neutral)
    axes[1].set(xlabel="step/outer", ylabel="Cumulative hours", ylim=(0, None))
    legend(axes[1], 1)
    save(
        fig,
        args.out,
        "timing",
        "Regular = 189 steps excluding startup and evaluation. "
        "Nested timers are not added; no unmeasured stage is fabricated.",
    )

    fig, axes = make_figure(2, 2, "E15: validation length, truncation and cost")
    for (dataset, label), color in zip(datasets, colors, strict=True):
        rows = analysis["evaluation"][dataset]
        x = [r["step"] for r in rows]
        for ax, y in [
            (axes[0], [r["generation/response_length/mean"] for r in rows]),
            (axes[1], [100 * r["generation/finish_reason/length_count"] / r["response_count"] for r in rows]),
            (axes[2], [100 * r["acc/prediction_missing_count"] / r["response_count"] for r in rows]),
        ]:
            ax.plot(x, y, marker=".", color=color, label=label.split(" (")[0])
            ax.set_xlabel("step/optimizer")
    axes[0].set_ylabel("Mean response tokens")
    axes[1].set_ylabel("Length-stopped responses (%)")
    axes[2].set_ylabel("Missing parsed answers (%)")
    plot_metric(axes[3], series, "perf/timing/validation_seconds", "Full validation", colors[0], 1 / 60)
    plot_metric(axes[3], series, "perf/timing/validation_rollout_seconds", "Validation rollout", colors[1], 1 / 60)
    axes[3].set_ylabel("Minutes / evaluation")
    for ax in axes:
        ax.set_ylim(0, None)
        legend(ax, 3)
    save(
        fig,
        args.out,
        "evaluation_cost",
        "Validation cap = 31,744 tokens. Missing parsed answers and length stops can overlap; "
        "all recorded request failures = 0.",
    )

    fig, axes = make_figure(2, 2, "E15: process memory and sampled device telemetry")
    plot_metric(axes[0], series, "perf/actor/allocated_peak_gib", "Actor allocated high-water", colors[0])
    plot_metric(axes[0], series, "perf/actor/reserved_peak_gib", "Actor reserved high-water", colors[1])
    axes[0].set(ylabel="GiB (process allocator)", ylim=(0, None))
    plot_metric(axes[1], series, "perf/actor/mfu", "Actor MFU estimate", colors[0], 100)
    axes[1].set(ylabel="Model FLOP utilization (%)", ylim=(0, 100))
    system_path = args.data / "system.jsonl.gz"
    if not system_path.is_file():
        system_path = args.data / "system-sampled.jsonl.gz"
    system = read_rows(system_path)
    start = min(r["_timestamp"] for r in system)
    for ax, suffix, scale, label, ylabel in [
        (axes[2], "memoryAllocatedBytes", 1 / 2**30, "Device memory (8 GPUs)", "GiB (device-wide)"),
        (axes[3], "gpu", 1, "GPU busy (8 GPUs)", "Utilization (%)"),
    ]:
        observed = gpu_observations(system, suffix)
        x = (observed[:, 0] - start) / 3600
        ax.plot(x, observed[:, 2] * scale, color=colors[0], linewidth=1, label=label)
        ax.fill_between(x, observed[:, 1] * scale, observed[:, 3] * scale, color=colors[0], alpha=0.15)
        ax.set(xlabel="Hours since first telemetry sample", ylabel=ylabel, ylim=(0, None))
    axes[3].set_ylim(0, 105)
    for ax in axes:
        legend(ax, 1)
    save(
        fig,
        args.out,
        "resources",
        "Top: allocator high-water / model-based MFU. Bottom: sampled median and device min-max; "
        "not a complete timeline or SM activity.",
    )
    print(f"Wrote 8 figure groups to {args.out}")


if __name__ == "__main__":
    main()
