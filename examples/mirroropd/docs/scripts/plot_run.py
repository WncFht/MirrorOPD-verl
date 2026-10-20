#!/usr/bin/env python3
"""Plot an auditable training snapshot with figure-viz's Matplotlib style.

Semantic ledger: every point is an observed console/API metric; connecting lines
only connect observations. Timing bars are arithmetic means after global step 1.
No validation/GPU-busy curves are fabricated when those streams are unavailable.
Geometric ledger: regular panels, all memory in GiB, utilization as percent;
zero-based bars, legends above data, captions below the panel bounding boxes.
"""

# ruff: noqa: E402 -- figure-viz is resolved before importing its style module.
import argparse
import json
from pathlib import Path

import fvlocate

fvlocate.ensure()

import fvstyle as fv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analyze_run import read_rows, write_json


def series(ax, rows, key, label, color, scale=1, linestyle="-"):
    points = [(r["training/global_step"], r[key] * scale) for r in rows if key in r]
    if points:
        x, y = zip(*points, strict=True)
        ax.plot(x, y, marker="o", markersize=3.5, color=color, label=label, linestyle=linestyle)
    return bool(points)


def axes_style(ax, ylabel, ylim=None):
    ax.set(xlabel="Global step", ylabel=ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", color=fv.NEUTRALS["vmist"], linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def legend(ax):
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, frameon=False, borderaxespad=0)


def save(fig, path, caption, caption_size=8):
    fig.text(0.5, 0.025, caption, ha="center", fontsize=caption_size, color=fv.NEUTRALS["baseline"])
    fig.canvas.draw()
    problems = fv.audit(fig, print_slot=fig.get_figwidth())
    for p in fv.format_audit_lines(problems):
        print(path.name + ": " + p)
    path.with_suffix(".audit.txt").write_text(
        "\n".join(fv.format_audit_lines(problems)) + "\n" if problems else "PASS: fv.audit(fig) returned no issues.\n"
    )
    if any(fv.problem_severity(p) == "error" for p in problems):
        raise RuntimeError("Figure audit failed")
    for extension in ("png", "pdf"):
        fig.savefig(path.with_suffix("." + extension), dpi=180, facecolor=fv.NEUTRALS["paper"])
    plt.close(fig)


def validation_with_baselines(row, source, student, teacher, output, data_directory):
    """Ledger: one color/model in all panels; all bars are observed metrics.

    External val-only references are shown beside one trained checkpoint;
    neither a same-run step-zero observation nor error bars are invented.
    """
    datasets = ["MATH-500", "AMC23", "AIME25"]
    metrics = [
        ("mean@16", "Mean accuracy@16"),
        ("best@16/mean", "Bootstrap best@16"),
        ("maj@16/mean", "Bootstrap majority@16"),
    ]
    models = [
        ("Student reference", student, fv.NEUTRALS["baseline"]),
        (f"OPD step {row['training/global_step']:g}", None, fv.HUES["vteal"]),
        ("Teacher reference", teacher, fv.HUES["vorange"]),
    ]
    records = []
    plt.rcParams.update(
        {
            "font.size": 7.6,
            "axes.labelsize": 7.6,
            "legend.fontsize": 6.6,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.6), sharey=True)
    fig.subplots_adjust(left=0.105, right=0.99, top=0.83, bottom=0.21, wspace=0.16)
    for ax, (metric, label) in zip(axes, metrics, strict=True):
        for offset, (model, directory, color) in zip((-0.25, 0, 0.25), models, strict=True):
            values = row if directory is None else json.loads((directory / "summary.json").read_text())
            origin = source if directory is None else json.loads((directory / "source.json").read_text())
            scores = []
            for dataset in datasets:
                key = f"val-core/{dataset}/acc/{metric}"
                score = values[key]
                if not isinstance(score, (int, float)) or not 0 <= score <= 1:
                    raise ValueError(f"Invalid accuracy: {origin['run']} {key}")
                scores.append(score * 100)
                records.append(
                    {
                        "model": model,
                        "run": origin["run"],
                        "dataset": dataset,
                        "metric": metric,
                        "key": key,
                        "value": score,
                    }
                )
            ax.bar([i + offset for i in range(3)], scores, width=0.24, label=model, color=color)
        ax.set(xticks=range(3), xticklabels=datasets, xlabel=label, ylim=(0, 100))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=fv.NEUTRALS["vmist"], linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=3, frameon=False)
    write_json(
        data_directory / "baseline-comparison.json",
        {
            "note": "External One-Shot-OPD references; matched sampling budgets, different execution pipelines. "
            "Not a controlled same-run step-zero comparison. All values are fractions.",
            "records": records,
        },
    )
    save(
        fig,
        output / "validation",
        "External student / teacher references; OPD step 20 (different pipelines)",
        caption_size=6.5,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--student-baseline", type=Path)
    p.add_argument("--teacher-baseline", type=Path)
    args = p.parse_args()
    if bool(args.student_baseline) != bool(args.teacher_baseline):
        p.error("Provide both student and teacher baseline snapshot directories")
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in read_rows(args.data / "history.jsonl") if "timing_s/step" in r]
    source = json.loads((args.data / "source.json").read_text())
    analysis = json.loads((args.data / "analysis.json").read_text())
    if not rows:
        raise SystemExit("No training rows")
    first = min(r["training/global_step"] for r in rows)
    last = max(r["training/global_step"] for r in rows)
    coverage = f"{source['run'].split('/')[-1]} | steps {first:g}-{last:g} | {source['kind']}"
    fv.apply_rc("paper")
    plt.rcParams.update(
        {"font.size": 10, "axes.labelsize": 10, "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9}
    )
    teal, orange, violet, coral = [fv.HUES[k] for k in ("vteal", "vorange", "vviolet", "vcoral")]

    fig, axs = plt.subplots(2, 2, figsize=(10, 6.8))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.91, bottom=0.13, hspace=0.65, wspace=0.3)
    ax = axs[0, 0]
    series(ax, rows, "critic/score/mean", "Train verifier score", coral, 100)
    axes_style(ax, "Correct (%)", (0, 100))
    legend(ax)
    ax = axs[0, 1]
    series(ax, rows, "actor/distillation/loss", "PG distillation loss", teal)
    series(ax, rows, "actor/distillation/abs_loss", "Mean absolute k1", orange)
    axes_style(ax, "Loss / token diagnostic", (0, 0.3))
    legend(ax)
    ax = axs[1, 0]
    series(ax, rows, "response_length/mean", "Mean response length", violet)
    series(ax, rows, "response_length/max", "Observed batch maximum", fv.NEUTRALS["baseline"], linestyle="--")
    axes_style(ax, "Response tokens", (0, 8000))
    legend(ax)
    ax = axs[1, 1]
    series(ax, rows, "response_length/clip_ratio", "At padded response maximum", coral, 100)
    axes_style(ax, "Responses (%)", (0, 100))
    legend(ax)
    save(fig, args.out / "training", coverage + " | training metrics; batches contain different questions")

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.9))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.84, bottom=0.23, wspace=0.38)
    ax = axs[0]
    series(ax, rows, "timing_s/step", "Full step", violet)
    series(ax, rows, "timing_s/gen", "Generation + postprocess", teal)
    series(ax, rows, "timing_s/update_actor", "Actor update", orange)
    axes_style(ax, "Seconds", (0, 240))
    legend(ax)
    ax = axs[1]
    names = ["gen", "update_actor", "old_log_prob", "update_weights", "adv", "other"]
    values = [analysis["stage_seconds"][k] for k in names]
    colors = [teal, orange, violet, fv.NEUTRALS["baseline"], fv.NEUTRALS["vmist"], fv.NEUTRALS["vink"]]
    ax.barh(names, values, color=colors)
    ax.invert_yaxis()
    ax.set(xlabel="Mean seconds / step (excluding step 1)", ylabel="Stage", xlim=(0, 140))
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, args.out / "timing", coverage + " | gen includes verifier, teacher scoring and request waits")

    fig, axs = plt.subplots(1, 3, figsize=(11, 3.9))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.23, wspace=0.4)
    series(axs[0], rows, "actor/perf/max_memory_allocated_gb", "Actor allocated peak", teal)
    series(axs[0], rows, "actor/perf/max_memory_reserved_gb", "Actor reserved peak", orange)
    axes_style(axs[0], "PyTorch memory (GiB)", (0, 22))
    legend(axs[0])
    series(axs[1], rows, "actor/perf/cpu_memory_used_gb", "Host used RAM", fv.NEUTRALS["baseline"])
    axes_style(axs[1], "Host memory (GiB)", (0, 120))
    legend(axs[1])
    series(axs[2], rows, "perf/mfu/actor", "Actor internal MFU", violet, 100)
    axes_style(axs[2], "Model FLOPs utilization (%)", (0, 100))
    legend(axs[2])
    save(fig, args.out / "memory-mfu", coverage + " | actor process memory and internal-update MFU")

    val_rows = [r for r in rows if any(k.startswith("val-core/") for k in r)]
    if val_rows and args.student_baseline:
        validation_with_baselines(
            val_rows[-1], source, args.student_baseline, args.teacher_baseline, args.out, args.data
        )
    elif val_rows:
        row = val_rows[-1]
        datasets = ["MATH-500", "AMC23", "AIME25"]
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.subplots_adjust(left=0.12, right=0.98, top=0.84, bottom=0.22)
        for offset, metric, label, color in [
            (-0.25, "mean@16", "Mean accuracy", teal),
            (0, "best@16/mean", "Bootstrap best@16", orange),
            (0.25, "maj@16/mean", "Bootstrap majority@16", violet),
        ]:
            ax.bar(
                [i + offset for i in range(3)],
                [row[f"val-core/{d}/acc/{metric}"] * 100 for d in datasets],
                width=0.24,
                label=label,
                color=color,
            )
        ax.set(xticks=range(3), xticklabels=datasets, xlabel="Validation dataset", ylabel="Accuracy (%)", ylim=(0, 100))
        legend(ax)
        save(
            fig,
            args.out / "validation",
            coverage + f" | evaluation at step {row['training/global_step']:g}; one observed checkpoint",
        )


if __name__ == "__main__":
    main()
