#!/usr/bin/env python3
"""Measured five-run comparison: one color per run, no imputed checkpoints.

Ledger: eval lines are checkpoint observations, CI bars resample prompts, update
losses precede their optimizer operation, and reused batches share trajectories.
Cost bars use nonoverlapping outer+validation timers; system means are periodic
samples. None of these plots represents variation across independent seeds.
"""

# ruff: noqa: E402 -- Resolve the shared plotting style before importing it.
import argparse
import gzip
import json
from pathlib import Path

import fvlocate

fvlocate.ensure()
import fvstyle as fv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.data / "comparison.json.gz", "rt") as f:
        bundle = json.load(f)
    runs, series = bundle["analysis"]["runs"], bundle["series"]
    order = list(runs)
    fv.apply_rc("paper")
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9})
    palette = fv.series_colors("paper", 4)
    colors = dict(zip(order, [palette[0], palette[1], fv.NEUTRALS["baseline"], palette[2], palette[3]], strict=True))
    labels = {
        "E13": "E13 full x1",
        "E14": "E14 full x1 / half LR",
        "E15": "E15 sampled x1",
        "E16": "E16 full x4",
        "E17": "E17 full x16 (partial)",
    }
    datasets = {"math500": "MATH-500", "amc23": "AMC23", "aime25": "AIME25"}

    def figure(nrows, ncols):
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows + 0.7), squeeze=False)
        for ax in axes.flat:
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color=fv.NEUTRALS["vmist"], lw=0.6)
            ax.set_axisbelow(True)
        return fig, axes.flatten()

    def save(fig, name, caption, legend=True):
        if legend:
            fig.legend(
                handles=[Line2D([0], [0], color=colors[k], label=labels[k], lw=2) for k in order],
                loc="upper center",
                ncol=3,
                frameon=False,
                fontsize=9,
            )
        fig.tight_layout(rect=(0, 0.10, 1, 0.88 if legend else 0.97), h_pad=3, w_pad=2)
        fig.text(0.5, 0.02, caption, ha="center", fontsize=9, color=fv.NEUTRALS["baseline"])
        fig.canvas.draw()
        issues = fv.audit(fig, print_slot=fig.get_figwidth())
        lines = fv.format_audit_lines(issues)
        (args.out / (name + ".audit.txt")).write_text("\n".join(lines) + "\n" if issues else "PASS\n")
        if any(fv.problem_severity(i) == "error" for i in issues):
            raise ValueError((name, lines))
        for ext in ("png", "pdf", "svg"):
            fig.savefig(args.out / f"{name}.{ext}", dpi=150, facecolor=fv.NEUTRALS["paper"])
        plt.close(fig)

    def line(ax, key, metric, scale=1, smooth=1):
        points = series[key].get(metric, {}).get("points", [])
        if not points:
            return
        x, y = np.array(points).T
        y = y * scale
        if smooth > 1:
            y = np.array([np.mean(y[max(0, i - smooth + 1) : i + 1]) for i in range(len(y))])
        ax.plot(x, y, color=colors[key], lw=1.4, label=labels[key])

    for mode in ("outer", "optimizer", "accounted_hours"):
        fig, axes = figure(1, 3)
        for ax, (dataset, title) in zip(axes, datasets.items(), strict=True):
            for key, run in runs.items():
                rows = run["evaluation"]
                ax.plot(
                    [e[mode] for e in rows],
                    [100 * e["metrics"][f"eval/{dataset}/acc/mean"] for e in rows],
                    color=colors[key],
                    marker="o",
                    markersize=3,
                    lw=1.4,
                    label=labels[key],
                )
            ax.set_title(title)
            ax.set_ylabel("Mean accuracy (%)")
            ax.set_xlabel(
                {
                    "outer": "Outer step",
                    "optimizer": "Applied optimizer updates (log axis)",
                    "accounted_hours": "Accounted train + validation hours",
                }[mode]
            )
            if mode == "optimizer":
                ax.set_xscale("log")
        save(
            fig,
            "evaluation-" + mode,
            "Actual checkpoints only; one seed; E17 has no recorded evaluation after outer step 160",
        )

    fig, axes = figure(2, 2)
    for key, run in runs.items():
        rows = [e for e in run["evaluation"] if e["optimizer"] <= 400]
        x = [e["optimizer"] for e in rows]
        for ax, metric, scale in zip(
            axes,
            [
                "acc/mean",
                "generation/response_length/p95",
                "acc/prediction_missing_count",
                "generation/finish_reason/length_count",
            ],
            [100, 1, 100 / 8000, 100 / 8000],
            strict=True,
        ):
            ax.plot(
                x,
                [e["metrics"]["eval/math500/" + metric] * scale for e in rows],
                marker="o",
                markersize=4,
                color=colors[key],
            )
    for ax, title, ylabel in zip(
        axes,
        ["MATH-500 mean accuracy", "MATH-500 response length p95", "Missing parsed answers", "Length-limit stops"],
        ["Accuracy (%)", "Tokens", "Fraction of 8000 responses (%)", "Fraction of 8000 responses (%)"],
        strict=True,
    ):
        ax.set_title(title)
        ax.set_xlabel("Applied optimizer updates (early recorded checkpoints)")
        ax.set_ylabel(ylabel)
    save(
        fig,
        "math-transient",
        "Checkpoints through update 400; E17 first evaluates at update 320; association does not establish causation",
    )

    fig, axes = figure(1, 3)
    for ax, (dataset, title) in zip(axes, datasets.items(), strict=True):
        prefix = f"eval/{dataset}/acc/"
        for i, key in enumerate(order):
            e = next(e for e in runs[key]["evaluation"] if e["outer"] == 160)["metrics"]
            y, low, high = [100 * e[prefix + k] for k in ("mean", "mean_ci_low", "mean_ci_high")]
            ax.errorbar(i, y, yerr=[[y - low], [high - y]], fmt="o", color=colors[key], capsize=4)
        ax.set_xticks(range(len(order)), order)
        ax.set_xlabel("Run at outer step 160")
        ax.set_ylabel("Mean accuracy / recorded 95% CI (%)")
        ax.set_title(title)
    save(fig, "evaluation-ci", "Intervals resample evaluation prompts, not training seeds; no paired significance test")

    fig, axes = figure(2, 3)
    for row, metric in enumerate(("pass", "majority")):
        for col, (dataset, title) in enumerate(datasets.items()):
            ax = axes[row * 3 + col]
            for key in order:
                e = next(e for e in runs[key]["evaluation"] if e["outer"] == 160)["metrics"]
                ax.plot(
                    range(1, 17),
                    [100 * e[f"eval/{dataset}/acc/{metric}@{k}"] for k in range(1, 17)],
                    color=colors[key],
                    lw=1.5,
                )
            ax.set_title(title + " / " + metric)
            ax.set_xlabel("Samples k")
            ax.set_ylabel(metric + "@k (%)")
    save(
        fig,
        "sample-budget",
        "Outer step 160; pass@k uses an oracle, majority@k votes on parsed answers; curves use logged estimates",
    )

    fig, axes = figure(2, 2)
    for key in order:
        line(axes[0], key, "rollout/score/mean", 100, 20)
        line(axes[1], key, "rollout/response_length/mean", 1, 20)
        counts = dict(series[key]["rollout/response_count"]["points"])
        rows = series[key]["rollout/finish_reason/length_count"]["points"]
        vals = [100 * v / counts[x] for x, v in rows]
        axes[2].plot(
            [r[0] for r in rows], [np.mean(vals[max(0, i - 19) : i + 1]) for i in range(len(vals))], color=colors[key]
        )
        line(axes[3], key, "optim/actor/grad_norm_preclip")
    for ax, title, ylabel in zip(
        axes,
        ["Training score", "Training response length", "Length-limit stops", "Gradient norm"],
        ["Mean task score (%)", "Tokens", "Fraction (%)", "Global norm (log axis)"],
        strict=True,
    ):
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Optimizer update" if ax is axes[3] else "Outer step")
    axes[3].set_yscale("log")
    axes[3].set_xscale("log")
    axes[3].axhline(1, color=fv.NEUTRALS["vink"], ls="--", lw=1)
    save(
        fig,
        "training",
        "Rollout panels: trailing 20-step means; gradient panel: raw preclip observations, threshold = 1",
    )

    fig, axes = figure(1, 3)
    for key in order:
        batches = runs[key]["reuse"]
        if not batches:
            continue
        axes[0].plot([b["outer"] for b in batches], [b["first"] for b in batches], color=colors[key], lw=1.3)
        line(axes[1], key, "distill/full_kl")
        if key in ("E16", "E17"):
            selected = [b for b in batches if 141 <= b["outer"] <= 160]
            n = len(selected[0]["values"])
            axes[2].plot(
                range(1, n + 1),
                np.mean([[v / b["first"] for v in b["values"]] for b in selected], axis=0),
                color=colors[key],
                marker="o",
                markersize=3,
            )
    for ax, title, xlabel, ylabel in zip(
        axes,
        ["Fresh-batch KL before first update", "All full-KL updates", "Within-batch reduction"],
        ["Outer step", "Optimizer update (log axis)", "Update index within batch"],
        ["Full KL (log axis)", "Full KL (log axis)", "KL / first-update KL"],
        strict=True,
    ):
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xscale("log")
    save(
        fig,
        "full-kl-reuse",
        "E15 sampled surrogate is not plotted as exact KL; within-batch panel averages outer steps 141-160",
    )

    fig, axes = figure(1, 2)
    group_colors = fv.series_colors("paper", 4)
    components = [[], [], [], []]
    for key in order:
        reg = runs[key]["training"]["160"]["time"]["regular"]

        def val(stage, reg=reg):
            return reg.get(f"perf/timing/{stage}_seconds", {}).get("mean", 0)

        known = [val("rollout_pipeline"), val("teacher_forward") + val("teacher_transfer"), val("update_actor")]
        for i, v in enumerate(known + [val("outer_step") - sum(known)]):
            components[i].append(v)
    bottom = np.zeros(len(order))
    for i, label in enumerate(("Rollout", "Teacher + transfer", "Actor", "Other step work")):
        axes[0].bar(order, components[i], bottom=bottom, color=group_colors[i], label=label)
        bottom += components[i]
    axes[0].set_ylabel("Mean regular-step seconds")
    axes[0].set_xlabel("Run through outer step 160")
    axes[0].legend(loc="upper left", frameon=False, fontsize=8)
    for i, key in enumerate(order):
        t = runs[key]["training"]["160"]["time"]
        axes[1].bar(i, t["training_hours"], color=group_colors[0], label="Training" if i == 0 else None)
        axes[1].bar(
            i,
            t["validation_hours"],
            bottom=t["training_hours"],
            color=group_colors[1],
            label="Validation" if i == 0 else None,
        )
    axes[1].set_xticks(range(len(order)), order)
    axes[1].set_xlabel("Run through outer step 160")
    axes[1].set_ylabel("Accounted hours")
    axes[1].legend(loc="upper left", frameon=False, fontsize=8)
    save(
        fig,
        "cost",
        ("Checkpoint is inside training; E13 uses A800, others A100; hardware is not controlled"),
        False,
    )

    fig, axes = figure(1, 3)
    for i, key in enumerate(order):
        run = runs[key]
        phases = run["system"]["phases"]
        axes[0].bar(i - 0.18, phases["train_trim10"]["gpu"]["mean"], width=0.35, color=colors[key])
        axes[0].bar(
            i + 0.18, phases["validation_trim10"]["gpu"]["mean"], width=0.35, color=colors[key], hatch="//", alpha=0.5
        )
        axes[1].bar(i, run["selected_metrics"]["perf/actor/allocated_peak_gib"]["max"], color=colors[key])
        axes[2].bar(i, run["system"]["nvml_sampled_max_gib"], color=colors[key])
    for ax, title, ylabel in zip(
        axes,
        ["GPU busy: train / hatched validation", "Actor process peak allocated", "Whole-card sampled memory peak"],
        ["Mean of complete GPU samples (%)", "GiB (PyTorch lifetime peak)", "GiB (NVML)"],
        strict=True,
    ):
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(len(order)), order)
        ax.set_xlabel("Run / available snapshot")
        ax.set_ylabel(ylabel)
    save(
        fig,
        "resources",
        "Approximate phases with 10-second edge trimming; periodic samples can miss peaks; GPU busy is not MFU",
    )


if __name__ == "__main__":
    main()
