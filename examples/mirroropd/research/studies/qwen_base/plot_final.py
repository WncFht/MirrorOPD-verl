#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Render audited figures from frozen E39 analysis tables.

Semantic ledger: validation series identify datasets, training series identify
first/last pre-update observations, position series identify training windows.
Repetition panels use one fixed text sample per question, not all 16 samples.
Teacher/student comparison uses only the same eight questions per benchmark;
subset teacher results never become full-validation horizontal reference lines.
Geometry: equal-width panels; outer-step axes have common 0..200 support.
Position curves use count-weighted relative-position bins, never raw sums.
"""


def main():
    import argparse
    import csv
    import hashlib
    import json
    from pathlib import Path

    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from examples.mirroropd.research.core.plotting import load_style

    fv = load_style()
    import matplotlib  # noqa: E402, I001 - runtime source/style path is configured before import

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402, I001 - runtime source/style path is configured before import

    fv.apply_rc("paper", fontsize=8)
    args.out.mkdir(parents=True, exist_ok=True)
    colors = [fv.HUES[k] for k in ("vteal", "vorange", "vviolet")]

    def csvrows(name):
        with (args.data / name).open() as f:
            return list(csv.DictReader(f))

    def figure(n=3):
        fig, axes = plt.subplots(1, n, figsize=(7.3, 3.95))
        axes = np.atleast_1d(axes)
        fig.subplots_adjust(left=0.10, right=0.98, bottom=0.31, top=0.70, wspace=0.50)
        return fig, axes

    def outer_axis(ax, label, ylim=None):
        ax.set(xlabel="Outer step", ylabel=label, xlim=(-10, 210))
        ax.set_xticks([0, 100, 200])
        ax.grid(axis="y", alpha=0.2)
        if ylim is not None:
            ax.set_ylim(*ylim)

    audits = {}

    def finish(fig, axes, name, caption, legend=True, ignore=()):
        fig.text(
            0.5,
            0.98,
            "E39: Student Qwen3-1.7B-Base  ←  Teacher Qwen3-8B\n"
            "Thinking OFF (enable_thinking=False); Full-RKL, reuse16, batch64, LR 1e-6",
            ha="center",
            va="top",
        )
        if legend:
            fig.legend(
                *axes[0].get_legend_handles_labels(),
                loc="upper center",
                bbox_to_anchor=(0.52, 0.85),
                ncol=3,
                frameon=False,
            )
        fig.text(0.5, 0.04, caption, ha="center", fontsize="medium", color=fv.NEUTRALS["baseline"])
        fig.canvas.draw()
        problems = fv.audit(fig, ignore=ignore)
        audits[name] = problems
        if any(fv.problem_severity(p) == "error" for p in problems):
            raise RuntimeError(f"{name}: {problems}")
        (args.out / (name + ".audit.txt")).write_text("\n".join(map(str, problems)) + "\n")
        for ext in ["png", "pdf", "svg"]:
            fig.savefig(args.out / (name + "." + ext), dpi=260)
        print(name, problems)
        plt.close(fig)

    vals = csvrows("validation.csv")
    fig, axes = figure()
    for (slug, name), color in zip(
        [("math500", "MATH-500"), ("amc23", "AMC23"), ("aime25", "AIME25")], colors, strict=False
    ):
        rows = [r for r in vals if r["slug"] == slug]
        x = [int(r["outer"]) for r in rows]
        for ax, field, scale in zip(axes, ["accuracy", "mean_length", "cap_fraction"], [100, 0.001, 100], strict=False):
            ax.plot(x, [float(r[field]) * scale for r in rows], color=color, marker="o", label=name)
        axes[0].fill_between(
            x,
            [float(r["accuracy_ci_low"]) * 100 for r in rows],
            [float(r["accuracy_ci_high"]) * 100 for r in rows],
            color=color,
            alpha=0.12,
            linewidth=0,
        )
    for ax, label, ylim in zip(
        axes,
        ["Accuracy: avg@16 (%)", "Mean length (k tokens)", "Length-capped (%)\ncap = 31,744 tokens"],
        [(-4, 100), (-0.5, 16), (-1, 50)],
        strict=False,
    ):
        outer_axis(ax, label, ylim)
    finish(
        fig,
        axes,
        "final-validation",
        "Student avg@16 = mean correctness of 16 responses/question; bands: 95% question-bootstrap CI.\n"
        "Teacher full-set reference unavailable; matched 8-question subset shown separately.\n"
        "Evaluation: thinking OFF; T=0.7, top-p=0.9; response cap=31,744 tokens.",
    )

    train = csvrows("training.csv")
    x = [int(r["outer"]) for r in train]
    fig, axes = figure()
    for end, color, label in [("first", colors[0], "First inner update"), ("last", colors[1], "Last inner update")]:
        for ax, key in zip(axes, ["kl", "identity", "grad"], strict=False):
            ax.plot(x, [float(r[f"{key}_{end}"]) for r in train], color=color, label=label)
    for ax, label in zip(axes, ["Optimized KL (nats)", "EOS identity KL (nats)", "Gradient norm"], strict=False):
        outer_axis(ax, label)
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(bottom=0)
    axes[2].set_yscale("log")
    finish(fig, axes, "final-optimization", "First/last are pre-update measurements within the same 16-update batch")

    fig, axes = figure()
    for ax, key, label, scale in zip(
        axes,
        ["mean_length", "cap_fraction", "train_score"],
        ["Train length (k tokens)", "Train length-capped (%)\ncap = 7,168 tokens", "Training accuracy: avg@1 (%)"],
        [0.001, 100, 100],
        strict=False,
    ):
        ax.plot([int(r["outer"]) for r in train], [float(r[key]) * scale for r in train], color=colors[0])
        outer_axis(ax, label)
        ax.set_ylim(bottom=-0.5 if key == "cap_fraction" else 0)
    finish(
        fig,
        axes,
        "final-training-behavior",
        "Training cap=7,168 tokens; one response/question (avg@1).\n"
        "DAPO batches change each step; correctness is diagnostic, outside the loss.",
        legend=False,
    )

    pos = csvrows("position-kl.csv")
    fig, axes = figure(2)
    for (lo, hi), color, label in [((1, 10), colors[0], "Steps 1-10"), ((181, 200), colors[1], "Steps 181-200")]:
        for ax, endpoint in zip(axes, ["first", "last"], strict=False):
            means = []
            for i in range(0, 100, 10):
                rows = [
                    r
                    for r in pos
                    if lo <= int(r["outer"]) <= hi
                    and r["endpoint"] == endpoint
                    and r["bin"] == f"p{i:02d}_{i + 10:02d}"
                ]
                means.append(sum(float(r["sum"]) for r in rows) / sum(float(r["token_count"]) for r in rows))
            ax.plot(np.arange(5, 100, 10), means, color=color, marker="o", label=label)
    for ax, title in zip(axes, ["First inner update", "Last inner update"], strict=False):
        ax.set(
            xlabel="Relative response position (%)", ylabel="Raw vocabulary KL (nats)", xlim=(0, 100), ylim=(0, 0.40)
        )
        ax.set_xticks([0, 50, 100])
        ax.grid(axis="y", alpha=0.2)
        ax.set_title(title)
    finish(
        fig,
        axes,
        "final-position-kl",
        "Token-weighted bins; raw KL includes stop-token identity; lengths change across windows",
    )

    panel = csvrows("sample-panel-summary.csv")
    fig, axes = figure()
    for name, color in zip(["MATH-500", "AMC23", "AIME25"], colors, strict=False):
        rows = [r for r in panel if r["dataset"] == name]
        x = [int(r["step"]) for r in rows]
        for ax, key in zip(axes, ["loop_fraction", "repeat8", "short_fraction"], strict=False):
            ax.plot(x, [float(r[key]) * 100 for r in rows], color=color, marker="o", label=name)
    for ax, label in zip(
        axes, ["Long-loop responses (%)", "Repeated 8-grams (%)", "Responses under 32 tokens (%)"], strict=False
    ):
        outer_axis(ax, label, (-2, 60))
    finish(
        fig,
        axes,
        "final-repetition",
        "Fixed sample_index=0: 570 responses/checkpoint; repetition uses re-tokenized text",
    )

    resources = json.loads((args.data / "resources.json").read_text())
    fig, axes = figure()
    fig.subplots_adjust(left=0.115, wspace=0.5)
    x = [r["outer"] for r in resources]
    for ax, key, label in zip(
        axes,
        ["minutes", "mean_gpu_util_pct", "max_mem_gib"],
        ["Validation time (min)", "Mean GPU utilization (%)", "Sampled peak memory (GiB)"],
        strict=False,
    ):
        ax.plot(x, [r[key] for r in resources], color=colors[0], marker="o")
        outer_axis(ax, label)
    axes[0].set_yscale("log")
    axes[1].set_ylim(0, 100)
    axes[2].set_ylim(0, 80)
    finish(
        fig,
        axes,
        "final-resources",
        "Validation windows only; GPU samples every ~30 s; utilization includes tail waiting",
        legend=False,
    )

    reference = json.loads((args.data / "teacher-subset-reference.json").read_text())
    rows = reference["rows"]
    assert [r["dataset"] for r in rows] == ["MATH-500", "AMC23", "AIME25"]
    assert all(r["questions"] == 8 and r["teacher_n"] == 4 and r["student_n"] == 16 for r in rows)
    fig, axes = figure(2)
    x = np.arange(len(rows))
    for field, color, label, shift in [
        ("student", colors[0], "Student outer200 avg@16", -0.18),
        ("teacher", colors[1], "Teacher avg@4 (8 questions/set)", 0.18),
    ]:
        for ax, suffix, scale in zip(axes, ["acc", "length"], [100, 0.001], strict=True):
            values = [r[f"{field}_{suffix}"] * scale for r in rows]
            bars = ax.bar(x + shift, values, width=0.34, color=color, label=label)
            ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=3)
    for ax, ylabel, limit in zip(
        axes, ["Accuracy: mean correctness (%)", "Mean length (k tokens)"], [105, 11], strict=True
    ):
        ax.set(
            xticks=x,
            xticklabels=[r["dataset"] for r in rows],
            ylabel=ylabel,
            ylim=(0, limit),
            xlabel="Exactly matched question subsets",
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
    finish(
        fig,
        axes,
        "final-teacher-subset",
        "Matched 8 questions/set; independent teacher avg@4 vs final student avg@16.\n"
        "Both thinking OFF; T=0.7, top-p=0.9; response cap=31,744 tokens; not a full-set baseline.",
    )
    sources = [
        "validation.csv",
        "training.csv",
        "position-kl.csv",
        "sample-panel-summary.csv",
        "resources.json",
        "teacher-subset-reference.json",
        "config.json",
    ]
    (args.out / "figure-audit.json").write_text(json.dumps(audits, indent=2) + "\n")
    (args.out / "figure-provenance.json").write_text(
        json.dumps(
            {
                "experiment": "E39",
                "run_id": "f0e44ec7",
                "student": "Qwen3-1.7B-Base",
                "teacher": "Qwen3-8B",
                "enable_thinking": False,
                "student_metric": "avg@16: mean correctness per question over 16 responses",
                "teacher_full_set": None,
                "teacher_subset": reference,
                "validation_response_cap": 31744,
                "training_response_cap": 7168,
                "sources": {
                    name: {
                        "path": str(args.data / name),
                        "sha256": hashlib.sha256((args.data / name).read_bytes()).hexdigest(),
                    }
                    for name in sources
                },
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
