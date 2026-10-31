# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Plot measured failure reproduction, stage generalization and target KL.

Semantic ledger: every failure-reproduction figure contains one student run;
teacher references retain their own question sets and sample counts. E39 teacher
subset results use matched questions and never serve as full-set horizontal
references. KL endpoints are observed before updates 1 and 16. Qwen stage panels
evaluate fixed banks. Target comparisons use each run's own rollout states and
frozen target, with no claim of a shared evaluation distribution.
Geometric ledger: accuracy is percent, conditional KL is nats/token, and normalized
KL is dimensionless. Different metrics/tasks have separately labelled scales.
All line segments connect measured observations without smoothing or extrapolation.
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--style", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.style))
    import fvstyle as fv
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads(args.data.read_text())
    fv.apply_rc("paper", font="sans", fontsize=10)
    colors = [fv.HUES[k] for k in ("vteal", "vorange", "vviolet")]
    args.output.mkdir(parents=True, exist_ok=True)
    audits = {}
    names = {"math500": "MATH-500", "amc23": "AMC23", "aime24": "AIME24", "aime25": "AIME25", "aime26": "AIME26"}

    def style(ax):
        fv.apply_frame(ax, "open")
        fv.apply_grid(ax, "y-only")

    def finish(fig, name, caption):
        import hashlib

        fig.text(0.5, 0.015, caption, ha="center", color=fv.NEUTRALS["baseline"], fontsize="small")
        fig.canvas.draw()
        # Facets intentionally represent distinct tasks or KL definitions.
        problems = fv.audit(fig, ignore=["facet-ylim-mismatch"])
        audits[name] = {"problems": problems, "errors": [p for p in problems if fv.problem_severity(p) == "error"]}
        sha = hashlib.sha256(args.data.read_bytes()).hexdigest()
        fig.savefig(args.output / f"{name}.png", dpi=220, bbox_inches="tight", metadata={"SourceSHA256": sha})
        fig.savefig(args.output / f"{name}.pdf", bbox_inches="tight", metadata={"Subject": sha})
        plt.close(fig)

    for experiment, run in data["failure"].items():
        tasks = ["math500", "amc23", "aime25"] if experiment != "E43" else list(names)
        fig = plt.figure(figsize=(12 if experiment == "E43" else 10.5, 6.7))
        grid = fig.add_gridspec(2, len(tasks), height_ratios=[1, 1.15])
        fig.subplots_adjust(left=0.08, right=0.98, top=0.85, bottom=0.25, hspace=0.7, wspace=0.35)
        fig.suptitle(f"{experiment}: {run['student']} student\nTeacher: {run['teacher']}", y=0.98)
        teacher = {r["task"]: r["accuracy"] for r in run["teacher_full"]}
        handles = []
        for task_index, task in enumerate(tasks):
            ax = fig.add_subplot(grid[0, task_index])
            style(ax)
            rows = [r for r in run["validation"] if r["task"] == task]
            (line,) = ax.plot(
                [r["outer"] for r in rows],
                [100 * r["accuracy"] for r in rows],
                marker="o",
                markersize=3,
                color=colors[0],
                label="Student avg@16",
            )
            if task_index == 0:
                handles.append(line)
            if task in teacher:
                line = ax.axhline(
                    100 * teacher[task],
                    color=colors[1],
                    linestyle=":",
                    linewidth=1.5,
                    label="Teacher avg@4 (independent)",
                )
                if task_index == 0:
                    handles.append(line)
            values = [100 * r["accuracy"] for r in rows] + ([100 * teacher[task]] if task in teacher else [])
            ax.set_ylim(max(0, min(values) - 7), min(100, max(values) + 7))
            ax.set_title(names[task])
            ax.set_xlabel("Outer")
            ax.set_ylabel("Accuracy (%)")
            ax.set_xticks([0, 50, 100] if experiment != "E39" else [0, 100, 200])
        ax = fig.add_subplot(grid[1, :])
        style(ax)
        for key, label, color, dash in [
            ("raw_first", "Raw KL: before update1", colors[0], "-"),
            ("raw_last", "Raw KL: before update16", colors[0], "--"),
            ("loss_first", "Aligned loss: before update1", colors[2], "-"),
            ("loss_last", "Aligned loss: before update16", colors[2], "--"),
        ]:
            if experiment == "E43" and key.startswith("loss"):
                continue
            (line,) = ax.plot(
                [r["outer"] for r in run["kl"]], [r[key] for r in run["kl"]], color=color, linestyle=dash, label=label
            )
            handles.append(line)
        ax.set_xlabel("Generation batches (outer)")
        ax.set_ylabel("KL / loss (nats/token)")
        ax.set_title("Recorded loss on each fresh student rollout batch")
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, 0.065), ncol=3)
        caption = "Different batches visit different states; raw and EOS-aligned KL remain separate."
        if experiment == "E43":
            caption = (
                "Raw KL equals training loss; teacher uses independent avg@4, student avg@16; "
                "same sampling temperature/top-p/cap."
            )
        finish(fig, f"rq41-{experiment.lower()}", caption)

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.8))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.23, top=0.8, wspace=0.35)
    fig.suptitle("E39: Qwen3-1.7B-Base student vs Qwen3-8B teacher", y=0.98)
    for ax, row in zip(axes, data["failure"]["E39"]["teacher_subset"], strict=True):
        style(ax)
        ax.plot(0, 100 * row["student_acc"], "o", color=colors[0])
        ax.plot(1, 100 * row["teacher_acc"], "s", color=colors[1])
        ax.set_xticks([0, 1], ["Student n=16", "Teacher n=4"])
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(0, 100)
        ax.set_title(row["dataset"])
        ax.set_xlabel("Model on matched subset")
        ax.set_ylabel("Mean accuracy (%)")
    finish(
        fig,
        "rq41-e39-teacher-subset",
        "Exactly 8 matched questions per benchmark; student outer200; "
        "historical teacher subset, not a full-set baseline.",
    )

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9), sharey=True)
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.28, top=0.85, wspace=0.22)
    for ax, (panel, title) in zip(
        axes,
        [("train_all", "Fixed training bank (320 prompts)"), ("held_all", "Held-out Math / Code / IF (128 prompts)")],
        strict=True,
    ):
        style(ax)
        for arm, label, color in [("Q1", "SFT teacher step80", colors[0]), ("Q2", "SFT teacher step640", colors[1])]:
            rows = [r for r in data["qwen_stages"] if r["arm"] == arm and r["panel"] == panel]
            ax.plot(
                [int(r["step"]) for r in rows],
                [float(r["rkl"]) / float(r["initial"]) for r in rows],
                color=color,
                label=label,
            )
        ax.set_title(title)
        ax.set_xlabel("Student optimizer updates")
        ax.set_ylabel("RKL / initial RKL")
        ax.set_ylim(0, 1.05)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.54, 0.09), ncol=2)
    finish(
        fig,
        "rq44-sft-generalization",
        "Same Qwen3-1.7B student, frozen S0/H1024; teacher step is checkpoint identity, not student training budget.",
    )

    rows = data["target_kl"]
    endpoints = {}
    for experiment in ["E55", "E56", "E57"]:
        selected = [r for r in rows if r["experiment"] == experiment]
        groups = {}
        for r in selected:
            groups.setdefault(r["outer"], []).append(r)
        endpoints[experiment] = [
            (outer, sorted(rs, key=lambda r: r["update"])[0], sorted(rs, key=lambda r: r["update"])[-1])
            for outer, rs in sorted(groups.items())
            if len(rs) == 16
        ]
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.4))
    fig.subplots_adjust(left=0.1, right=0.98, top=0.88, bottom=0.2, hspace=0.55, wspace=0.3)
    specs = [
        ("teacher", 1, "Teacher KL before update1"),
        ("teacher", 2, "Teacher KL before update16"),
        ("target", 2, "Run-specific target KL before update16"),
        ("anchor", 2, "Anchor KL before update16"),
    ]
    for ax, (metric, endpoint, title) in zip(axes.flat, specs, strict=True):
        style(ax)
        for experiment, color in zip(endpoints, colors, strict=True):
            records = endpoints[experiment]
            ax.plot([r[0] for r in records], [r[endpoint][metric] for r in records], color=color, label=experiment)
        ax.set_title(title)
        ax.set_xlabel("Generation batches (outer)")
        ax.set_ylabel("KL (nats/token)")
        ax.set_ylim(bottom=0)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.55, 0.075), ncol=3)
    finish(
        fig,
        "rq5-kl-over-outers",
        "Own online rollout states and target per run; E55/E56/E57 use kappa=0/0.5/1, eta=0.1; no smoothing.",
    )

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.27, top=0.83, wspace=0.35)
    for ax, metric, title in zip(
        axes,
        ["teacher", "target", "anchor"],
        ["Original teacher KL", "Target KL / first target KL", "Anchor KL"],
        strict=True,
    ):
        style(ax)
        for experiment, color in zip(endpoints, colors, strict=True):
            selected = sorted(
                [r for r in rows if r["experiment"] == experiment and r["outer"] == 30], key=lambda r: r["update"]
            )
            values = [r[metric] / selected[0][metric] if metric == "target" else r[metric] for r in selected]
            ax.plot(range(1, 17), values, color=color, marker="o", markersize=3, label=experiment)
        ax.set_title(title)
        ax.set_xlabel("Pre-update observation (1 to 16)")
        ax.set_ylabel("Relative KL" if metric == "target" else "KL (nats/token)")
        ax.set_xticks([1, 6, 11, 16])
        ax.set_ylim(bottom=0)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.55, 0.08), ncol=3)
    finish(
        fig,
        "rq5-kl-within-batch",
        "Common outer30; each run freezes its own batch/target; 16 pre-update observations cover 15 completed updates.",
    )

    (args.output / "followup-figure-audit.json").write_text(json.dumps(audits, indent=2) + "\n")
    errors = {k: v["errors"] for k, v in audits.items() if v["errors"]}
    if errors:
        raise RuntimeError(errors)
    print(f"Rendered {len(audits)} figures; audit errors = 0")


if __name__ == "__main__":
    main()
