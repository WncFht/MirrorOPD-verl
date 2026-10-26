#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Measured long-window and mirror comparisons; no synthetic values.

Ledger: one color per run; evaluation markers are actual optimizer checkpoints.
New-batch curves average 20 consecutive recorded first-update observations.
Inner profiles use outer 205-224, before each optimizer update. Position profiles
pool sums/counts over outer 181-200; each panel has its own objective/scale.
Geometry: independent dataset scales; equal axes for comparable teacher KL;
no missing values, evaluations or seed variability are imputed.
"""

# ruff: noqa: E402
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
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
    x = json.loads((args.data / "analysis.json").read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    fv.apply_rc("paper", fontsize=8)
    palette = fv.series_colors("paper", 4)
    colors = {
        "E17": fv.NEUTRALS["baseline"],
        "E20": palette[0],
        "E35": palette[1],
        "E45": palette[2],
        "E47": palette[3],
    }
    audits = {}

    def axes(n=3):
        fig, axs = plt.subplots(1, n, figsize=(5.5, 2.55), squeeze=False)
        for ax in axs.flat:
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color=fv.NEUTRALS["vmist"], lw=0.5)
            ax.set_axisbelow(True)
        return fig, list(axs.flat)

    def finish(fig, name, caption, handles, labels):
        fig.tight_layout(rect=(0, 0.22, 1, 1), pad=0.8)
        if handles:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.075),
                ncol=3 if len(labels) == 6 else min(5, len(labels)),
                frameon=False,
                fontsize=7,
            )
        fig.text(0.5, 0.012, caption, ha="center", fontsize=5.8, color=fv.NEUTRALS["baseline"])
        fig.canvas.draw()
        # Dataset/target panels intentionally have independent scales, declared in captions.
        ignored = (
            ["facet-ylim-mismatch"]
            if name in ["length-evaluation", "mirror-evaluation", "inner-fit", "position-fit"]
            else []
        )
        audits[name] = [str(v) for v in fv.audit(fig, ignore=ignored)]
        for ext in ["png", "pdf"]:
            fig.savefig(args.out / f"{name}.{ext}", dpi=240)
        plt.close(fig)

    for name, labels in [("length-evaluation", ["E17", "E35"]), ("mirror-evaluation", ["E20", "E45", "E47"])]:
        fig, axs = axes()
        for ax, ds, title in zip(axs, ["math500", "amc23", "aime25"], ["MATH-500", "AMC23", "AIME25"], strict=True):
            for label in labels:
                ev = [e for e in x["runs"][label]["evaluation"] if e["optimizer"] > 0]
                ax.plot(
                    [e["optimizer"] / 1000 for e in ev],
                    [100 * e["metrics"][f"eval/{ds}/acc/mean"] for e in ev],
                    color=colors[label],
                    marker="o",
                    ms=2.6,
                    lw=1.1,
                    label=label,
                )
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("Updates (1000s)")
            ax.set_ylabel("Accuracy (%)")
            ax.set_xticks([0, 1, 2, 3, 4])
            ax.tick_params(labelsize=7)
        handles = [Line2D([], [], color=colors[run_label], marker="o", ms=3, lw=1) for run_label in labels]
        finish(fig, name, "Actual checkpoints; panel scales differ; one seed per run.", handles, labels)
    fig, axs = axes(2)
    for ax, labels in zip(axs, [["E17", "E35"], ["E20", "E45", "E47"]], strict=True):
        for label in labels:
            rows = x["runs"][label]["batch_curves"]
            values = np.array([r["teacher_first"] for r in rows])
            smoothed = np.convolve(values, np.ones(20) / 20, "valid")
            ax.plot([r["outer"] for r in rows[19:]], smoothed, color=colors[label], lw=1.2, label=label)
        ax.set_yscale("log")
        ax.set_xlabel("Outer step")
        ax.set_ylabel("New-batch teacher KL")
        ax.set_ylim(5e-4, 4e-2)
        ax.set_xlim(0, 260)
    labels = ["E17", "E35", "E20", "E45", "E47"]
    finish(
        fig,
        "teacher-learning",
        "20-batch means; on-policy prefixes differ between runs.",
        [Line2D([], [], color=colors[run_label], lw=1.2) for run_label in labels],
        labels,
    )
    fig, axs = axes()
    fields = [("teacher", "Teacher KL"), ("target", "Target KL"), ("anchor", "Anchor KL")]
    for ax, (field, title) in zip(axs, fields, strict=True):
        labels = ["E20", "E45", "E47"] if field == "teacher" else ["E45", "E47"]
        for label in labels:
            metric = "distill/full_kl" if label == "E20" else f"distill/{field}_kl"
            values = x["windows"][label]["205-224"]["metrics"][metric]["profile"]
            ax.plot(range(1, 17), np.array(values) * 1e4, color=colors[label], marker="o", ms=2, lw=1.1, label=label)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Observation index")
        ax.set_ylabel("KL (1e-4)")
        ax.set_xticks([1, 4, 8, 12, 16])
    labels = ["E20", "E45", "E47"]
    finish(
        fig,
        "inner-fit",
        "Outer 205-224; pre-update observations; independent scales.",
        [Line2D([], [], color=colors[run_label], marker="o", ms=3, lw=1) for run_label in labels],
        labels,
    )
    fig, axs = axes()
    handles = []
    labels = []
    for ax, label in zip(axs, ["E35", "E45", "E47"], strict=True):
        positions = x["windows"][label]["181-200"]["position"]
        for phase, style in [("first", "--"), ("last", "-")]:
            values = [v["mean"] * 1e4 for v in positions[phase]]
            ax.plot(
                range(5, 100, 10),
                values,
                color=colors[label],
                linestyle=style,
                lw=1.2,
                marker="o",
                ms=2,
                label=f"{label} {phase}",
            )
        ax.set_title(label + (" Full-KL" if label == "E35" else " objective"), fontsize=8)
        ax.set_xlabel("Response position (%)")
        ax.set_ylabel("Value (1e-4)")
        ax.set_xticks([0, 50, 100])
        for phase, style in [("first", "--"), ("last", "-")]:
            handles.append(Line2D([], [], color=colors[label], lw=1.2, ls=style))
            labels.append(f"{label} {phase}")
    finish(
        fig, "position-fit", "Outer 181-200; token-weighted bins; objectives and panel scales differ.", handles, labels
    )
    fig, axs = axes()
    cost = x["comparisons"]["mirror"]["costs"]["224"]
    labels = ["E20", "E45", "E47"]
    for ax, key, title in zip(
        axs,
        ["perf/timing/update_actor_seconds", "perf/timing/rollout_pipeline_seconds", "perf/timing/outer_step_seconds"],
        ["Actor (16 updates)", "Rollout pipeline", "Training outer"],
        strict=True,
    ):
        vals = [cost[run_label]["timing"]["regular"][key]["mean"] for run_label in labels]
        ax.bar(labels, vals, color=[colors[run_label] for run_label in labels], width=0.55)
        ax.set_title(title, fontsize=8)
        ax.set_ylabel("Seconds")
        ax.set_xlabel("Run")
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals) * 0.03, f"{v:.1f}", ha="center", fontsize=7)
        ax.set_ylim(0, 170)
    finish(fig, "mirror-cost", "Through outer 224; regular steps; E20 A800 vs E45/E47 A100.", [], [])
    (args.out / "audit.json").write_text(json.dumps(audits, indent=2) + "\n")
    print(json.dumps(audits, indent=2))


if __name__ == "__main__":
    main()
