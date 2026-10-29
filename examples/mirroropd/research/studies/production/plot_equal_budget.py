#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Measured E18/E19/E20 comparisons; one arm keeps one color throughout.

Ledger: evaluation points are actual optimizer checkpoints; error bars are
within-run prompt bootstrap intervals, not seed variation or paired tests.
KL windows contain complete batches in a fixed 512-update interval; every loss
precedes its optimizer update. Reuse profiles normalize each batch separately.
Costs sum nonoverlapping outer and validation timers, not nested stages.
Geometry: panels have independent units/scales, except matching percentage
facets. No missing evaluation, position, entropy or ESS values are imputed.
"""

# ruff: noqa: E402 -- Locate shared figure-viz styling before importing it.
import argparse
import json
from pathlib import Path

from examples.mirroropd.research.core.plotting import load_style

LABELS = {"E18": "E18: reuse 64 (A800)", "E19": "E19: reuse 32 (A100)", "E20": "E20: reuse 16 (A800)"}
DATASETS = {"math500": "MATH-500", "amc23": "AMC23", "aime25": "AIME25"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    fv = load_style()
    runs = json.loads((args.data / "analysis.json").read_text())["runs"]
    args.out.mkdir(parents=True, exist_ok=True)
    fv.apply_rc("paper", fontsize=10)
    colors = dict(zip(runs, fv.series_colors("paper", len(runs)), strict=True))

    def figure(rows=1, cols=3):
        fig, axes = plt.subplots(rows, cols, figsize=(11.4 if cols == 3 else 8.2, 3.8 * rows), squeeze=False)
        for ax in axes.flat:
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color=fv.NEUTRALS["vmist"], lw=0.55)
            ax.set_axisbelow(True)
        return fig, axes

    def save(fig, name, caption, legend=True):
        if legend:
            handles = [Line2D([], [], color=colors[k], lw=1.8, label=LABELS[k]) for k in runs]
            fig.legend(
                handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.065), ncol=3, frameon=False, fontsize=9
            )
        bottom = 0.22 if fig.get_figheight() < 5 else 0.13
        fig.tight_layout(rect=(0, bottom, 1, 1), h_pad=2.2, w_pad=2)
        fig.text(0.5, 0.018, caption, ha="center", va="bottom", fontsize=8, color=fv.NEUTRALS["baseline"])
        fig.canvas.draw()
        issues = fv.audit(fig, print_slot=fig.get_figwidth())
        lines = fv.format_audit_lines(issues)
        (args.out / f"{name}.audit.txt").write_text("\n".join(lines) + "\n" if issues else "PASS: no audit issues.\n")
        if any(fv.problem_severity(p) == "error" for p in issues):
            raise RuntimeError(f"Figure audit failed: {name}: {lines}")
        for suffix in ("png", "pdf"):
            fig.savefig(args.out / f"{name}.{suffix}", dpi=180, facecolor=fv.NEUTRALS["paper"])
        plt.close(fig)
        print(name, "audit warnings", len(issues))

    fig, axes = figure()
    for ax, (ds, name) in zip(axes.flat, DATASETS.items(), strict=True):
        for label, run in runs.items():
            ev = run["evaluation"]
            ax.plot(
                [x["optimizer"] for x in ev],
                [100 * x["metrics"][f"eval/{ds}/acc/mean"] for x in ev],
                color=colors[label],
                marker="o",
                markersize=3.5,
                lw=1.5,
            )
        ax.set(title=name, xlabel="Optimizer updates", ylabel="Mean accuracy (%)", xticks=[512, 1536, 2560, 4096])
    save(
        fig,
        "evaluation",
        "Actual checkpoints; E20 ends at update 2560 for evaluation. Independent dataset scales; one training seed.",
    )

    fig, axes = figure()
    for ax, (ds, name) in zip(axes.flat, DATASETS.items(), strict=True):
        for i, (label, run) in enumerate(runs.items()):
            m = next(e for e in run["evaluation"] if e["optimizer"] == 2560)["metrics"]
            prefix = f"eval/{ds}/acc/"
            value, low, high = [100 * m[prefix + k] for k in ["mean", "mean_ci_low", "mean_ci_high"]]
            ax.errorbar(
                value, i, xerr=np.array([[value - low], [high - value]]), color=colors[label], fmt="o", capsize=3
            )
        ax.set(title=name, xlabel="Mean accuracy (%)", yticks=[0, 1, 2], yticklabels=list(runs), ylim=(-0.5, 2.5))
        ax.invert_yaxis()
    save(
        fig,
        "evaluation-ci",
        "Update 2560: recorded 95% prompt-bootstrap intervals; not paired differences or seed intervals.",
        False,
    )

    fig, axes = figure(2, 3)
    for row, metric in enumerate(["pass@16", "majority@16"]):
        for ax, (ds, name) in zip(axes[row], DATASETS.items(), strict=True):
            for label, run in runs.items():
                ev = run["evaluation"]
                ax.plot(
                    [x["optimizer"] for x in ev],
                    [100 * x["metrics"][f"eval/{ds}/acc/{metric}"] for x in ev],
                    color=colors[label],
                    marker="o",
                    markersize=3.5,
                    lw=1.5,
                )
            ax.set(title=name, xlabel="Optimizer updates", ylabel=f"{metric} (%)", xticks=[512, 2560, 4096])
    save(
        fig,
        "sampling-outcomes",
        "Pass@16 measures any-correct coverage; majority@16 measures answer voting. Independent dataset/metric scales.",
    )

    fig, axes = figure()
    for label, run in runs.items():
        blocks = [run["reuse_windows"][f"{lo + 1}-{lo + 512}"] for lo in range(0, run["updates"] - 511, 512)]
        for ax, key, title in zip(
            axes[0][:2], ["first_kl", "last_preupdate_kl"], ["New-batch first KL", "Same-batch last KL"], strict=True
        ):
            ax.plot(
                [b["update_through"] for b in blocks],
                [b[key]["mean"] for b in blocks],
                color=colors[label],
                marker="o",
                markersize=3.5,
            )
            ax.set(
                title=title,
                yscale="log",
                xlabel="Optimizer updates",
                ylabel="KL (nats/token)",
                xticks=[512, 2560, 4096],
            )
        profile = run["reuse_windows"]["1025-2560"]["normalized_profile"]
        axes[0, 2].plot(np.linspace(0, 1, len(profile)), profile, color=colors[label], lw=1.6)
    axes[0, 2].set(
        title="Within-batch fitting",
        xlabel="Fraction of recorded reuse positions",
        ylabel="KL / first KL",
        ylim=(0, 1.05),
    )
    save(
        fig,
        "reuse",
        "Left/middle: 512-update windows. Right: updates 1025-2560, normalized per batch. All KL precedes updates.",
    )

    fig, axes = figure(cols=2)
    for label, run in runs.items():
        blocks = [run["reuse_windows"][f"{lo + 1}-{lo + 512}"] for lo in range(0, run["updates"] - 511, 512)]
        axes[0, 0].plot(
            [b["update_through"] for b in blocks],
            [b["within_drop"]["mean"] * 100 for b in blocks],
            color=colors[label],
            marker="o",
            markersize=3.5,
        )
        axes[0, 1].plot(
            [b["update_through"] for b in blocks],
            [b["last_transition_drop"]["mean"] * 100 for b in blocks],
            color=colors[label],
            marker="o",
            markersize=3.5,
        )
    axes[0, 0].set(title="First-to-last reduction", xlabel="Optimizer updates", ylabel="Within-batch KL reduction (%)")
    axes[0, 1].set(
        title="Last adjacent reduction", xlabel="Optimizer updates", ylabel="Last-transition KL reduction (%)"
    )
    save(
        fig,
        "reuse-marginal",
        "512-update windows; mean per-batch reductions. Different reuse counts span different update counts.",
    )

    fig, axes = figure()
    for label, run in runs.items():
        for ax, kind in zip(axes[0], ["length", "truncate", "score"], strict=True):
            x, y = [], []
            for end in range(512, run["updates"] + 1, 512):
                rows = [r for r in run["outer_series"] if end - 512 < r["step/optimizer"] <= end]
                count = sum(r["rollout/response_count"] for r in rows)
                value = (
                    sum(r["rollout/response_tokens"] for r in rows) / count
                    if kind == "length"
                    else sum(r["rollout/finish_reason/length_count"] for r in rows) / count * 100
                    if kind == "truncate"
                    else sum(r["rollout/score/mean"] * r["rollout/response_count"] for r in rows) / count * 100
                )
                x.append(end)
                y.append(value)
            ax.plot(x, y, color=colors[label], marker="o", markersize=3.5)
        for ax, title, ylabel in zip(
            axes[0],
            ["Training answer length", "Training length stops", "Training task score"],
            ["Mean response tokens", "Length-stopped responses (%)", "Mean task score (%)"],
            strict=True,
        ):
            ax.set(title=title, xlabel="Optimizer updates", ylabel=ylabel, xticks=[512, 2560, 4096])
    save(
        fig,
        "rollout",
        "Response-weighted 512-update windows; train cap 7168. Training prompt prefixes differ across runs.",
    )

    fig, axes = figure(2, 3)
    for column, (ds, name) in enumerate(DATASETS.items()):
        for label, run in runs.items():
            ev = run["evaluation"]
            x = [e["optimizer"] for e in ev]
            axes[0, column].plot(
                x,
                [e["metrics"][f"eval/{ds}/generation/response_length/mean"] for e in ev],
                color=colors[label],
                marker="o",
                markersize=3.5,
            )
            axes[1, column].plot(
                x,
                [
                    100
                    * e["metrics"][f"eval/{ds}/generation/finish_reason/length_count"]
                    / e["metrics"][f"eval/{ds}/response_count"]
                    for e in ev
                ],
                color=colors[label],
                marker="o",
                markersize=3.5,
            )
        axes[0, column].set(
            title=name, xlabel="Optimizer updates", ylabel="Mean validation tokens", xticks=[512, 2560, 4096]
        )
        axes[1, column].set(
            title=name, xlabel="Optimizer updates", ylabel="Validation length stops (%)", xticks=[512, 2560, 4096]
        )
    save(
        fig,
        "validation-generation",
        "Validation cap 31744, 16 responses per prompt. No missing evaluations are filled; independent dataset scales.",
    )

    fig, axes = figure()
    for label, run in runs.items():
        rows = run["outer_series"]
        x = [r["step/optimizer"] for r in rows]
        total = (
            np.cumsum([r["perf/timing/outer_step_seconds"] + r.get("perf/timing/validation_seconds", 0) for r in rows])
            / 3600
        )
        actor = np.cumsum([r["perf/timing/update_actor_seconds"] for r in rows]) / 3600
        tokens = np.cumsum([r["rollout/response_tokens"] for r in rows]) / 1e6
        for ax, y in zip(axes[0], [total, actor, tokens], strict=True):
            ax.plot(x, y, color=colors[label], lw=1.6)
    for ax, title, ylabel in zip(
        axes[0],
        ["Recorded training + validation", "Actor stage", "New rollout work"],
        ["Cumulative hours", "Cumulative actor hours", "New response tokens (million)"],
        strict=True,
    ):
        ax.set(title=title, xlabel="Optimizer updates", ylabel=ylabel)
    save(
        fig,
        "cost",
        "Timers exclude startup/logging overhead; actor is inside training. Hardware differs (A800 / A100).",
    )


if __name__ == "__main__":
    main()
