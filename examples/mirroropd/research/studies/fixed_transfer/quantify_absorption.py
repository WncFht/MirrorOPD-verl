# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Descriptive trajectory diagnostics; dependent checkpoints are not replicates.

Color identifies teacher; circle/diamond encodes same/cross size. Solid lines
connect chronological snapshots only. Ratios require >0.01 dex train progress;
this is a reporting stability guard, not a scientific absorption threshold.
Early window is 0:64; late window is end-64:end, only when end >= 128.
"""

import csv
import hashlib
import json
from pathlib import Path

from examples.mirroropd.research.studies.fixed_transfer.explorer_schema import METRICS, teacher_rgb
from examples.mirroropd.research.studies.fixed_transfer.plot_correlations import STUDENTS

BASE = Path(__file__).resolve().parents[3] / "docs/experiments/teacher-transfer/fixed-state-results"
SOURCE = BASE / "figures/student-paths"
OUT = BASE / "figures/absorption-diagnostics"
PANELS = ["Math", "Code", "IF", "All held-out"]
MIN_PROGRESS = 0.01


def ratio(numerator, denominator):
    return numerator / denominator if denominator > MIN_PROGRESS else None


def ratio_status(denominator):
    if denominator is None:
        return "insufficient_budget"
    if denominator <= 0:
        return "train_rebound"
    return "eligible" if denominator > MIN_PROGRESS else "small_train_progress"


def diagnose(run, panel, end):
    train = {p["step"]: p for p in run["panels"]["Training scored"]}
    held = {p["step"]: p for p in run["panels"][panel]}
    x, y = train[end]["log_absorption"], held[end]["log_absorption"]
    early_x = train[64]["log_absorption"] if end >= 64 else None
    early_y = held[64]["log_absorption"] if end >= 64 else None
    late_x = x - train[end - 64]["log_absorption"] if end >= 128 else None
    late_y = y - held[end - 64]["log_absorption"] if end >= 128 else None
    return dict(
        step=end,
        panel=panel,
        train_absorption=x,
        held_absorption=y,
        gap=x - y,
        efficiency=ratio(y, x),
        efficiency_status=ratio_status(x),
        early_train=early_x,
        early_held=early_y,
        early_efficiency=ratio(early_y, early_x) if end >= 64 else None,
        early_status=ratio_status(early_x),
        late_train=late_x,
        late_held=late_y,
        late_efficiency=ratio(late_y, late_x) if end >= 128 else None,
        late_status=ratio_status(late_x),
        train_initial=train[0]["rkl"],
        held_initial=held[0]["rkl"],
        train_rkl=train[end]["rkl"],
        held_rkl=held[end]["rkl"],
        train_prompts=train[end]["prompts"],
    )


def load_runs():
    with (SOURCE / "inventory.csv").open() as stream:
        inventory = list(csv.DictReader(stream))
    runs = {r["run_id"]: r for r in inventory}
    if len(runs) != len(inventory):
        raise ValueError("Duplicate run IDs in inventory")
    for run in runs.values():
        run["same_size"] = run["same_size"] == "True"
        run["budget"] = int(run["budget"])
        run["panels"] = {p: [] for p in ["Training scored", *PANELS]}
    with (SOURCE / "snapshots.csv").open() as stream:
        for row in csv.DictReader(stream):
            p = {k: float(row[k]) for k in ["rkl", "log_absorption", "initial", "relative", "absolute"]}
            p.update(step=int(row["step"]), prompts=int(row["prompts"]))
            runs[row["run_id"]]["panels"][row["panel"]].append(p)
    for run in runs.values():
        for points in run["panels"].values():
            points.sort(key=lambda p: p["step"])
            if [p["step"] for p in points] != list(range(0, run["budget"] + 1, 16)):
                raise ValueError(f"Invalid grid: {run['run_id']}")
    return list(runs.values())


def render(runs):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from examples.mirroropd.research.core.plotting import load_style, save_figure

    fv = load_style()
    fv.apply_rc("paper", fontsize=10)
    for student, title in STUDENTS.items():
        chosen = sorted([r for r in runs if r["student"] == student], key=lambda r: r["label"])
        colors = [teacher_rgb(i) for i in range(len(chosen))]
        fv.register_colors(*colors)
        for view in ["absorption", "efficiency", "windows", "drift"]:
            fig, axes = plt.subplots(2, 2, figsize=(13, 10))
            for ax, panel in zip(axes.flat, PANELS, strict=True):
                skipped = []
                for i, (run, color) in enumerate(zip(chosen, colors, strict=True)):
                    q = diagnose(run, panel, run["budget"])
                    marker = "o" if run["same_size"] else "D"
                    label = f"{i + 1}: {run['teacher']} [{run['budget']}]"
                    if view == "absorption":
                        x, y = q["train_absorption"], q["held_absorption"]
                    elif view == "efficiency":
                        x, y = i + 1, q["efficiency"]
                    elif view == "windows":
                        x, y = q["early_efficiency"], q["late_efficiency"]
                    else:
                        qs = [diagnose(run, panel, p["step"]) for p in run["panels"][panel]]
                        ax.plot([p["step"] for p in qs], [p["gap"] for p in qs], color=color, label=label)
                        ax.scatter(q["step"], q["gap"], color=color, marker=marker, s=35)
                        continue
                    if x is None or y is None:
                        skipped.append(run["teacher"])
                        continue
                    ax.scatter(x, y, color=color, marker=marker, s=42, label=label)
                ax.set_title("Avg" if panel == "All held-out" else panel)
                ax.grid(alpha=0.18)
                if view == "absorption":
                    lo, hi = min(0, *ax.get_xlim(), *ax.get_ylim()), max(*ax.get_xlim(), *ax.get_ylim())
                    ax.axline((0, 0), slope=1, color=fv.NEUTRALS["baseline"], linewidth=0.7, label="A_D = A_T")
                    ax.set(
                        xlim=(lo, hi),
                        ylim=(lo, hi),
                        xlabel="Training absorption $A_T$ (dex)",
                        ylabel="Held-out absorption $A_D$ (dex)",
                    )
                elif view == "efficiency":
                    ax.axhline(1, color=fv.NEUTRALS["baseline"], linewidth=0.7)
                    ax.set(xlabel="Teacher index (legend order)", ylabel=r"Efficiency $\eta_D=A_D/A_T$")
                    ax.set_xticks(range(1, len(chosen) + 1))
                elif view == "windows":
                    ax.axhline(1, color=fv.NEUTRALS["baseline"], linewidth=0.7)
                    ax.axvline(1, color=fv.NEUTRALS["baseline"], linewidth=0.7)
                    ax.set(xlabel=r"Early $\eta_D[0,64]$", ylabel=r"Late $\eta_D[e-64,e]$")
                else:
                    ax.axhline(0, color=fv.NEUTRALS["baseline"], linewidth=0.7)
                    ax.set(xlabel="Optimizer updates $t$", ylabel=r"Gap $\Delta_D=A_T-A_D$ (dex)")
                if skipped:
                    ax.set_title(ax.get_title() + f"; {len(skipped)} undefined ratios")
            # Dedicated figure legend preserves even teachers whose ratios are undefined.
            from matplotlib.lines import Line2D

            handles = [
                Line2D(
                    [],
                    [],
                    marker="o" if r["same_size"] else "D",
                    linestyle="None",
                    color=c,
                    label=f"{i + 1}: {r['teacher']} [{r['budget']}]",
                )
                for i, (r, c) in enumerate(zip(chosen, colors, strict=True))
            ]
            if view == "absorption":
                handles.append(Line2D([], [], color=fv.NEUTRALS["baseline"], linewidth=0.7, label="A_D = A_T"))
            ylo = min(a.get_ylim()[0] for a in axes.flat)
            yhi = max(a.get_ylim()[1] for a in axes.flat)
            for ax in axes.flat:
                ax.set_ylim(ylo, yhi)
            if view in {"absorption", "windows"}:
                xlo = min(a.get_xlim()[0] for a in axes.flat)
                xhi = max(a.get_xlim()[1] for a in axes.flat)
                if view == "absorption":
                    xlo = ylo = min(xlo, ylo)
                    xhi = yhi = max(xhi, yhi)
                for ax in axes.flat:
                    ax.set_xlim(xlo, xhi)
                    ax.set_ylim(ylo, yhi)
            fig.suptitle(title + " / " + view, y=0.98)
            fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.045), ncol=4, fontsize=8)
            fig.subplots_adjust(left=0.09, right=0.97, bottom=0.24, top=0.92, wspace=0.27, hspace=0.36)
            save_figure(
                fig,
                OUT / f"{student}-{view}.figure",
                "Circle: same size; diamond: cross size. Actual endpoints. Ratio requires >0.01 dex train progress.",
            )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    rows = []
    for run in runs:
        for panel in PANELS:
            for p in run["panels"][panel]:
                rows.append(
                    {k: run[k] for k in ["run_id", "student", "teacher", "same_size", "added_stage", "budget"]}
                    | diagnose(run, panel, p["step"])
                )
    with (OUT / "metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    render(runs)
    receipt = dict(
        source_hashes={
            n: hashlib.sha256((SOURCE / n).read_bytes()).hexdigest() for n in ["inventory.csv", "snapshots.csv"]
        },
        min_progress_dex=MIN_PROGRESS,
        runs=len(runs),
        rows=len(rows),
        early_window=[0, 64],
        late_window="[end-64,end], end>=128",
        metric_definitions=METRICS,
    )
    (OUT / "provenance.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"{len(runs)} runs; {len(rows)} derived rows; 28 figures")


if __name__ == "__main__":
    main()
