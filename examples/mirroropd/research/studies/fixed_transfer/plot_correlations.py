# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Plot fixed-state endpoint correlations without training or reading model weights.

Semantic ledger: one point is one execution's first-pass endpoint; teacher color
is stable within each student/input group, shape denotes update budget, and fill
denotes same versus different model size. All risks use prompts times H1024.
Geometry ledger: linear axes show decreases or log10 absorption factors; final
risks use log-log axes, with correlations computed on log10 risks. Shared limits
within a figure make the equality reference comparable across panels.
No jitter, interpolation, regression, or repeated checkpoint observations.
"""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from examples.mirroropd.research.core.paths import REPO
from examples.mirroropd.research.core.plotting import load_style
from examples.mirroropd.research.studies.fixed_transfer.overview import MODEL_LABELS

BASE = REPO / "examples/mirroropd/docs/experiments/teacher-transfer"
SOURCE = BASE / "figures/fixed-state-clean-20261107/endpoints.csv"
OUTPUT = BASE / "fixed-state-results/figures/correlations"
PANELS = ["Training scored", "Math", "Code", "IF", "All held-out"]
STUDENTS = {
    "r1": "R1-1.5B",
    "r7": "R1-7B",
    "q3-1.7": "Qwen3-1.7B / thinking",
    "q3-1.7-nonthinking": "Qwen3-1.7B / non-thinking",
    "llama3-3b-off": "Llama-3.2-3B-Instruct",
    "q3-1.7-native-thinking2507": "Qwen3-1.7B / native Thinking-2507",
    "q3-1.7-native-instruct2507": "Qwen3-1.7B / native Instruct-2507",
}


def load_points(source):
    runs = defaultdict(dict)
    with source.open() as stream:
        for row in csv.DictReader(stream):
            if row["band"] != "all":
                continue
            panel = row["panel"]
            run = runs[row["run_id"]]
            if panel in run:
                raise ValueError(f"Duplicate endpoint: {row['run_id']}/{panel}")
            run[panel] = row
    points = []
    for run_id, panels in runs.items():
        if set(panels) != set(PANELS):
            raise ValueError(f"Incomplete panels: {run_id}")
        identities = {(r["student"], r["teacher"], r["step"], r["same_size"]) for r in panels.values()}
        if len(identities) != 1:
            raise ValueError(f"Mixed endpoint identities: {run_id}")
        student, teacher, step, same_size = identities.pop()
        if int(step) not in (64, 320):
            raise ValueError(f"Unexpected endpoint budget: {run_id}/{step}")
        label_key = "coreward-i-llama" if student == "llama3-3b-off" and teacher == "coreward-i" else teacher
        point = dict(
            run_id=run_id,
            student=student,
            teacher=teacher,
            teacher_label=MODEL_LABELS[label_key][0],
            step=int(step),
            same_size=same_size == "True",
        )
        for panel, row in panels.items():
            initial, final = float(row["fixed_h_initial"]), float(row["fixed_h_final"])
            if initial <= 0 or final <= 0 or not np.isfinite([initial, final]).all():
                raise ValueError(f"Invalid risk: {run_id}/{panel}")
            point[panel] = dict(
                initial=initial,
                final=final,
                absolute=initial - final,
                relative=100 * (1 - final / initial),
                log_absorption=float(np.log10(initial) - np.log10(final)),
            )
        for field in ("initial", "final"):
            pooled = sum(point[p][field] * w for p, w in [("Math", 0.5), ("Code", 0.25), ("IF", 0.25)])
            if not np.isclose(pooled, point["All held-out"][field], rtol=1e-10, atol=1e-12):
                raise ValueError(f"Held-out pooling mismatch: {run_id}/{field}")
        points.append(point)
    return points


def ranks(values):
    """Average ranks for ties; Pearson of these ranks is Spearman rho."""
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    average = np.cumsum(counts) - (counts - 1) / 2
    return average[inverse]


def correlations(x, y):
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None, None
    return float(np.corrcoef(x, y)[0, 1]), float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render(points, student, metric, kind, out, fv):
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from matplotlib.lines import Line2D

    pairs = (
        [("Training scored", p) for p in PANELS[1:]]
        if kind == "cross-batch"
        else [("Math", "Code"), ("Math", "IF"), ("Code", "IF")]
    )
    final_risk = metric == "final"
    units = {
        "relative": "Relative RKL decrease (%)",
        "absolute": "Absolute RKL decrease (nats / H1024)",
        "log_absorption": "Absorption: log10(initial RKL / final RKL)",
        "final": "Final RKL (log-log axes; correlations on log10 RKL)",
    }
    suffix = {
        "relative": " decrease (%)",
        "absolute": " decrease (nats)",
        "log_absorption": " absorption (orders of magnitude)",
        "final": " final RKL (nats)",
    }[metric]
    reference = (
        "Equal final RKL" if final_risk else "Equal absorption" if metric == "log_absorption" else "Equal decrease"
    )
    fig, grid = plt.subplots(2, 2, figsize=(14, 11))
    axes = list(grid.flat)
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.33, top=0.84, hspace=0.65, wspace=0.32)
    title = "Training pool vs. held-out" if kind == "cross-batch" else "Held-out domain vs. domain"
    fig.suptitle(f"{STUDENTS[student]} | {title}\n{units[metric]}", y=0.97)
    values = [p[panel][metric] for p in points for panel in set(sum((list(pair) for pair in pairs), []))]
    if final_risk:
        values = np.log10(values)
    lower, upper = min(values) if final_risk else min(0, min(values)), max(values)
    span = max(upper - lower, 1e-5)
    limits = lower - 0.08 * span, upper + 0.10 * span
    if final_risk:
        limits = tuple(10**v for v in limits)
    colors = []
    # Four hue families; brightness separates further teacher identities.
    for i in range(len(points)):
        base = np.array(to_rgb(fv.HUES[["vteal", "vorange", "vcoral", "vviolet"][i % 4]]))
        colors.append(tuple(base * (1 - 0.17 * (i // 4))))
    fv.register_colors(*colors)
    statistics = []
    for index, ((xp, yp), ax) in enumerate(zip(pairs, axes, strict=False)):
        x, y = [[p[panel][metric] for p in points] for panel in (xp, yp)]
        pearson, spearman = correlations(np.log10(x), np.log10(y)) if final_risk else correlations(x, y)
        stats = (
            f"n={len(points)}; r={pearson:.2f}; rho={spearman:.2f}"
            if pearson is not None
            else f"n={len(points)}; correlation unavailable"
        )
        xname, yname = ("Training" if xp == "Training scored" else xp), ("Avg" if yp == "All held-out" else yp)
        ax.set(
            title=f"({chr(97 + index)}) {xname} vs. {yname}\n{stats}",
            xlabel=xname + suffix,
            ylabel=yname + suffix,
            xlim=limits,
            ylim=limits,
        )
        if final_risk:
            ax.set(xscale="log", yscale="log")
        ax.plot(limits, limits, color=fv.NEUTRALS["baseline"], linestyle="--", linewidth=0.8, label=reference)
        if not final_risk:
            ax.axhline(0, color=fv.NEUTRALS["vmist"], linewidth=0.6)
            ax.axvline(0, color=fv.NEUTRALS["vmist"], linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        for p, color in zip(points, colors, strict=True):
            ax.scatter(
                p[xp][metric],
                p[yp][metric],
                s=68,
                marker="^" if p["step"] == 64 else "o",
                facecolor=color if p["same_size"] else "none",
                edgecolor=color,
                linewidth=1.5,
                label=f"{p['teacher_label']} [{p['step']}]",
                zorder=3,
            )
        statistics.append(
            dict(
                student=student,
                kind=kind,
                metric=metric,
                correlation_space="log10 RKL" if final_risk else "displayed values",
                x_panel=xp,
                y_panel=yp,
                n=len(points),
                n64=sum(p["step"] == 64 for p in points),
                n320=sum(p["step"] == 320 for p in points),
                pearson_r=pearson,
                spearman_rho=spearman,
            )
        )
    for ax in axes[len(pairs) :]:
        ax.set_visible(False)
    handles = [
        Line2D(
            [],
            [],
            color=color,
            marker="^" if p["step"] == 64 else "o",
            markerfacecolor=color if p["same_size"] else "none",
            linestyle="none",
            markersize=7,
            label=f"{p['teacher_label']} [{p['step']}]",
        )
        for p, color in zip(points, colors, strict=True)
    ]
    handles.append(Line2D([], [], color=fv.NEUTRALS["baseline"], linestyle="--", label=reference))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.065), ncol=2, frameon=False, fontsize=9)
    fig.text(
        0.5,
        0.035,
        "One point = one run endpoint; triangle: 64 updates; circle: 320; filled: same size; open: cross size.",
        ha="center",
        fontsize=9,
        color=fv.NEUTRALS["baseline"],
    )
    fig.text(
        0.5,
        0.012,
        "Fixed-H RKL; dashed: equality; correlations pool all displayed endpoints (mixed budgets/sizes).",
        ha="center",
        fontsize=9,
        color=fv.NEUTRALS["baseline"],
    )
    fig.canvas.draw()
    issues = fv.audit(fig, print_slot=fig.get_figwidth())
    stem = out / f"{student}-{kind}-{metric}"
    Path(str(stem) + ".audit.json").write_text(json.dumps([str(p) for p in issues], indent=2) + "\n")
    if any(fv.problem_severity(p) == "error" for p in issues):
        raise RuntimeError(f"{stem.name}: {issues}")
    for extension in ("png", "pdf"):
        fig.savefig(Path(str(stem) + f".{extension}"), dpi=160)
    plt.close(fig)
    return statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    fv = load_style()
    fv.apply_rc("paper", fontsize=11)
    points = load_points(args.source)
    args.output.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    rows = []
    for p in points:
        grouped[p["student"]].append(p)
        for panel in PANELS:
            rows.append({k: v for k, v in p.items() if k not in PANELS} | dict(panel=panel) | p[panel])
    stats = []
    for student in STUDENTS:
        for kind in ("cross-batch", "cross-domain"):
            metrics = (
                ("relative", "absolute", "log_absorption", "final")
                if kind == "cross-batch"
                else ("relative", "absolute")
            )
            for metric in metrics:
                stats.extend(
                    render(
                        sorted(grouped[student], key=lambda p: p["teacher_label"]),
                        student,
                        metric,
                        kind,
                        args.output,
                        fv,
                    )
                )
    write_csv(args.output / "points.csv", rows)
    write_csv(args.output / "correlations.csv", stats)
    provenance = dict(
        source=str(args.source.relative_to(REPO)),
        sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        runs=len(points),
        figures=len(STUDENTS) * 6,
        final_risk_correlation_space="log10 RKL; nonpositive risk rejected, no epsilon floor",
        normalization="sum KL / (prompts * 1024)",
        selection="All band=all endpoint rows in clean diagnostics; one endpoint per run; no extra filtering",
        budgets=sorted({p["step"] for p in points}),
    )
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
