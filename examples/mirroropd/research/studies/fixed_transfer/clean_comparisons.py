# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Comparison ledger: one curve = one teacher on the same student/input bank.

Heatmap rows are named teachers, columns are actual saved optimizer updates.
Unavailable budgets are masked, never zeros. Focal comparisons share the same
320-step budget and separate all permanent banks. Teacher keys have stable colors
within a focal comparison. Initial/final position profiles use dashed/solid lines.
No RL/SFT algorithmic causal effect is identified by comparing public checkpoints.
"""

import csv

import numpy as np

from examples.mirroropd.research.core.io import dump, write_csv, write_text
from examples.mirroropd.research.studies.fixed_transfer.clean_diagnostics import BANDS, PANEL_LABELS, CleanRenderer


def read_metrics(output, result):
    rows = list(csv.DictReader((output / result["spec"]["run_id"] / "metrics.csv").open()))
    for row in rows:
        row["step"] = int(row["step"])
        for key in ["observed", "fixed_h", "decrease_fixed_h", "remaining", "coverage"]:
            row[key] = float(row[key]) if row[key] else None
    return rows


def renderer(result, output):
    data = dict(spec=result["spec"], panels=result["panels"])
    return CleanRenderer(data, output, [])


def series(rows, panel, metric="rkl", band="all", column="observed"):
    return [
        (r["step"], r[column])
        for r in rows
        if r["panel"] == panel and r["metric"] == metric and r["band"] == band and r[column] is not None
    ]


def comparison_heatmaps(results, rows, output):
    entries = []
    for student in sorted({r["spec"]["student"] for r in results}):
        subsets = [r for r in results if r["spec"]["student"] == student]
        for same, scope in [(True, "same-size"), (False, "cross-size-context")]:
            chosen = sorted([r for r in subsets if r["same_size"] == same], key=lambda r: r["spec"]["teacher"])
            if not chosen:
                continue
            render = renderer(chosen[0], output)
            render.title = f"{student}: {scope}; teacher keys label rows"
            for column, label in [("fixed_h", "Absolute RKL / H1024"), ("remaining", "RKL / initial")]:
                fig, axes = render.figure(label, len(render.panels), 2)
                for ax, panel in zip(axes, render.panels, strict=True):
                    ax.set(title=PANEL_LABELS[panel], xlabel="Optimizer updates")
                steps = sorted({s for r in chosen for s in r["snapshots"]})
                matrices = []
                for panel in render.panels:
                    matrix = []
                    for result in chosen:
                        values = dict(series(rows[result["spec"]["run_id"]], panel, column=column))
                        matrix.append([values.get(s, np.nan) for s in steps])
                    matrices.append(np.array(matrix))
                maximum = max(float(np.nanmax(m)) for m in matrices)
                edges = np.array([*steps, steps[-1] + 16]) - 8
                cmap = render.fv.ramp_cmap("seq")
                cmap.set_bad(render.fv.NEUTRALS["baseline"], alpha=0.15)
                labels = [f"{r['spec']['teacher']} ({r['spec']['budget']})" for r in chosen]
                for ax, matrix in zip(axes, matrices, strict=True):
                    im = ax.pcolormesh(
                        edges,
                        np.arange(len(chosen) + 1) - 0.5,
                        np.ma.masked_invalid(matrix),
                        cmap=cmap,
                        vmin=0,
                        vmax=maximum,
                    )
                    ax.set(yticks=range(len(chosen)), yticklabels=labels, ylabel="Teacher key (last update)")
                    ax.invert_yaxis()
                    fig.colorbar(im, ax=ax, fraction=0.055, pad=0.04).set_label(label)
                fig.subplots_adjust(left=0.19, wspace=0.95)
                key = f"{student}-{scope}-{'absolute' if column == 'fixed_h' else 'remaining'}"
                render.finish(
                    fig,
                    key,
                    label,
                    "One student/input mode per figure. Common color scale across banks; "
                    "gray cells have no saved observation. Teacher labels resolve in runs.csv.",
                )
                entries.append(dict(key=key, title=f"{student} / {scope} / {label}"))
    return entries


def focal_comparisons(results, rows, output):
    entries = []
    groups = {
        "q3-1.7": ["grpo-kl", "e3", "sft-gsm8k", "sft-deepmath"],
        "llama3-3b-off": ["gt", "coreward-i", "arithmetic-full"],
        "r1": ["deepscaler", "justrl", "nemotron", "brorl"],
    }
    for student, teachers in groups.items():
        chosen = [
            r
            for t in teachers
            for r in results
            if r["spec"]["student"] == student and r["spec"]["teacher"] == t and r["spec"]["budget"] == 320
        ]
        if len(chosen) < 2:
            continue
        render = renderer(chosen[0], output)
        render.title = f"{student}: same-size checkpoint comparison, 320 updates"
        for column, key, ylabel, log in [
            ("fixed_h", "absolute", "RKL / (prompts x 1024)", False),
            ("fixed_h", "log", "RKL / (prompts x 1024), log", True),
            ("decrease_fixed_h", "decrease", "Initial RKL - current RKL", False),
            ("remaining", "remaining", "RKL / initial RKL", False),
            ("observed", "entropy-gap", "Student - teacher entropy (nats)", False),
            ("observed", "top16", "Top-16 intersection / 16", False),
        ]:
            metric = "overlap16" if key == "top16" else "hp" if key == "entropy-gap" else "rkl"
            if not all(series(rows[r["spec"]["run_id"]], "All held-out", metric) for r in chosen):
                continue
            fig, axes = render.axes(ylabel)
            for ax, panel in zip(axes, render.panels, strict=True):
                for result, color in zip(chosen, render.colors, strict=False):
                    values = rows[result["spec"]["run_id"]]
                    points = series(values, panel, metric, column=column)
                    if key == "entropy-gap":
                        teacher = dict(series(values, panel, "hq"))
                        points = [(x, y - teacher[x]) for x, y in points]
                    ax.plot(*zip(*points, strict=True), color=color, label=result["spec"]["teacher"])
                ax.set_ylabel(ylabel)
                if log:
                    ax.set_yscale("log")
                elif key in ["decrease", "entropy-gap"]:
                    ax.axhline(0, color=render.ink, linewidth=0.5)
                else:
                    ax.set_ylim(bottom=0)
            render.legend(fig, axes[0])
            stem = f"{student}-focal-{key}"
            render.finish(
                fig,
                stem,
                ylabel,
                "Same student, input mode and 320-update protocol; public checkpoint history and initial KL differ. "
                "These are checkpoint comparisons, not controlled RL vs SFT interventions.",
            )
            entries.append(dict(key=stem, title=f"{student} / {ylabel}"))
        for metric, name in [
            ("rkl", "RKL"),
            ("hp", "Student entropy"),
            ("hq", "Teacher entropy"),
            ("overlap16", "Top-16 overlap"),
        ]:
            if not all(series(rows[r["spec"]["run_id"]], "All held-out", metric) for r in chosen):
                continue
            fig, axes = render.axes(f"{name}: initial and final position profiles")
            for ax, panel in zip(axes, render.panels, strict=True):
                for result, color in zip(chosen, render.colors, strict=False):
                    values = rows[result["spec"]["run_id"]]
                    for step, style in [(0, "--"), (320, "-")]:
                        ys = [dict(series(values, panel, metric, b))[step] for b in BANDS]
                        ax.plot(
                            range(3),
                            ys,
                            color=color,
                            linestyle=style,
                            marker="o",
                            markersize=3,
                            label=result["spec"]["teacher"] if step == 320 else None,
                        )
                ax.set(
                    xticks=range(3),
                    xticklabels=BANDS,
                    xlabel="Absolute position band",
                    ylabel=f"{name} / observed token",
                    ylim=(0, None),
                )
            render.legend(fig, axes[0])
            stem = f"{student}-focal-position-{metric}"
            render.finish(
                fig,
                stem,
                name,
                "Dashed: update 0; solid: update 320. early=0-255, middle=256-511, late=512-1023; "
                "categorical spacing, unequal token spans.",
            )
            entries.append(dict(key=stem, title=f"{student} / position {name}"))
    return entries


def comparisons(results, output):
    target = output / "comparisons"
    target.mkdir(exist_ok=True)
    rows = {r["spec"]["run_id"]: read_metrics(output, r) for r in results}
    entries = comparison_heatmaps(results, rows, target) + focal_comparisons(results, rows, target)
    endpoints, trajectory = [], []
    for result in results:
        spec = result["spec"]
        values = rows[spec["run_id"]]
        trajectory += [r for r in values if r["band"] == "all"]
        for panel in result["panels"]:
            for band in ["all", *BANDS]:
                rkl = [r for r in values if r["panel"] == panel and r["band"] == band and r["metric"] == "rkl"]
                if not rkl:
                    continue
                first, last = rkl[0], rkl[-1]
                hp = dict(series(values, panel, "hp", band))
                hq = dict(series(values, panel, "hq", band))
                overlap = dict(series(values, panel, "overlap16", band))
                endpoints.append(
                    dict(
                        run_id=spec["run_id"],
                        student=spec["student"],
                        teacher=spec["teacher"],
                        same_size=result["same_size"],
                        step=spec["budget"],
                        panel=panel,
                        band=band,
                        initial=first["observed"],
                        final=last["observed"],
                        fixed_h_initial=first["fixed_h"],
                        fixed_h_final=last["fixed_h"],
                        decrease=first["observed"] - last["observed"],
                        remaining=last["remaining"],
                        coverage=last["coverage"],
                        hp_initial=hp[0],
                        hp_final=hp[spec["budget"]],
                        hq=hq[0],
                        overlap_initial=overlap.get(0),
                        overlap_final=overlap.get(spec["budget"]),
                    )
                )
    write_csv(target / "endpoints.csv", endpoints)
    write_csv(target / "all-panel-trajectories.csv", trajectory)
    dump(target / "figures.json", entries)
    lines = [
        "# 跨 teacher 比较",
        "",
        f"{len(entries)} 张比较图。热图按 student／输入模式及同／跨尺寸拆开；跨尺寸只作上下文。"
        "重点折线限相同 student 与 320 更新，公开 checkpoint 的前史与初始 KL 没有控制为相同。",
        "",
        "[返回完整图集](../README.md) · [端点 CSV](endpoints.csv) · [轨迹 CSV](all-panel-trajectories.csv)",
        "",
    ]
    for entry in entries:
        lines += [
            f"## {entry['title']}",
            "",
            f"![{entry['title']}]({entry['key']}.png)",
            "",
            f"[PDF]({entry['key']}.pdf) · [审计]({entry['key']}.audit.json)",
            "",
        ]
    write_text(target / "README.md", "\n".join(lines))
