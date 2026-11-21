# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Standalone report and audited figures, including honest pending-state status.

Ledger: fixed-bank curves map optimizer update -> question-mean reverse KL;
line colour identifies the pair, line style identifies the response bank.
State maps use one coordinate scale per pair; one dot is one saved-prefix state.
"""

import base64
import json
from pathlib import Path

from examples.mirroropd.research.core.io import write_text
from examples.mirroropd.research.core.plotting import load_style, save_figure

from .report_data import payload


def plot_risk(rows, path):
    import matplotlib.pyplot as plt

    fv = load_style()
    fv.apply_rc("paper", font="sans")
    pairs = list(dict.fromkeys(r["pair"] for r in rows))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, bank in zip(axes, ("public_initial", "public_teacher"), strict=True):
        for pair, color in zip(pairs, fv.series_colors("paper", len(pairs)), strict=True):
            data = [r for r in rows if r["pair"] == pair and r["bank"] == bank and r["domain"] == "math"]
            ax.plot([r["steps"] for r in data], [r["rkl"] for r in data], color=color, marker="o", label=pair)
        ax.set(title=bank.replace("public_", "") + " response bank", xlabel="Optimizer updates", yscale="log")
        ax.set_xticks([0, 160, 320, 480, 640])
        ax.legend(frameon=False)
        ax.set_ylabel("Math reverse KL (nats / token; question mean)")
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.24, top=0.88, wspace=0.3)
    save_figure(fig, path, "Existing fixed response banks; log y scale; no new response generation")


def plot_map(points, groups, path):
    import matplotlib.pyplot as plt
    import numpy as np

    fv = load_style()
    fv.apply_rc("paper", font="sans")
    xy = np.array([[p["x"], p["y"]] for p in points])
    names = np.array([p["group"] for p in points])
    colors = [fv.NEUTRALS["baseline"], *fv.series_colors("paper", 4), fv.NEUTRALS["vink"]]
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    for ax, group, color in zip(axes.flat, groups, colors, strict=True):
        ax.scatter(*xy.T, s=1, color=fv.NEUTRALS["vmist"], rasterized=True, label="All sources")
        selected = xy[names == group]
        ax.scatter(*selected.T, s=3, color=color, rasterized=True, label=group)
        ax.set_title(group, fontsize=11)
        ax.set_xlabel("t-SNE 1", fontsize=9)
        ax.set_ylabel("t-SNE 2", fontsize=9)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    legend = {}
    for ax in axes.flat:
        handles, labels = ax.get_legend_handles_labels()
        legend.update(zip(labels, handles, strict=True))
    fig.legend(legend.values(), legend.keys(), loc="upper center", ncol=7, frameon=False, fontsize=8)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.90, bottom=0.13, hspace=0.35, wspace=0.15)
    save_figure(
        fig, path, "t-SNE display only; common equal-scale coordinates; grey = all sources; includes prompt-end states"
    )


def report(plan_path, *, output):
    """Render into the owning experiment topic; figures live in its figures directory."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = payload(plan_path)
    figures = output.parent / "figures"
    figures.mkdir(exist_ok=True)
    prefix = output.stem
    risk = figures / f"{prefix}-rkl"
    plot_risk(data["risks"], risk)
    data["risk_figure"] = "data:image/png;base64," + base64.b64encode(risk.with_suffix(".png").read_bytes()).decode()
    for pair, spec in data["pairs"].items():
        plot_map(spec["points"], data["groups"], figures / f"{prefix}-{pair}-map")
    assets = Path(__file__).parent / "viz"
    template = (assets / "report.html").read_text()
    styles = "\n".join((assets / name).read_text() for name in ("report.css", "scatter.css", "overview.css"))
    scripts = "\n".join(
        (assets / name).read_text()
        for name in ("scatter.js", "trajectory.js", "occupancy.js", "overview.js", "report.js")
    )
    serialized = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    page = template.replace("<!--STYLE-->", f"<style>{styles}</style>")
    page = page.replace("<!--DATA-->", f'<script type="application/json" id="report-data">{serialized}</script>')
    page = page.replace("<!--SCRIPT-->", f"<script>{scripts}</script>")
    write_text(output, page)
