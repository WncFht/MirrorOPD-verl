"""Replot frozen full-layer evidence.

Ledger: each line point is the median of seven projection matrices at one real
layer. Coral=teacher update, teal=OPD endpoint update. Geometry lines each denote
one reconstruction constraint. Scatter points are conditions, not independent
training seeds; repeated Samsung modes share teacher weights. All axes express
measured quantities, not inferred causality. Figures consume only adjacent data.
"""

import gzip
import json
import os
import sys
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.environ["FIGURE_VIZ_MPL"])
import fvstyle as fv

ROOT = Path(__file__).resolve().parent
DATA = json.loads(gzip.decompress((ROOT / "layer-data.json.gz").read_bytes()))
OUT = ROOT.parent.parent / "figures/layer-review"
OUT.mkdir(exist_ok=True)
fv.apply_rc("paper")
COLORS = fv.series_colors("paper", 4)
NAMES = {
    0: "R1 DeepCoder",
    1: "R1 DeepScaleR",
    2: "R1 JustRL",
    3: "R1 MiniMath SFT",
    4: "R1 BroRL",
    5: "R1 Nemotron",
    6: "R1-7B Light SFT",
    7: "Qwen GRPO",
    8: "Qwen SFT thinking",
    9: "Qwen e3",
    10: "Qwen SFT nonthinking",
    11: "Llama CoReward-I",
    12: "Llama GT",
    13: "Llama arithmetic SFT",
    19: "MiMo SFT to RL",
    20: "MiMo branch",
}
# Standalone wide-screen figures retain their declared 11-13 inch width in PDF;
# the generic 5.5 inch print-slot downscale does not apply to these deliverables.
audits = {}


def save(fig, name):
    fig.canvas.draw()
    audits[name] = [str(p) for p in fv.audit(fig, ignore=["print-fit"])]
    fig.savefig(OUT / (name + ".png"), dpi=180)
    fig.savefig(OUT / (name + ".pdf"))
    plt.close(fig)


def line_values(job, stage, field):
    rows = [x for x in DATA["matrices"] if x["job_id"] == job and x["stage"] == stage]
    layers = sorted({x["layer"] for x in rows})
    return layers, [median(x[field] for x in rows if x["layer"] == layer) for layer in layers]


fig, axes = plt.subplots(4, 4, figsize=(13, 10), sharey=True, layout="constrained")
for ax, m in zip(axes.flat, DATA["models"], strict=True):
    n = m["job_id"]
    for stage, color, label in [("teacher", COLORS[1], "Teacher update"), ("endpoint", COLORS[0], "OPD update")]:
        x, y = line_values(n, stage, "top16")
        ax.plot(x, np.array(y) * 100, color=color, label=label)
    ax.set(title=NAMES[n], xlabel="Layer index", ylabel="Top-16 energy (%)", ylim=(0, 100))
axes[0, 0].legend(fontsize=8)
save(fig, "all-model-layer-top16")

fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), layout="constrained", sharey=True)
for ax, n in zip(axes.flat, [2, 7, 8, 11, 19, 20], strict=True):
    for k, (restriction, label) in enumerate(
        [
            ("mix", "Old left + right spans"),
            ("left", "Fixed left span"),
            ("right", "Fixed right span"),
            ("iso", "Moving frames, fixed spectrum"),
        ]
    ):
        rows = sorted(
            [
                x
                for x in DATA["geometry"]
                if x["job_id"] == n and x["transition"] == "base-to-endpoint" and x["restriction"] == restriction
            ],
            key=lambda x: x["layer"],
        )
        ax.plot([x["layer"] for x in rows], [x["median"] for x in rows], label=label, color=COLORS[k])
    ax.set(title=NAMES[n], xlabel="Layer index", ylabel="Unexplained update norm ratio", ylim=(0, 1.03))
fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside upper center", ncol=4, fontsize=8)
save(fig, "layer-reconstruction-constraints")

fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
features = [
    ("teacher_rho", "Teacher spectral drift / update", True),
    ("teacher_kappa", "Teacher kappa", True),
    ("teacher_top16", "Teacher update top-16 energy", False),
    ("teacher_relative_update", "Teacher update / initial norm", True),
]
markers = {"R1": "o", "Qwen": "s", "Llama": "^", "MiMo": "D"}
for ax, (field, label, logx) in zip(axes.flat, features, strict=True):
    for family, marker in markers.items():
        for low in [True, False]:
            group = [m for m in DATA["models"] if m["family"] == family and (m["remaining"] <= 0.1) == low]
            ax.scatter(
                [m[field] for m in group],
                [m["remaining"] * 100 for m in group],
                color=COLORS[0] if low else COLORS[1],
                marker=marker,
                s=38,
                label=family + (" low residual" if low else " elevated residual"),
            )
    for n, offset in [(2, (5, 6)), (8, (5, -12)), (13, (-8, 7)), (20, (5, -12))]:
        m = next(m for m in DATA["models"] if m["job_id"] == n)
        ax.annotate(str(n), (m[field], m["remaining"] * 100), xytext=offset, textcoords="offset points", fontsize=8)
    if logx:
        ax.set_xscale("log")
    if field == "teacher_rho":
        ax.set_xticks([0.02, 0.05, 0.1, 0.2, 0.5], labels=["0.02", "0.05", "0.1", "0.2", "0.5"])
        ax.minorticks_off()
    ax.set(xlabel=label, ylabel="Endpoint / initial RKL (%)", ylim=(0, 52))
fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=4, fontsize=8)
save(fig, "absorption-versus-spectrum")

fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
for ax, (field, label) in zip(
    axes,
    [("top16", "Update top-16 energy (%)"), ("rho_sigma", "Spectral drift / update"), ("kappa", "Kappa")],
    strict=True,
):
    for k, n in enumerate([2, 7, 8]):
        stages = sorted(
            {x["stage"] for x in DATA["matrices"] if x["job_id"] == n and x["stage"].startswith("step")},
            key=lambda s: int(s[4:]),
        ) + ["endpoint"]
        x = [320 if s == "endpoint" else int(s[4:]) for s in stages]
        y = [median(v[field] for v in DATA["matrices"] if v["job_id"] == n and v["stage"] == s) for s in stages]
        if field == "top16":
            y = np.array(y) * 100
        ax.plot(x, y, marker="o", label=NAMES[n], color=COLORS[k])
    ax.set(xlabel="OPD training step", ylabel=label)
axes[0].legend(fontsize=8)
save(fig, "checkpoint-spectrum-dynamics")
# Per-cell values are matrix medians, with the same 0-80% color limits.

module_data = json.loads(gzip.decompress((ROOT / "module-angle-analysis.json.gz").read_bytes()))
fig, axes = plt.subplots(1, 2, figsize=(13, 9), layout="constrained")
labels = ["Attn shallow", "Attn middle", "Attn deep", "MLP shallow", "MLP middle", "MLP deep"]
for ax, stage, color, title in zip(
    axes, ["teacher", "endpoint"], [COLORS[1], COLORS[0]], ["Teacher update", "OPD student update"], strict=True
):
    values = []
    for m in DATA["models"]:
        values.append(
            [
                next(
                    x["top16"]
                    for x in module_data["modules"]
                    if x["job_id"] == m["job_id"]
                    and x["stage"] == stage
                    and x["kind"] == kind
                    and x["depth_third"] == band
                )
                * 100
                for kind in ["attention", "mlp"]
                for band in range(3)
            ]
        )
    cmap = LinearSegmentedColormap.from_list(title, ["white", color])
    ax.imshow(values, vmin=0, vmax=80, cmap=cmap, aspect="auto")
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=9, color="black")
    ax.set_xticks(range(6), labels=labels, rotation=35, ha="right")
    ax.set_yticks(range(16), labels=[NAMES[m["job_id"]] for m in DATA["models"]])
    ax.set(title=title, xlabel="Module and depth third", ylabel="Model condition")
save(fig, "module-depth-top16")

(ROOT / "figure-audit.json").write_text(json.dumps(audits, indent=2) + "\n")
print(json.dumps(audits, indent=2))
