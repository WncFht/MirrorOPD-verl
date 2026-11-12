"""Render frozen trajectory data beside the report.

Semantic ledger: heatmap rows are distinct training conditions with explicit
reference steps; columns are matrix modules; cells are matrix-median adjusted
subspace overlaps, never model replicate means. Line colors identify the three
rollout policies in every panel. New-run x axes are optimizer steps (outer*16).
Geometry ledger: seven equal module columns, one equal row per trajectory;
all comparisons use the same [0,1] color scale. Last E65 point is provisional.
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

sys.path.insert(0, os.environ["FIGURE_VIZ_MPL"])
import fvstyle as fv

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent.parent / "figures/layer-review"
OUT.mkdir(parents=True, exist_ok=True)
data = json.loads((ROOT / "cohort-summary.json").read_text())
records = json.loads(gzip.decompress((ROOT / "cohort-dynamics.json.gz").read_bytes()))["records"]
readouts = json.loads((ROOT / "cohort-readouts.json").read_text())
fv.apply_rc("paper")


def save(fig, name, caption):
    fig.get_layout_engine().set(rect=(0, 0.06, 1, 0.94))
    fig.text(0.5, 0.01, caption, ha="center", fontsize=8, color=fv.NEUTRALS["baseline"])
    fig.canvas.draw()
    audit = [str(x) for x in fv.audit(fig, ignore=["print-fit"])]
    (ROOT / (name + "-audit.json")).write_text(json.dumps(audit, indent=2) + "\n")
    fig.savefig(OUT / (name + ".png"), dpi=180)
    fig.savefig(OUT / (name + ".pdf"))
    plt.close(fig)
    print(name, audit)


modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
conditions = data["conditions"]
cmap = fv.ramp_cmap("seq")
for rank in [16, 128]:
    fig, axes = plt.subplots(1, 2, figsize=(12, 8), layout="constrained", sharey=True)
    labels = []
    for condition in conditions:
        suffix = "*" if condition.get("training_complete") is False else ""
        labels.append(f"{condition['teacher']}  {condition['steps'][0]}→{condition['steps'][-1]}{suffix}")
    for ax, side in zip(axes, ["left", "right"], strict=True):
        grid = np.full((len(conditions), len(modules)), np.nan)
        for ri, condition in enumerate(conditions):
            for ci, module in enumerate(modules):
                found = [
                    r
                    for r in data["subspaces"]
                    if r["job_id"] == condition["job_id"]
                    and r["label"] == condition["points"][0][0]
                    and r["side"] == side
                    and r["module"] == module
                    and r["requested_rank"] == rank
                ]
                assert len(found) == 1
                grid[ri, ci] = found[0]["adjusted_final_overlap"]
        mesh = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=1)
        ax.set(
            xticks=range(len(modules)),
            xticklabels=[m.replace("_proj", "") for m in modules],
            yticks=range(len(conditions)),
            yticklabels=labels,
            title=f"{side.capitalize()} update space, k={rank}",
        )
        for (ri, ci), value in np.ndenumerate(grid):
            ax.text(
                ci,
                ri,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color=fv.NEUTRALS["vink"],
            )
    fig.colorbar(mesh, ax=axes, shrink=0.7, label="Adjusted subspace overlap")
    save(
        fig,
        f"cohort-module-rank{rank}",
        "k/v: 28 layers; q/o: 5 layers; MLP: middle layer. Reference fractions differ; * = last retained E65 snapshot.",
    )

new = [c for c in conditions if "rollout_policy" in c]
colors = fv.series_colors("paper", len(new))
fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), layout="constrained")
for condition, color in zip(new, colors, strict=True):
    job = condition["job_id"]
    label = condition["teacher"]
    reference = condition["steps"][-1]
    risks = readouts[str(job)]["rows"]
    for ax, bank in zip(axes[0, :2], ["public_initial", "public_teacher"], strict=True):
        key = f"eval/rkl/{bank}/all/fixed_h1024"
        rows = [r for r in risks if key in r["data"] and r["step"] * 16 <= reference]
        ax.plot([r["step"] * 16 for r in rows], [r["data"][key] for r in rows], color=color, marker=".", label=label)
        ax.set(xlabel="Optimizer step", ylabel="Fixed-H1024 RKL", title=bank.replace("public_", "") + " prefix")
    energy, top16, subspace, spectral = [], [], [], []
    for stage, step in zip(condition["points"], condition["steps"], strict=True):
        matrices = [
            r
            for r in records
            if r["task"]["job_id"] == job and r["task"]["tensor"].split(".")[-2] in ["k_proj", "v_proj"]
        ]
        rows = [next(s for s in r["stages"] if s["label"] == stage[0]) for r in matrices]
        assert len(rows) == 56
        energy.append(sum(r["energy"] for r in rows))
        top16.append(median(sum(s * s for s in r["singular_values"][:16]) / r["energy"] for r in rows))
        spectral.append(median(r["weight_geometry"]["delta_sigma"] for r in rows))
        overlap = [
            s
            for r in matrices
            for s in r["rows"]
            if s["label"] == stage[0] and s["side"] == "right" and s["requested_rank"] == 16
        ]
        subspace.append(median(r["adjusted_final_overlap"] for r in overlap))
    x = condition["steps"]
    for ax, values, title, ylabel in [
        (axes[0, 2], energy, "Cumulative k/v update", "Selected update energy"),
        (axes[1, 0], top16, "Update concentration", "Median top-16 energy fraction"),
        (axes[1, 1], subspace, "Right update space to reference", "Adjusted rank-16 overlap"),
        (axes[1, 2], spectral, "Full-weight singular values", "Median δΣ"),
    ]:
        ax.plot(x, values, color=color, marker=".", label=label + (f" (ref {reference})" if ax is axes[1, 1] else ""))
        ax.set(xlabel="Optimizer step", ylabel=ylabel, title=title)
axes[0, 0].set_ylim(0.20, 0.57)
axes[0, 1].set_ylim(0.20, 0.57)
axes[1, 1].legend(fontsize=8)
fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside upper center", ncol=3)
save(
    fig,
    "qwen-prefix-spectral-dynamics",
    "Same 1.7B student / 4B teacher; all 28 k/v layers. E65 is partial with a different reference step.",
)
