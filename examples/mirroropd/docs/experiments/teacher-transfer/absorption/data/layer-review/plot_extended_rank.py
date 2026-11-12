"""Plot observed rank curves and exact checkpoint-to-final subspace overlap.

Ledger: each rank point is the unweighted median over all seven projections;
coral teacher, teal OPD. Temporal panels are medians over k_proj layers 0/13/27,
not confidence intervals. Each curve is one rank, row is left/right of delta W.
No interpolation implies unmeasured checkpoints; markers are measured steps.
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

sys.path.insert(0, os.environ["FIGURE_VIZ_MPL"])
import fvstyle as fv

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent.parent / "figures/layer-review"
fv.apply_rc("paper")
colors = fv.series_colors("paper", 4)
models = json.loads((ROOT / "energy-amplitude-analysis.json").read_text())["models"]
data = json.loads(gzip.decompress((ROOT / "extended-rank-analysis.json.gz").read_bytes()))
audits = {}


def save(fig, name, caption):
    fig.get_layout_engine().set(rect=(0, 0.04, 1, 0.94))
    fig.text(0.5, 0.005, caption, ha="center", fontsize=8, color=fv.NEUTRALS["baseline"])
    fig.canvas.draw()
    audits[name] = [str(p) for p in fv.audit(fig, ignore=["print-fit"])]
    fig.savefig(OUT / (name + ".png"), dpi=180)
    fig.savefig(OUT / (name + ".pdf"))
    plt.close(fig)


fig, axes = plt.subplots(4, 4, figsize=(13, 11), sharey=True, layout="constrained")
for ax, model in zip(axes.flat, models, strict=True):
    for stage, color, label in [("teacher", colors[1], "Teacher"), ("endpoint", colors[0], "OPD")]:
        row = next(x for x in data["summary"] if x["job_id"] == model["job_id"] and x["stage"] == stage)
        ax.plot(
            data["ranks"],
            [row["top_median"][str(k)] * 100 for k in data["ranks"]],
            marker="o",
            ms=3,
            color=color,
            label=label,
        )
    ax.set(xscale="log", ylim=(0, 101), xlabel="Rank per matrix", ylabel="Update energy (%)", title=model["name"])
    ax.set_xticks([16, 64, 256, 512], ["16", "64", "256", "512"])
    ax.minorticks_off()
fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside upper center", ncol=2)
save(
    fig,
    "extended-rank-curves",
    "Exact stored spectra; median energy capture across all layers and seven projection types.",
)

rows = json.loads((ROOT / "extended-temporal-analysis.json").read_text())["rows"]
fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharey=True, layout="constrained")
for col, pair in enumerate(["qwen-grpo", "qwen-sft", "r1-justrl"]):
    for row_index, side in enumerate(["left", "right"]):
        ax = axes[row_index, col]
        for i, rank in enumerate([16, 64, 128]):
            group = [r for r in rows if r["pair"] == pair and r["side"] == side and r["requested_rank"] == rank]
            steps = sorted({r["step"] for r in group})
            values = [median(r["final_adjusted_overlap"] for r in group if r["step"] == step) for step in steps]
            ax.plot(steps, values, marker="o", ms=4, color=colors[i], label=f"Rank {rank}")
        ax.set(xlabel="Training step", ylabel=f"{side.capitalize()} adjusted overlap", title=pair, ylim=(-0.05, 1.05))
fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside upper center", ncol=3)
save(
    fig,
    "checkpoint-to-final-both-sides",
    "Delta W subspaces; (overlap - k/d) / (1 - k/d); median of k_proj layers 0/13/27; no new training.",
)
(ROOT / "extended-rank-figure-audit.json").write_text(json.dumps(audits, indent=2) + "\n")
print(audits)
