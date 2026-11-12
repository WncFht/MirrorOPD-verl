"""Plot measured absorption groups and matched training windows.

Ledger: each endpoint row is one named teacher/mode/budget condition, not an
independent training seed. Line colors identify a condition within a family;
every repeated condition has identical color in risk and geometry panels.
Risk curves use all recorded readouts; geometry uses saved checkpoints only,
with visible markers and straight connecting segments. Parameter statistics
cover the same 69 matrices; k/v spectrum and frame panels use 28 layers each.
Geometry: three equal family columns, shared limits within each metric row.
No global risk threshold or smoothing is introduced.
"""

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

sys.path.insert(0, os.environ["FIGURE_VIZ_MPL"])
import fvstyle as fv

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent.parent / "figures/grouped-dynamics"
OUT.mkdir(parents=True, exist_ok=True)
data = json.loads((ROOT / "grouped-dynamics.json").read_text())
trajectories = {r["job_id"]: r for r in data["trajectories"]}
colors = fv.series_colors("paper", 4)
fv.apply_rc("paper")
palette = {
    100: colors[0],
    101: colors[2],
    102: colors[1],
    103: colors[0],
    104: colors[2],
    108: colors[1],
    109: colors[0],
    110: colors[1],
}
names = {
    100: "Llama CoReward-I",
    101: "Llama GT",
    102: "Llama arithmetic SFT",
    103: "Qwen e3 thinking",
    104: "Samsung non-thinking",
    108: "Thinking-Space non-thinking",
    109: "Qwen e3 lineage",
    110: "Qwen DeepMath SFT",
}


def save(fig, name, caption):
    fig.get_layout_engine().set(rect=(0, 0.08, 1, 0.88))
    fig.text(0.5, 0.009, caption, ha="center", fontsize=8, color=fv.NEUTRALS["baseline"])
    fig.canvas.draw()
    audit = [str(x) for x in fv.audit(fig, ignore=["print-fit"])]
    (ROOT / (name + "-audit.json")).write_text(json.dumps(audit, indent=2) + "\n")
    fig.savefig(OUT / (name + ".png"), dpi=180)
    fig.savefig(OUT / (name + ".pdf"))
    plt.close(fig)
    print(name, audit)


models = data["h128_classification"]
row_names = {
    8: "Qwen Samsung thinking",
    10: "Qwen Samsung non-thinking",
    19: "MiMo SFT to SFT+RL",
    20: "MiMo RL-Zero branch",
}
models = sorted(models, key=lambda r: (r["group"] != "lower residual", r["endpoint_rkl"]))
fig, axes = plt.subplots(1, 2, figsize=(10.5, 7), layout="constrained", sharey=True)
y = np.arange(len(models))
cs = [colors[0] if r["group"] == "lower residual" else colors[1] for r in models]
axes[0].scatter([r["initial_rkl"] for r in models], y, marker="|", color=fv.NEUTRALS["baseline"], label="Initial")
axes[0].scatter([r["endpoint_rkl"] for r in models], y, color=cs, label="_nolegend_")
axes[0].add_collection(
    LineCollection(
        [[(row["endpoint_rkl"], i), (row["initial_rkl"], i)] for i, row in enumerate(models)],
        colors=cs,
        linewidths=1,
        alpha=0.6,
    )
)
axes[0].set(
    xscale="log",
    xlabel="Fixed-H128 RKL (log scale)",
    yticks=y,
    yticklabels=[row_names.get(r["job_id"], r["name"]) for r in models],
    title="Absolute residual and initial gap",
)
axes[0].invert_yaxis()
axes[0].set_ylabel("Teacher / mode / budget condition")
axes[1].set_ylabel("Same conditions")
fig.legend(
    handles=[
        Line2D([], [], marker="|", linestyle="none", color=fv.NEUTRALS["baseline"], label="Initial"),
        Line2D([], [], marker="o", linestyle="none", color=colors[0], label="Lower residual"),
        Line2D([], [], marker="o", linestyle="none", color=colors[1], label="Higher residual"),
    ],
    loc="upper center",
    bbox_to_anchor=(0.58, 1.0),
    ncol=3,
    frameon=False,
)
axes[1].barh(y, [100 * r["remaining"] for r in models], color=cs)
axes[1].set(xlabel="Remaining initial RKL (%)", title="Same protocol, relative residual", xlim=(0, 55))
for i, row in enumerate(models):
    axes[1].text(row["remaining"] * 100 + 0.8, i, f"{row['remaining'] * 100:.1f}%", va="center", fontsize=8)
for ax in axes:
    ax.axhline(6.5, color=fv.NEUTRALS["baseline"], ls="--", lw=0.8)
save(
    fig,
    "absorption-groups",
    "Initial = gray tick; endpoint = dot. Teal = lower residual; orange = higher. "
    "DeepCoder / Light: 64 steps; others: 320.",
)

fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), layout="constrained", sharey="row")
families = [
    ([100, 101, 102], "Llama: through step 640", 640),
    ([109, 110], "Qwen: first 320 steps", 320),
    ([103, 104, 108], "Qwen: through step 1280", 1280),
]
for col, (jobs, title, limit) in enumerate(families):
    for job in jobs:
        points = [p for p in trajectories[job]["risks"] if p["step"] <= limit]
        for ri, metric in enumerate(["rkl", "remaining"]):
            axes[ri, col].plot(
                [p["step"] for p in points], [p[metric] for p in points], color=palette[job], label=names[job]
            )
    axes[0, col].set(
        xlabel="Optimizer step", title=title, yscale="log", ylim=(0.0003, 0.8), ylabel="Absolute H1024 RKL (log)"
    )
    axes[1, col].set(xlabel="Optimizer step", ylabel="RKL / initial RKL", ylim=(0, 1.05))
    axes[0, col].legend(fontsize=8)
save(
    fig,
    "risk-dynamics",
    "Fixed S0 training: 20 batches × 16 reuse, repeated every 320 steps. Same fixed H1024 heldout panel.",
)

# A common 320→1280 window removes the unequal-reference problem in prior figures.
jobs = [103, 104, 108]
fig, axes = plt.subplots(3, 2, figsize=(11, 9), layout="constrained")
for job in jobs:
    trajectory = trajectories[job]
    points = trajectory["stages"]
    x = [r["step"] for r in points]
    for ci, module in enumerate(["k_proj", "v_proj"]):
        axes[0, ci].plot(
            x,
            [r["modules"][module]["right_overlap_16"] for r in points],
            color=palette[job],
            marker="o",
            label=names[job],
        )
        axes[1, ci].plot(
            x, [r["modules"][module]["top16"] for r in points], color=palette[job], marker="o", label=names[job]
        )
        axes[0, ci].set(
            title=module + ": right rank16 → step1280", ylabel="Adjusted subspace overlap", ylim=(0.5, 1.02)
        )
        axes[1, ci].set(
            title=module + ": update concentration", ylabel="Median top16 energy fraction", ylim=(0.2, 0.62)
        )
    axes[2, 0].plot(
        x, [r["teacher_geometry"]["norm_ratio"] for r in points], color=palette[job], marker="o", label=names[job]
    )
    axes[2, 1].plot(
        x, [r["teacher_geometry"]["residual"] for r in points], color=palette[job], marker="o", label=names[job]
    )
axes[2, 0].set(title="Selected parameter update amplitude", ylabel="Student / teacher update norm", ylim=(-0.04, 1.4))
axes[2, 1].set(title="Selected parameter residual", ylabel="Squared distance / teacher update energy", ylim=(0.9, 2.4))
for ax in axes.flat:
    ax.set(xlabel="Optimizer step", xticks=[320, 640, 1280], xlim=(280, 1320))
axes[0, 0].legend(fontsize=8)
save(
    fig,
    "matched-window-geometry",
    "Same Qwen initial, same 320→1280 window. Markers = measured weight snapshots; k/v medians over all 28 layers.",
)
