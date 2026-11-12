"""Right-frame stage diagnostics, exact CPU checkpoint data.

Ledger: E63/E64/E65 are rollout-source conditions with the same initial and
teacher, not three teachers. k/v curves aggregate all 28 layers by median.
Heatmap cells are checkpoint pairs or individual layers; they are not seeds.
Angles describe cumulative-update spaces except explicit full-W panels.
Capture curves project consecutive later increments into saved anchor spaces.
"""

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

sys.path.insert(0, os.environ["FIGURE_VIZ_MPL"])
import fvstyle as fv

ROOT = Path(__file__).resolve().parent
D = json.loads((ROOT / "summary.json").read_text())
OUT = ROOT.parent.parent / "figures/right-frame-20261112"
OUT.mkdir(parents=True, exist_ok=True)
fv.apply_rc("paper")
C = fv.series_colors("paper", 4)
N = fv.NEUTRALS["baseline"]
CONDS = ["E63", "E64", "E65"]
NAMES = {"E63": "Current rollout", "E64": "Initial rollout", "E65": "Teacher rollout"}
STEPS = [16, 32, 48, 64, 80, 128, 160, 240, 320]


def save(fig, name, caption):
    fig.get_layout_engine().set(rect=(0, 0.08, 1, 0.83))
    fig.text(0.5, 0.01, caption, ha="center", fontsize=8, color=N)
    fig.canvas.draw()
    issues = [str(x) for x in fv.audit(fig, ignore=["print-fit"])]
    (ROOT / (name + "-audit.json")).write_text(json.dumps(issues, indent=2) + "\n")
    fig.savefig(OUT / (name + ".png"), dpi=180)
    fig.savefig(OUT / (name + ".pdf"))
    plt.close(fig)
    print(name, issues)


def legend(fig, labels):
    fig.legend(
        handles=[Line2D([], [], color=color, ls=ls, label=label) for label, color, ls in labels],
        loc="upper center",
        bbox_to_anchor=(0.55, 1),
        ncol=len(labels),
        frameon=False,
        fontsize=9,
    )


def pairs(cond, module, rank):
    return [r for r in D["pairs"] if r["condition"] == cond and r["module"] == module and r["rank"] == rank]


def endcurve(cond, module, rank):
    return sorted([r for r in pairs(cond, module, rank) if r["step_b"] == 320], key=lambda r: r["step_a"])


fig, axes = plt.subplots(2, 3, figsize=(12, 7.4), layout="constrained")
cmap = LinearSegmentedColormap.from_list("angles", ["white", C[0]])
for ri, rank in enumerate([16, 128]):
    for ci, cond in enumerate(CONDS):
        arr = np.zeros((9, 9))
        for ia, a in enumerate(STEPS):
            for ib, b in enumerate(STEPS):
                arr[ia, ib] = np.median(
                    [
                        r["mean_angle_deg"]
                        for mod in ["k_proj", "v_proj"]
                        for r in pairs(cond, mod, rank)
                        if r["step_a"] == min(a, b) and r["step_b"] == max(a, b)
                    ]
                )
        ax = axes[ri, ci]
        im = ax.imshow(arr, vmin=0, vmax=80, cmap=cmap)
        ax.set(
            xticks=range(9),
            xticklabels=STEPS,
            yticks=range(9),
            yticklabels=STEPS,
            title=f"{NAMES[cond]}, rank {rank}",
            xlabel="Checkpoint step",
            ylabel="Checkpoint step",
        )
        ax.tick_params(axis="x", labelrotation=60, labelsize=8)
fig.colorbar(im, ax=axes.ravel().tolist(), label="Median mean principal angle (degrees)", shrink=0.8)
save(
    fig,
    "pairwise-angles",
    "Cumulative updates: k/v module medians combined; every cell uses measured checkpoints <=320.",
)

fig, axes = plt.subplots(2, 3, figsize=(12, 6.8), layout="constrained", sharey=True)
for ci, cond in enumerate(CONDS):
    for ri, rank in enumerate([16, 128]):
        ax = axes[ri, ci]
        for mi, mod in enumerate(["k_proj", "v_proj"]):
            rows = endcurve(cond, mod, rank)
            for key, ls in [("mean_angle_deg", "-"), ("p90_angle_deg", "--")]:
                ax.plot([r["step_a"] for r in rows], [r[key] for r in rows], color=C[mi], ls=ls)
        ax.set(
            title=f"{NAMES[cond]}, rank {rank}",
            xlabel="Checkpoint step",
            ylabel="Angle to step320 (degrees)",
            ylim=(0, 90),
        )
legend(fig, [("k: mean", C[0], "-"), ("k: p90", C[0], "--"), ("v: mean", C[1], "-"), ("v: p90", C[1], "--")])
save(
    fig,
    "angles-to-reference",
    "Per-matrix angle mean/p90, then layer median. Zero at step320 is a self-comparison identity.",
)

fig, axes = plt.subplots(2, 3, figsize=(12, 6.8), layout="constrained", sharey=True)
for ci, cond in enumerate(CONDS):
    for ri, key in enumerate(["capture", "relative_to_optimal"]):
        ax = axes[ri, ci]
        for ai, anchor in enumerate([16, 64, 160]):
            rows = [
                r
                for r in D["increments"]
                if r["condition"] == cond
                and r["rank"] == 128
                and r["anchor"] == anchor
                and r["module"] in ["k_proj", "v_proj"]
            ]
            intervals = sorted({(r["start"], r["end"]) for r in rows})
            vals = [np.median([r[key] for r in rows if (r["start"], r["end"]) == iv]) for iv in intervals]
            ax.plot([iv[1] for iv in intervals], vals, color=C[ai], marker="o", label=f"Anchor step{anchor}")
        ax.set(
            title=NAMES[cond],
            xlabel="Later interval end step",
            ylabel="Increment energy captured" if ri == 0 else "Capture / optimal rank128",
            ylim=(0, 1.05),
        )
legend(fig, [(f"Anchor step{a}", C[i], "-") for i, a in enumerate([16, 64, 160])])
save(
    fig,
    "later-increment-capture",
    "Right rank128: later increments projected into earlier cumulative frames; k/v layer/module medians.",
)

records = json.loads(__import__("gzip").decompress((ROOT / "measurements.json.gz").read_bytes()))
fig, axes = plt.subplots(2, 3, figsize=(12, 5.6), layout="constrained")
for ci, cond in enumerate(CONDS):
    for ri, mod in enumerate(["k_proj", "v_proj"]):
        values = []
        for rec in sorted(
            [r for r in records if r["task"]["condition"] == cond and r["task"]["tensor"].split(".")[-2] == mod],
            key=lambda r: int(r["task"]["tensor"].split(".")[2]),
        ):
            values.append(
                next(
                    r["mean_angle_deg"]
                    for r in rec["pairs"]
                    if r["rank"] == 16 and r["step_a"] == 64 and r["step_b"] == 320
                )
            )
        ax = axes[ri, ci]
        im = ax.imshow(np.array(values)[None, :], cmap=cmap, vmin=0, vmax=80, aspect="auto")
        ax.set(
            yticks=[0],
            yticklabels=[mod],
            xticks=[0, 6, 13, 20, 27],
            xlabel="Transformer layer",
            ylabel="Module",
            title=NAMES[cond],
        )
fig.colorbar(im, ax=axes.ravel().tolist(), label="Mean principal angle (degrees)", shrink=0.8)
save(
    fig,
    "angles-by-layer",
    "One cell = one matrix: right top16 cumulative-update angle, step64 to step320; all 28 k/v layers.",
)

fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), layout="constrained", sharey=True)
for ax, cond in zip(axes, CONDS, strict=True):
    for i, mod in enumerate(["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]):
        rows = sorted(
            [r for r in D["weight_angles"] if r["condition"] == cond and r["module"] == mod and r["rank"] == 16],
            key=lambda r: r["step"],
        )
        ax.plot(
            [r["step"] for r in rows],
            [r["mean_angle_deg"] for r in rows],
            color=C[[2, 0, 1, 3, 2, 0, 1][i]],
            ls="--" if i >= 4 else "-",
        )
    ax.set(title=NAMES[cond], xlabel="Checkpoint step", ylabel="Full W angle from initial (degrees)")
legend(
    fig,
    [
        (m.replace("_proj", ""), C[[2, 0, 1, 3, 2, 0, 1][i]], "--" if i >= 4 else "-")
        for i, m in enumerate(["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    ],
)
save(
    fig,
    "full-weight-angles",
    "Layer13 only; full-weight right top16 angles from initial. These are not cumulative-update frame angles.",
)

fig, axes = plt.subplots(2, 3, figsize=(12, 6.8), layout="constrained", sharey="row")
for ci, cond in enumerate(CONDS):
    for mi, mod in enumerate(["k_proj", "v_proj"]):
        rows = sorted(
            [r for r in D["stages"] if r["condition"] == cond and r["module"] == mod], key=lambda r: r["step"]
        )
        for ri, key in enumerate(["stable_rank", "energy"]):
            axes[ri, ci].plot([r["step"] for r in rows], [r[key] for r in rows], color=C[mi], marker="o")
    axes[0, ci].set(title=NAMES[cond], xlabel="Checkpoint step", ylabel="Median stable rank")
    axes[1, ci].set(xlabel="Checkpoint step", ylabel="Median squared update norm", yscale="log")
legend(fig, [("k", C[0], "-"), ("v", C[1], "-")])
save(
    fig,
    "update-rank-scale",
    "Geometry of OPD companion diagnostics: exact stable rank and squared Frobenius norm, all 28 k/v layers.",
)

hist = ["GRPO", "Samsung", "JustRL", "Light", "e3-lineage", "DeepMath-lineage"]
fig, axes = plt.subplots(2, 3, figsize=(12, 6.8), layout="constrained", sharey=True)
for ax, cond in zip(axes.flat, hist, strict=True):
    reference = max(r["step_b"] for r in D["pairs"] if r["condition"] == cond)
    for i, rank in enumerate([16, 64, 128]):
        for mod, ls in [("k_proj", "-"), ("v_proj", "--")]:
            rows = sorted([r for r in pairs(cond, mod, rank) if r["step_b"] == reference], key=lambda r: r["step_a"])
            ax.plot(
                [r["step_a"] for r in rows],
                [r["mean_angle_deg"] for r in rows],
                color=C[i],
                ls=ls if len(rows) > 2 else "none",
                marker="o" if mod == "k_proj" else "^",
            )
    ax.set(title=cond, xlabel="Checkpoint step", ylabel="Angle to first-pass reference (degrees)", ylim=(-3, 90))
legend(fig, [(f"rank{k}", C[i], "-") for i, k in enumerate([16, 64, 128])])
save(
    fig,
    "historical-angles",
    "Layers0/13/27; k circles / v triangles. Lines only for JustRL; others have two measured checkpoints.",
)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained", sharey=True)
for ax, bank in zip(axes, ["public_initial", "public_teacher"], strict=True):
    for i, cond in enumerate(CONDS):
        rows = sorted(
            [r for r in D["risks"] if r["condition"] == cond and r["bank"] == bank and r["domain"] == "all"],
            key=lambda r: r["step"],
        )
        ax.plot([r["step"] for r in rows], [r["fixed_h1024"] for r in rows], color=C[i], marker="o")
    ax.set(
        title=bank.replace("public_", "").capitalize() + " prefix",
        xlabel="Optimizer step",
        ylabel="Fixed-H1024 full-vocab RKL",
    )
legend(fig, [(NAMES[c], C[i], "-") for i, c in enumerate(CONDS)])
save(
    fig,
    "matched-risk",
    "Same 128 heldout questions per bank: Math64 / Code32 / IF32; common initial and teacher, three rollout sources.",
)

fig, axes = plt.subplots(2, 3, figsize=(12, 6.8), layout="constrained", sharey=True)
for ci, cond in enumerate(CONDS):
    for ri, rank in enumerate([16, 128]):
        ax = axes[ri, ci]
        for ai, anchor in enumerate([16, 64, 160]):
            distributions = [
                p["angles_deg"]
                for rec in records
                if rec["task"]["condition"] == cond and rec["task"]["tensor"].split(".")[-2] in ["k_proj", "v_proj"]
                for p in rec["pairs"]
                if p["rank"] == rank and p["step_a"] == anchor and p["step_b"] == 320
            ]
            arr = np.median(np.asarray(distributions), axis=0)
            ax.plot(100 * np.arange(1, rank + 1) / rank, arr, color=C[ai])
        ax.set(
            title=f"{NAMES[cond]}, rank {rank}",
            xlabel="Ordered principal-angle index (%)",
            ylabel="Angle to step320 (degrees)",
            ylim=(0, 90),
        )
legend(fig, [(f"Checkpoint step{a}", C[i], "-") for i, a in enumerate([16, 64, 160])])
save(
    fig,
    "principal-angle-distribution",
    "All k principal angles retained; curves take the median at each ordered angle index across 56 k/v matrices.",
)

endpoint = json.loads((ROOT / "endpoint-context.json").read_text())
models = sorted(
    endpoint["models"],
    key=lambda m: (
        m["historical_h1024"] is None,
        m["historical_h1024"]["remaining"] if m["historical_h1024"] else m["remaining"],
    ),
)
fig, ax = plt.subplots(figsize=(8.5, 7.4), layout="constrained")
y = np.arange(len(models))
for transition, color, label in [("base-to-teacher", C[1], "Teacher"), ("base-to-endpoint", C[0], "Student")]:
    vals = [
        next(
            r["median_mean_angle_deg"]
            for r in endpoint["full_weight_angles"]
            if r["job_id"] == m["job_id"] and r["side"] == "right" and r["rank"] == 16 and r["transition"] == transition
        )
        for m in models
    ]
    ax.scatter(vals, y, color=color, label=label)
ax.set(
    yticks=y,
    yticklabels=[m["name"] for m in models],
    xscale="log",
    xlabel="Full W right top16 angle from initial (degrees)",
    ylabel="Model / mode condition",
)
ax.invert_yaxis()
legend(fig, [("Teacher", C[1], "-"), ("Student", C[0], "-")])
save(
    fig,
    "endpoint-right-angles",
    "16 endpoint conditions; all-layer seven-projection matrix medians. Full-weight angle, not update-frame angle.",
)
