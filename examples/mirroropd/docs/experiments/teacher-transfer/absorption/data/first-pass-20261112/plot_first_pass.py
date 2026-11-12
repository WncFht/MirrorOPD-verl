"""First-pass measured risks, frames, spectra and parameter residuals.

Ledger: one row is a named teacher/mode/run condition, not a training seed.
Endpoint geometry covers all seven projections at all layers (16 conditions).
H1024 risks use 128 heldout questions, 14 conditions; MiMo risk is absent here.
Update-frame heatmaps cover six trajectories, with k/v all layers, q/o five
layers and MLP one middle layer. Left/right and rank16/128 remain separate.
Every stage reference is <=320 steps. No continuation or repeated pass is used.
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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator

sys.path.insert(0, os.environ["FIGURE_VIZ_MPL"])
import fvstyle as fv

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent.parent / "figures/first-pass-20261112"
OUT.mkdir(parents=True, exist_ok=True)
D = json.loads((ROOT / "analysis.json").read_text())
fv.apply_rc("paper")
C = fv.series_colors("paper", 4)
N = fv.NEUTRALS["baseline"]
MODELS = sorted(
    D["models"],
    key=lambda m: (
        m["historical_h1024"] is None,
        m["historical_h1024"]["remaining"] if m["historical_h1024"] else m["remaining"],
    ),
)
NAMES = {
    8: "Qwen Samsung thinking",
    10: "Qwen Samsung non-thinking",
    19: "MiMo SFT to SFT+RL",
    20: "MiMo RL-Zero branch",
}


def save(fig, name, caption):
    fig.get_layout_engine().set(rect=(0, 0.08, 1, 0.88))
    fig.text(0.5, 0.009, caption, ha="center", fontsize=8, color=N)
    fig.canvas.draw()
    issues = [str(x) for x in fv.audit(fig, ignore=["print-fit"])]
    (ROOT / (name + "-audit.json")).write_text(json.dumps(issues, indent=2) + "\n")
    fig.savefig(OUT / (name + ".png"), dpi=180)
    fig.savefig(OUT / (name + ".pdf"))
    plt.close(fig)
    print(name, issues)


def rows_axes(n, rows, width=13):
    fig, axes = plt.subplots(1, n, figsize=(width, 8.2), layout="constrained", sharey=True)
    y = np.arange(len(rows))
    for i, ax in enumerate(axes):
        ax.set(
            yticks=y,
            yticklabels=[NAMES.get(m["job_id"], m["name"]) for m in rows],
            ylabel="Teacher / mode" if i == 0 else "Same conditions",
        )
    axes[0].invert_yaxis()
    return fig, axes, y


def legend(fig, entries):
    fig.legend(
        handles=[Line2D([], [], marker="o", linestyle="none", color=c, label=label) for label, c in entries],
        loc="upper center",
        bbox_to_anchor=(0.62, 1),
        ncol=len(entries),
        frameon=False,
    )


risk_models = [m for m in MODELS if m["historical_h1024"]]
fig, axes, y = rows_axes(2, risk_models, 10.5)
for key, c, label in [("initial", N, "Initial"), ("final", C[0], "Student")]:
    axes[0].scatter([m["historical_h1024"][key] for m in risk_models], y, color=c, label=label)
axes[0].set(xscale="log", xlabel="H1024 RKL (nats; log)", title="Absolute initial and final gap")
axes[1].scatter([100 * m["historical_h1024"]["remaining"] for m in risk_models], y, color=C[0])
axes[1].add_collection(
    LineCollection(
        [
            [
                (100 * m["historical_h1024"]["remaining_ci95"][0], i),
                (100 * m["historical_h1024"]["remaining_ci95"][1], i),
            ]
            for i, m in enumerate(risk_models)
        ],
        colors=C[0],
        linewidths=1,
    )
)
axes[1].set(xlabel="Remaining initial RKL (%)", title="Same-question ratio: 95% bootstrap CI", xlim=(0, 65))
legend(fig, [("Initial", N), ("Student", C[0])])
save(
    fig,
    "h1024-risk",
    "First pass only; 128 heldout questions (64 Math / 32 Code / 32 IF). DeepCoder and Light: 64 steps; others: 320.",
)

fig, axes, y = rows_axes(2, MODELS, 11)
for i, side in enumerate(["left", "right"]):
    for transition, color, label in [("base-to-teacher", C[1], "Teacher"), ("base-to-endpoint", C[0], "Student")]:
        values = [
            next(
                r["median_mean_angle_deg"]
                for r in D["full_weight_angles"]
                if r["job_id"] == m["job_id"]
                and r["rank"] == 16
                and r["side"] == side
                and r["transition"] == transition
            )
            for m in MODELS
        ]
        axes[i].scatter(values, y, color=color, label=label)
    axes[i].set(
        xscale="log",
        xlabel="Median mean principal angle (degrees; log)",
        title=f"Full W: initial to {side} top16 space",
    )
legend(fig, [("Teacher", C[1]), ("Student", C[0])])
save(
    fig,
    "frame-full-weight",
    "16 conditions, all layers and seven projections; full-weight frames are distinct from cumulative-update frames.",
)

frame_jobs = [7, 8, 2, 6, 109, 110]
frame_names = [
    "GRPO 64→320",
    "Samsung thinking 64→320",
    "JustRL 16→320",
    "Light 32→64",
    "e3 lineage 64→320",
    "DeepMath lineage 64→320",
]
modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), layout="constrained", sharex=True, sharey=True)
cmap = LinearSegmentedColormap.from_list("frame-overlap", ["white", C[0]])
for ri, rank in enumerate([16, 128]):
    for ci, side in enumerate(["left", "right"]):
        ax = axes[ri, ci]
        values = np.array(
            [
                [
                    next(
                        r["adjusted"]
                        for r in D["frames"]
                        if r["job_id"] == job
                        and r["rank"] == rank
                        and r["side"] == side
                        and r["module"] == mod
                        and r["step"] == (16 if job == 2 else 32 if job == 6 else 64)
                    )
                    for mod in modules
                ]
                for job in frame_jobs
            ]
        )
        ax.imshow(values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for y0, x0 in np.ndindex(values.shape):
            ax.text(x0, y0, f"{values[y0, x0]:.2f}", ha="center", va="center", fontsize=8)
        ax.set(
            title=f"Update space: {side}, rank {rank}",
            xticks=range(7),
            xticklabels=[m.replace("_proj", "") for m in modules],
            yticks=range(6),
            yticklabels=frame_names,
            xlabel="Projection module",
            ylabel="First-pass checkpoint → reference",
        )
save(
    fig,
    "frame-first-pass",
    "Adjusted overlap (0 = random baseline; 1 = same space). k/v: all layers; q/o: five layers; MLP: middle layer.",
)

fig, axes, y = rows_axes(4, MODELS, 14)
for key, c, label in [("teacher", C[1], "Teacher"), ("student", C[0], "Student")]:
    for ax, suffix, scale in [
        (axes[0], "top16", 100),
        (axes[1], "rho", 1),
        (axes[2], "kappa", 1),
        (axes[3], "relative_update", 100),
    ]:
        ax.scatter([m[f"{key}_{suffix}"] * scale for m in MODELS], y, color=c, label=label)
axes[0].set(xlabel="Top16 update energy (%)", title="Delta W concentration", xlim=(0, 65))
axes[1].set(xlabel="Spectral drift / update norm", title="Full W: rho Sigma", xscale="log")
axes[1].xaxis.set_major_locator(FixedLocator([0.02, 0.1, 0.4]))
axes[1].set_xticklabels(["0.02", "0.1", "0.4"])
axes[1].xaxis.set_minor_locator(NullLocator())
axes[2].set(xlabel="Diagonal enrichment", title="Base-frame kappa", xscale="log")
axes[3].set(xlabel="Update / initial norm (%)", title="Update magnitude", xscale="log")
legend(fig, [("Teacher", C[1]), ("Student", C[0])])
save(
    fig,
    "spectrum-models",
    "16 endpoint conditions; matrix medians for top16/rho/kappa, pooled Frobenius norms for update magnitude.",
)

fig, axes, y = rows_axes(4, MODELS, 14)
available = [(i, m) for i, m in enumerate(MODELS) if m["historical_h1024"]]
axes[0].scatter(
    [m["historical_h1024"]["final"] for i, m in available], [i for i, m in available], color=C[0], label="Student"
)
for i, m in enumerate(MODELS):
    if not m["historical_h1024"]:
        axes[0].text(
            0.05, i, "separate risk protocol", transform=axes[0].get_yaxis_transform(), va="center", fontsize=8
        )
for ax, key in [
    (axes[1], "student_teacher_norm_ratio"),
    (axes[2], "parameter_residual_energy"),
    (axes[3], "update_cosine"),
]:
    ax.scatter([m[key] for m in MODELS], y, color=C[0])
axes[0].set(xlabel="H1024 RKL (log)", title="Output residual", xscale="log")
axes[1].set(xlabel="Student / teacher update norm", title="Update amplitude a", xscale="log")
axes[2].set(xlabel="Squared distance / teacher energy", title="Parameter residual", xlim=(0.85, 2.8))
axes[2].axvline(1, color=N, ls="--", lw=1)
axes[3].set(xlabel="Frobenius update cosine", title="Update alignment c", xlim=(0, 0.30))
legend(fig, [("Student", C[0])])
save(
    fig,
    "parameter-residuals",
    "Same all-layer parameter set in each row: R = 1 + a² − 2ac. Dashed line: initial parameter distance to teacher.",
)

fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), layout="constrained", sharey=True)
for ax, job, label in zip(axes.flat, frame_jobs, frame_names, strict=True):
    for module, color in [("k_proj", C[0]), ("v_proj", C[1])]:
        rows = sorted(
            [r for r in D["spectra"] if r["job_id"] == job and r["module"] == module], key=lambda r: r["step"]
        )
        ax.plot([r["step"] for r in rows], [r["top16"] for r in rows], color=color, marker="o", label=module)
    ax.set(title=label, xlabel="Optimizer step", ylabel="Median top16 update energy", ylim=(0.15, 0.75))
legend(fig, [("k_proj", C[0]), ("v_proj", C[1])])
save(
    fig,
    "spectrum-first-pass",
    "Six first-pass trajectories; markers are measured checkpoints, connecting segments do not "
    "add temporal observations.",
)

fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), layout="constrained")
for col, family in enumerate(["R1", "Qwen", "Llama"]):
    models = [m for m in D["models"] if m["family"] == family and m["historical_h1024"]]
    colors = [C[i % 4] for i in range(len(models))]
    for mi, (m, c) in enumerate(zip(models, colors, strict=True)):
        rows = sorted([r for r in D["risks"] if r["job_id"] == m["job_id"]], key=lambda r: r["step"])
        for row, key in enumerate(["rkl", "remaining"]):
            axes[row, col].plot(
                [r["step"] for r in rows],
                [r[key] for r in rows],
                color=c,
                linestyle="--" if mi >= 4 else "-",
                label=NAMES.get(m["job_id"], m["name"]),
            )
    for row in range(2):
        axes[row, col].set(
            xlabel="First-pass optimizer step",
            ylabel="Absolute H1024 RKL" if row == 0 else "RKL / initial RKL",
            yscale="log",
            title=family,
            xlim=(0, 320),
        )
    axes[0, col].legend(fontsize=7, loc="lower left", bbox_to_anchor=(0, 1.12), ncol=2, frameon=False)
    axes[0, col].set_ylim(0.0004, 0.4)
    axes[1, col].set_ylim(0.015, 1.3)
save(
    fig,
    "h1024-dynamics",
    "14 conditions; first pass only. DeepCoder and Light stop at 64; all other series stop at 320.",
)
