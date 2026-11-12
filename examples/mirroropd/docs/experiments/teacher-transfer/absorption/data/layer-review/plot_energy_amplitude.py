"""Recompute update amplitude and energy fractions from adjacent frozen evidence.

Ledger: rows are 16 model conditions; teal OPD, orange teacher. Each matrix has
its own rank-16 truncation. Pooled concentration sums those energies, not an SVD
of concatenated weights and not the median concentration used in layer plots.
Shared y-axis model labels apply to all three panels; missing-axis-label is
therefore exempted. PDF width is 13 inches, exempting the 5.5-inch print slot.
All ratios compare one pair internally. Absolute norms use stored coordinates
of all seven projections, and must not rank different backbones by raw scale.
"""

import gzip
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.environ["FIGURE_VIZ_MPL"])
import fvstyle as fv

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent.parent / "figures/layer-review"
fv.apply_rc("paper")
colors = fv.series_colors("paper", 4)
data = json.loads(gzip.decompress((ROOT / "layer-data.json.gz").read_bytes()))
models = json.loads(gzip.decompress((ROOT / "module-angle-analysis.json.gz").read_bytes()))["models"]
names = [
    "R1 DeepCoder (64)",
    "R1 DeepScaleR",
    "R1 JustRL",
    "R1 MiniMath SFT",
    "R1 BroRL",
    "R1 Nemotron",
    "R1-7B Light SFT (64)",
    "Qwen GRPO",
    "Qwen SFT thinking",
    "Qwen e3",
    "Qwen SFT nonthinking",
    "Llama CoReward-I",
    "Llama GT",
    "Llama arithmetic SFT",
    "MiMo SFT to RL",
    "MiMo branch",
]
rows = []
for m, name in zip(models, names, strict=True):
    result = dict(m, name=name)
    for stage in ["teacher", "endpoint"]:
        mats = [x for x in data["matrices"] if x["job_id"] == m["job_id"] and x["stage"] == stage]
        total = sum(x["update_energy"] for x in mats)
        top = sum(x["update_energy"] * x["top16"] for x in mats)
        result[stage + "_energy"] = total
        result[stage + "_top16_energy"] = top
        result[stage + "_pooled_top16"] = top / total
    a = m["student_teacher_norm_ratio"]
    c = m["update_cosine"]
    result["residual_from_amplitude_angle"] = 1 + a * a - 2 * a * c
    result["residual_identity_error"] = abs(result["residual_from_amplitude_angle"] - m["parameter_residual_energy"])
    result["student_relative_update"] = a * m["teacher_relative_update"]
    rows.append(result)
assert max(x["residual_identity_error"] for x in rows) < 1e-8
(ROOT / "energy-amplitude-analysis.json").write_text(
    json.dumps(
        {
            "protocol": (
                "All-layer seven projection matrices; per-matrix rank16 energies pooled by summing; stored coordinates."
            ),
            "models": rows,
        },
        indent=2,
    )
    + "\n"
)
y = np.arange(len(rows))
audits = {}


def save(fig, name, caption):
    fig.text(0.5, 0.005, caption, ha="center", fontsize=8, color=fv.NEUTRALS["baseline"])
    fig.get_layout_engine().set(rect=(0, 0.045, 1, 0.95))
    fig.canvas.draw()
    audits[name] = [str(p) for p in fv.audit(fig, ignore=["print-fit", "missing-axis-label"])]
    fig.savefig(OUT / (name + ".png"), dpi=180)
    fig.savefig(OUT / (name + ".pdf"))
    plt.close(fig)


fig, axes = plt.subplots(1, 3, figsize=(13, 8), sharey=True, layout="constrained")
axes[0].set_yticks(y, names)
for ax in axes:
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.18)
for j, stage in enumerate(["teacher", "endpoint"]):
    axes[0].barh(
        y + (j - 0.5) * 0.3,
        [r[stage + "_pooled_top16"] * 100 for r in rows],
        height=0.29,
        color=colors[1 - j],
        label="Teacher" if j == 0 else "OPD",
    )
axes[0].set(xlabel="Pooled top-16 fraction (%)", xlim=(0, 100), title="1. Share of each update")
fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=2, fontsize=9)
for ax, key, title in [
    (axes[1], "student_teacher_norm_ratio", "2. Total energy: OPD / teacher"),
    (axes[2], "absolute_top16_energy_ratio", "3. Top-16 energy: OPD / teacher"),
]:
    vals = [r[key] ** 2 if key == "student_teacher_norm_ratio" else r[key] for r in rows]
    ax.scatter(vals, y, color=colors[0], s=32)
    ax.axvline(1, color=fv.NEUTRALS["baseline"], lw=1, ls="--")
    ax.set(xscale="log", xlim=(1e-4, 12), xlabel="Energy ratio (log scale)", title=title)
save(
    fig,
    "energy-fraction-versus-absolute",
    "Per-matrix rank-16 energies summed across all layers; dashed line = equal to teacher.",
)

fig, axes = plt.subplots(1, 3, figsize=(13, 8), sharey=True, layout="constrained")
axes[0].set_yticks(y, names)
for ax in axes:
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.18)
for stage, color, label in [("teacher", colors[1], "Teacher"), ("student", colors[0], "OPD")]:
    axes[0].scatter([r[stage + "_update_norm"] for r in rows], y, color=color, s=30, label=label)
axes[0].set(xscale="log", xlabel="Update Frobenius norm", title="1. Absolute update amplitude")
fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=2, fontsize=9)
axes[1].scatter([r["student_teacher_norm_ratio"] for r in rows], y, color=colors[0], s=30)
axes[1].axvline(1, color=fv.NEUTRALS["baseline"], ls="--", lw=1)
axes[1].set(xscale="log", xlabel="OPD norm / teacher norm", title="2. Amplitude within each pair")
axes[2].scatter([r["remaining"] * 100 for r in rows], y, color=colors[0], s=30)
axes[2].set(xlabel="Heldout RKL / initial RKL (%)", xlim=(0, 52), title="3. Output matching")
save(
    fig,
    "update-amplitude-and-risk",
    "Seven projection matrices per layer; heldout Math/Code/IF, initial prefix, 12 questions/domain, H128.",
)
(ROOT / "energy-amplitude-figure-audit.json").write_text(json.dumps(audits, indent=2) + "\n")
for r in rows:
    print(
        r["name"],
        *(
            round(r[k], 5)
            for k in [
                "teacher_energy",
                "endpoint_energy",
                "teacher_pooled_top16",
                "endpoint_pooled_top16",
                "teacher_top16_energy",
                "endpoint_top16_energy",
            ]
        ),
    )
print("audit", audits)
