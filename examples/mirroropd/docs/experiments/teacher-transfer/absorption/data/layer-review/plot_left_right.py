"""Plot both full-weight subspace sides and decompose parameter residual.

Ledger: one row per model condition; left/right points are median matrix-level
mean principal angles in degrees, rank16 of full W (not update delta W).
Residual records use a=OPD/teacher update norm, c=update cosine, and verify
R=1+a*a-2*a*c. Shared condition y labels apply to both panels. The standalone
13-inch PDF is not intended for a 5.5-inch print slot.
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
data = json.loads(gzip.decompress((ROOT / "module-angle-analysis.json.gz").read_bytes()))
models = json.loads((ROOT / "energy-amplitude-analysis.json").read_text())["models"]
rows = []
for model in models:
    a, c = model["student_teacher_norm_ratio"], model["update_cosine"]
    row = dict(
        job_id=model["job_id"],
        name=model["name"],
        amplitude_ratio=a,
        update_cosine=c,
        added_squared_norm=a * a,
        subtracted_alignment=2 * a * c,
        residual=1 + a * a - 2 * a * c,
        teacher_parallel_coefficient=a * c,
        orthogonal_energy=a * a * (1 - c * c),
    )
    for transition in ["base-to-teacher", "base-to-endpoint"]:
        for side in ["left", "right"]:
            record = next(
                x
                for x in data["angle_summary"]
                if x["job_id"] == model["job_id"]
                and x["rank"] == 16
                and x["transition"] == transition
                and x["side"] == side
            )
            row[transition + "_" + side] = record["median_mean_angle_deg"]
    rows.append(row)
fig, axes = plt.subplots(1, 2, sharey=True, figsize=(13, 8), layout="constrained")
y = np.arange(len(rows))
axes[0].set_yticks(y, [r["name"] for r in rows])
axes[0].invert_yaxis()
for ax, transition, title in zip(
    axes, ["base-to-teacher", "base-to-endpoint"], ["Initial to teacher", "Initial to OPD"], strict=True
):
    for side, offset, marker, color in [("left", -0.08, "o", colors[0]), ("right", 0.08, "x", colors[1])]:
        ax.scatter([r[transition + "_" + side] for r in rows], y + offset, marker=marker, color=color, s=38, label=side)
    ax.set(xscale="log", xlabel="Median mean principal angle (degrees)", title=title)
    ax.grid(axis="x", alpha=0.18)
fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=2)
fig.get_layout_engine().set(rect=(0, 0.045, 1, 0.95))
fig.text(
    0.5,
    0.005,
    "Full-weight top-16 subspaces; matrix mean angles, then median across all seven projections per layer.",
    ha="center",
    fontsize=8,
    color=fv.NEUTRALS["baseline"],
)
fig.canvas.draw()
audit = [str(p) for p in fv.audit(fig, ignore=["print-fit", "missing-axis-label"])]
fig.savefig(OUT / "left-right-full-weight-angles.png", dpi=180)
fig.savefig(OUT / "left-right-full-weight-angles.pdf")
(ROOT / "left-right-residual-analysis.json").write_text(json.dumps(rows, indent=2) + "\n")
(ROOT / "left-right-figure-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
print("audit", audit)
for row in rows:
    print(
        row["name"],
        *(round(row[k], 5) for k in ["amplitude_ratio", "added_squared_norm", "subtracted_alignment", "residual"]),
    )
