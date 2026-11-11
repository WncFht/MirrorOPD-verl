"""Semantic ledger: teal=GRPO, orange=SFT; solid=heldout, dashed=calibration.
Geometry ledger: common normalized-risk axis; equal bar slots for each condition.
Digital native-width panels, reserved footer, no data labels or inset overlays.
"""

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = next(p for p in Path(__file__).resolve().parents if (p / "examples/mirroropd").is_dir())
ROOT = REPO / "runs/teacher-spectrum-followup-20261110"
sys.path.insert(0, str(REPO.parent / "dotfiles/.agents/skills/figure-viz/assets/mpl"))
import fvstyle as fv  # noqa: E402

fv.apply_rc("paper", fontsize=10)
OUT = REPO / "examples/mirroropd/docs/experiments/teacher-transfer/figures/teacher-spectrum-review-003"
OUT.mkdir(exist_ok=True)
raw = (ROOT / "progress-analysis.json").read_bytes()
data = json.loads(raw)
(OUT / "plot-data.json").write_bytes(raw)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
for index, (label, name, color) in enumerate(
    [
        ("qwen-grpo", "GRPO", fv.HUES["vteal"]),
        ("qwen-sft", "SFT", fv.HUES["vorange"]),
    ]
):
    values = []
    for key in ["top0.01-restore0", "top0.01-restore1", "top0.05-restore0", "top0.05-restore1"]:
        values.append(data["interventions"][f"{label}.intervention-{key}"]["heldout"]["gain_retained"])
    axes[0].bar(np.arange(4) + (index - 0.5) * 0.34, values, width=0.34, color=color, label=name)
    for split, ls in [("train", "--"), ("heldout", "-")]:
        denominator = data["pairs"][label]["native_risk"][f"{split}/all/s320"]["loss"]
        rows = [(0, 1.0)]
        for beta in [0.25, 0.5, 1]:
            key = f"{label}.intervention-patch-off-b{beta:g}"
            if key in data["interventions"]:
                rows.append((beta, data["interventions"][key][split]["loss"] / denominator))
        axes[1].plot(*zip(*rows, strict=True), color=color, ls=ls, marker="o", label=f"{name} {split}")
axes[0].set(
    xticks=range(4),
    xticklabels=["Top 1%", "Top 1%\nrescaled", "Top 5%", "Top 5%\nrescaled"],
    ylim=(0, 1.1),
    ylabel="Fraction of original KL gain retained",
    title="Student truncation: both are concentrated",
)
axes[1].set(
    xlabel="Missing off component coefficient",
    ylabel="KL / original student KL",
    title="Missing off patch: amplitude matters",
    ylim=(0, 1.15),
)
for ax in axes:
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.16), ncol=2)
fig.subplots_adjust(left=0.08, right=0.98, top=0.72, bottom=0.23, wspace=0.32)
fig.text(
    0.08,
    0.025,
    "Heldout fixed-bank full-vocabulary KL. SFT beta=1 is pending in this snapshot; no curve extrapolation.",
    fontsize=9,
    color=fv.NEUTRALS["baseline"],
)
fig.canvas.draw()
problems = fv.audit(fig, print_slot=11)
(OUT / "figure.audit.json").write_text(json.dumps(problems, indent=2))
for extension in ["png", "svg", "pdf"]:
    fig.savefig(OUT / f"spectral-interventions.{extension}", dpi=170)
(OUT / "provenance.json").write_text(
    json.dumps(
        dict(
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            input_sha256=hashlib.sha256(raw).hexdigest(),
            counts=data["counts"],
        ),
        indent=2,
    )
)
(OUT / "plot_review003.py").write_bytes(Path(__file__).read_bytes())
print(problems)
