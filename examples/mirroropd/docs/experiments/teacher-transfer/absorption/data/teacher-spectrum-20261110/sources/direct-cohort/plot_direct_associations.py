"""Within-parent rank correlations: columns are distinct functional outcomes.

Ledger: rows are teacher-only metrics, cells exact measured correlations across
11 canonical teachers in parent strata of 5/3/3. Color is signed correlation;
no cell is declared statistically established after the 57-feature correction.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(
    0,
    str(
        Path("/mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/fht_data")
        / "dotfiles/.agents/skills/figure-viz/assets/mpl"
    ),
)
import fvstyle as fv

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root
    out = args.output or root / "direct-pairs-analysis-v4"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(root / "direct-pairs-analysis-v4/parent-matched-all-metrics.csv")
    metrics = {
        "norm_over_base": "Update / initial norm",
        "sketch_top16_median": "Five-module rank16 capture",
        "delta_effective_rank_fraction": "Exact k/v entropy rank fraction",
        "delta_energy_top1": "Exact k/v top1% energy",
        "entry_support_fraction": "Entry effective support",
        "sampled_kurtosis_median": "Sampled entry kurtosis",
        "layer_effective_fraction": "Layer effective coverage",
        "depth_center": "Depth energy center",
        "attention_fraction": "Attention energy fraction",
        "delta_base_cosine": "Update / initial cosine",
        "v_proj_energy_fraction": "V projection energy fraction",
        "o_proj_energy_fraction": "O projection energy fraction",
    }
    targets = ["final_K", "remaining", "late_log_drop"]
    matrix = df.pivot(index="feature", columns="target", values="parent_centered_rank_correlation").loc[
        list(metrics), targets
    ]
    fv.apply_rc("paper")
    fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")
    cmap = LinearSegmentedColormap.from_list("signed", [fv.HUES["vcoral"], fv.NEUTRALS["paper"], fv.HUES["vteal"]])
    image = ax.imshow(matrix, vmin=-1, vmax=1, cmap=cmap, aspect="auto")
    for row in range(len(matrix)):
        for col in range(3):
            ax.text(col, row, f"{matrix.iloc[row, col]:+.2f}", ha="center", va="center", fontsize=10)
    ax.set(
        xticks=range(3),
        xticklabels=["Final absolute RKL", "Remaining RKL fraction", "Late improvement"],
        yticks=range(len(matrix)),
        yticklabels=list(metrics.values()),
        xlabel="Functional outcome",
        ylabel="Teacher-only metric",
    )
    fig.colorbar(image, ax=ax, label="Within-parent centered rank correlation")
    fig.set_layout_engine("constrained", rect=(0, 0.065, 1, 0.93))
    fig.text(
        0.5,
        0.012,
        "11 teachers, fixed step320; 57 metrics tested per target; no BH q < .05",
        ha="center",
        fontsize=8,
        color=fv.NEUTRALS["baseline"],
    )
    fig.canvas.draw()
    audit = fv.audit(fig, print_slot=8)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out / ("parent-matched-metrics." + ext), dpi=180)
    (out / "parent-matched-figure-audit.json").write_text(json.dumps([str(p) for p in audit], indent=2))


if __name__ == "__main__":
    main()
