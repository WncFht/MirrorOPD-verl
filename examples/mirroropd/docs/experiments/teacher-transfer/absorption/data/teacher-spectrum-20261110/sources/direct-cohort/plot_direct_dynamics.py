"""Absolute and normalized risk dynamics for continuous parents.

Ledger: a line is one historical teacher condition, panels are model families;
colors distinguish teachers locally, solid RL / dashed SFT. Upper axes retain
absolute risk; lower axes divide by the same line's initial risk. MiMo uses its
original macro teacher-bank protocol, not historical H1024 normalization.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(
    0,
    str(
        Path("/mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/fht_data")
        / "dotfiles/.agents/skills/figure-viz/assets/mpl"
    ),
)
import fvstyle as fv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    inv = json.loads((args.root / "direct-pairs-v1/inventory.json").read_text())
    runs = json.loads((args.root / "spectrum-expanded-v4/explorer-data.json").read_text())["runs"]
    selected = {
        "R1": ["deepscaler", "justrl", "minimath"],
        "Qwen": ["grpo-kl", "e3", "sft-gsm8k"],
        "Llama": ["coreward-i", "gt", "arithmetic-full"],
        "MiMo": ["MiMo-SFT", "MiMo-Zero", "MiMo-RL-from-SFT"],
    }
    rows = []
    for c in inv["conditions"]:
        if c["teacher"] not in selected[c["family"]] or c["student"] == "q3-1.7-nonthinking":
            continue
        if c["family"] == "MiMo":
            values = defaultdict(list)
            for v in c["curves"]:
                if v["bank"] == "public_teacher":
                    values[int(v["optimizer_updates"])].append((v["domain"], float(v["rkl"])))
            line = []
            for step, vs in sorted(values.items()):
                allvalues = [v for d, v in vs if d == "all"]
                line.append(dict(step=step, rkl=float(np.mean(allvalues or [v for _, v in vs]))))
        else:
            run = next(r for r in runs if r["run_id"] == c["run_id"])
            line = run["panels"]["All held-out"]
        for v in line:
            if v["step"] <= 320:
                rows.append(
                    dict(
                        family=c["family"],
                        teacher=c["teacher"],
                        method=c["label"],
                        step=v["step"],
                        rkl=v["rkl"],
                        remaining=v["rkl"] / line[0]["rkl"],
                    )
                )
    df = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output / "dynamics.csv", index=False)
    fv.apply_rc("paper")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), layout="constrained")
    colors = fv.series_colors("paper", 3)
    for col, family in enumerate(selected):
        for color, teacher in zip(colors, selected[family], strict=True):
            g = df[(df.family == family) & (df.teacher == teacher)].sort_values("step")
            style = "--" if g.iloc[0].method == "SFT" else "-"
            for row, value in enumerate(["rkl", "remaining"]):
                axes[row, col].plot(g.step, g[value], color=color, linestyle=style, label=teacher)
        for row in range(2):
            axes[row, col].set(
                yscale="log", xlabel="OPD optimizer updates", ylabel="Absolute RKL" if row == 0 else "RKL / initial RKL"
            )
        axes[0, col].set_title(family)
        axes[0, col].legend(loc="upper center", bbox_to_anchor=(0.5, 1.38), fontsize=8)
    fig.text(
        0.5,
        0.006,
        "Continuous student to teacher pairs; solid RL, dashed SFT; MiMo preserves original macro-bank protocol",
        ha="center",
        fontsize=8,
        color=fv.NEUTRALS["baseline"],
    )
    for ax in axes[0]:
        ax.set_ylim(0.0004, 0.7)
    for ax in axes[1]:
        ax.set_ylim(0.015, 1.3)
    fig.set_layout_engine("constrained", rect=(0, 0.06, 1, 0.93))
    fig.canvas.draw()
    problems = [str(p) for p in fv.audit(fig, print_slot=16)]
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(args.output / ("continuous-dynamics." + ext), dpi=180)
    (args.output / "dynamics-audit.json").write_text(json.dumps(problems, indent=2))


if __name__ == "__main__":
    main()
