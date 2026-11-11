"""Research synthesis from frozen numerical evidence.

Ledger: heatmap cells are teacher-pair/domain Fisher residual energy ratios;
response points are median unit-parameter JVP energies divided by Gaussian;
deletion points and intervals are paired prompt accuracy differences, in pp.
All task panels contain the same expanded 72 prompts. No GPU computation.
Geometry: panels share scales only for identical metrics; percentage bars start
at zero; log response axes are explicitly marked. Each figure states its panel.
"""

import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2] / "dotfiles/.agents/skills/figure-viz/assets/mpl"))
import fvstyle as fv  # noqa: E402

OUT = ROOT / "figures-research-synthesis"
OUT.mkdir(exist_ok=True)
SOURCES = {}


def read(name):
    path = ROOT / name
    SOURCES[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


def save(fig, name, caption):
    fig.get_layout_engine().set(rect=(0, 0.09, 1, 0.91))
    fig.text(0.5, 0.015, caption, ha="center", fontsize=8, color=fv.NEUTRALS["baseline"])
    fig.canvas.draw()
    findings = [str(x) for x in fv.audit(fig, print_slot=11)]
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(OUT / f"{name}.{ext}", dpi=180, bbox_inches="tight")
    (OUT / f"{name}-audit.json").write_text(json.dumps(findings, indent=2))
    print(name, findings)
    plt.close(fig)


def main():
    fv.apply_rc("paper", fontsize=10)
    colors = fv.series_colors("paper", 4)
    final = read("final-scientific-data-010.json")
    pairs = ["qwen-grpo", "qwen-sft", "r1-justrl", "r1-polaris"]
    labels = ["Qwen RL", "Qwen SFT", "R1 JustRL", "Polaris"]
    values = np.array(
        [
            [
                final["pairs"][p]["stages"]["320"]["aggregates"][f"heldout/{d}"]["bands"]["all"]["missing"]
                for d in ["math", "code", "if"]
            ]
            for p in pairs
        ]
    )
    fig, ax = plt.subplots(figsize=(7.4, 3.4), layout="constrained")
    cmap = LinearSegmentedColormap.from_list("residual", ["white", colors[0]])
    im = ax.imshow(values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set(
        xticks=range(3),
        xticklabels=["Math", "Code", "Instruction following"],
        yticks=range(4),
        yticklabels=labels,
        xlabel="Historical diagnostic domain",
        ylabel="Teacher pair",
    )
    for i in range(4):
        for j in range(3):
            ax.text(j, i, f"{values[i, j] * 100:.1f}%", ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, label="Remaining Fisher residual / teacher energy")
    save(fig, "domain-residuals", "Endpoint step320; fixed calibration frame; energy-weighted heldout aggregation")

    controls = read("response-base-random-controls-008.json")
    endpoint = read("response-base-controls-analysis-008.json")
    fig, ax = plt.subplots(figsize=(8.6, 3.6), layout="constrained")
    for i, (label, pair) in enumerate(zip(labels[:3], pairs[:3], strict=True)):
        base = controls[pair]["final-learned"]
        trained = endpoint["pairs"][pair]["directions"]["learned"]["final_energy_over_gaussian_median"]
        ax.scatter(
            [i - 0.12], [base], color=colors[0], marker="o", label="Base: final learned direction" if i == 0 else None
        )
        ax.scatter(
            [i + 0.12], [trained], color=colors[1], marker="s", label="Endpoint: same direction" if i == 0 else None
        )
        random = [controls[pair][f"random_frames-{s}"] for s in range(3)]
        ax.scatter(
            [i - 0.1, i, i + 0.1],
            random,
            color=colors[2],
            marker="x",
            label="Base: same-spectrum random frames" if i == 0 else None,
        )
    ax.set(
        xticks=range(3),
        xticklabels=labels[:3],
        xlim=(-0.45, 2.45),
        ylim=(0.4, 1000),
        yscale="log",
        xlabel="Same-coordinate teacher pair",
        ylabel="Median JVP energy / Gaussian",
    )
    ax.axhline(1, color=fv.NEUTRALS["baseline"], linestyle=":", linewidth=1)
    fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncol=1)
    save(
        fig,
        "preexisting-direction-response",
        "64 questions per pair; 3 positions x 3 k_proj layers; directions selected from final checkpoint",
    )

    deletion = read("delete-gaussian-paired-009.json")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")
    conditions = ["sft-teacher", "sft-opd", "rl-teacher", "rl-opd"]
    names = ["SFT teacher", "SFT student", "RL teacher", "RL student"]
    for ax, compare, title, color in [
        (axes[0], "delete_off", "Actual deletion minus unedited", colors[0]),
        (axes[1], "deletion_vs_gaussian", "Actual deletion minus Gaussian", colors[0]),
    ]:
        entries = [deletion["conditions"][c]["supplemented"][compare]["all"] for c in conditions]
        means = np.array([v["paired_accuracy_difference"] for v in entries]) * 100
        intervals = np.array([v["paired_bootstrap_95"] for v in entries]) * 100
        ax.errorbar(
            means,
            range(4),
            xerr=[means - intervals[:, 0], intervals[:, 1] - means],
            color=color,
            fmt="o",
            capsize=4,
            label="Paired estimate and 95% interval",
        )
        ax.axvline(0, color=fv.NEUTRALS["baseline"], linestyle=":", linewidth=1)
        ax.set(
            title=title,
            yticks=range(4),
            yticklabels=names,
            ylim=(3.4, -0.4),
            xlim=(-15, 27),
            xlabel="Paired accuracy difference (percentage points)",
            ylabel="Edited model",
        )
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center")
    save(
        fig,
        "deletion-direction-specificity",
        "Same72 prompts; explicit-answer supplemented; paired bootstrap95%; one Gaussian seed, exploratory intervals",
    )

    expanded = read("supplemented-generation-analysis-008.json")["expanded72"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7), layout="constrained")
    for ax, pair, title in zip(axes, ["sft", "rl"], ["Qwen SFT", "Qwen RL"], strict=True):
        conditions = expanded[pair]["conditions"]
        values = [
            sum(v["correct"] for v in conditions[key].values())
            for key in ["endpoint", "patch-off-b1", "teacher", "top1"]
        ]
        ax.bar(range(4), values, color=[colors[0], colors[1], colors[2], colors[3]])
        ax.set(
            title=title,
            xticks=range(4),
            xticklabels=["Endpoint", "Missing off", "Teacher", "Top1%"],
            ylim=(0, 76),
            yticks=[0, 18, 36, 54, 72],
            xlabel="Fixed model condition",
            ylabel="Correct answers /72",
        )
        for i, value in enumerate(values):
            ax.text(i, value + 1, str(value), ha="center", va="bottom")
    save(
        fig,
        "task-content-comparison",
        "Disjoint expanded panel:48 MATH-500 +24 AMC23; greedy16K; explicit-final-answer supplementation",
    )
    (OUT / "provenance.json").write_text(
        json.dumps(SOURCES | {Path(__file__).name: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2)
    )
    (OUT / "plot-data.json").write_text(
        json.dumps(
            {"domain_missing": {p: final["pairs"][p]["stages"]["320"]["aggregates"] for p in pairs}},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
