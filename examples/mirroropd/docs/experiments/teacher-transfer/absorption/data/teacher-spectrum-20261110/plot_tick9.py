"""Training absorbs teacher distributions while task behavior and formatting diverge."""

import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2] / "dotfiles/.agents/skills/figure-viz/assets/mpl"))
import fvstyle as fv  # noqa: E402


def save(fig, name, sources):
    out = ROOT / "figures-review-009"
    out.mkdir(exist_ok=True)
    fig.canvas.draw()
    audit = [str(x) for x in fv.audit(fig, print_slot=11)]
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out / f"{name}.{ext}", dpi=180)
    (out / f"{name}-audit.json").write_text(json.dumps(audit, indent=2))
    (out / f"{name}-provenance.json").write_text(
        json.dumps(
            {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [*sources, Path(__file__)]},
            indent=2,
        )
    )
    print(name, audit)
    plt.close(fig)


def main():
    fv.apply_rc("paper", fontsize=10)
    colors = fv.series_colors("paper", 3)
    source = ROOT / "stage-and-format-analysis-009.json"
    data = json.loads(source.read_text())
    progress = ROOT / "progress-analysis.json"
    risk = json.loads(progress.read_text())["pairs"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7), layout="constrained")
    for pair, color in zip(["sft", "grpo"], colors, strict=False):
        keys = ["base", ("sft" if pair == "sft" else "rl") + "-early", ("sft" if pair == "sft" else "rl") + "-endpoint"]
        values = [sum(v["correct"] for v in data["stages"][key]["supplemented"].values()) / 72 * 100 for key in keys]
        axes[0].plot([0, 64, 320], values, marker="o", color=color, label=pair.upper())
        values = [
            risk[f"qwen-{pair}"]["native_risk"][f"heldout/all/{stage}"]["remaining"] * 100
            for stage in ["base", "s64", "s320"]
        ]
        axes[1].plot([0, 64, 320], values, marker="o", color=color, label=pair.upper())
    axes[0].set(
        xlabel="OPD training step",
        ylabel="Correct answers / 72 (%)",
        ylim=(-3, 105),
        title="Same 72 generated-task prompts",
    )
    axes[1].set(
        xlabel="OPD training step",
        ylabel="Remaining native KL / base (%)",
        ylim=(-3, 105),
        title="Historical fixed bank, H1024",
    )
    for ax in axes:
        ax.set_xticks([0, 64, 320])
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=2)
    save(fig, "stage-task-versus-fit", [source, progress])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7), layout="constrained", sharey=True)
    for ax, metric, title in [
        (axes[0], "accuracy", "Explicit-answer supplemented accuracy"),
        (axes[1], "boxed", "Outputs containing boxed"),
    ]:
        for key, color, style in [("original", colors[0], "--"), ("with_instruction", colors[1], "-")]:
            values = []
            for lane in ["endpoint", "patch-off-b1", "teacher", "top1"]:
                entry = data["formatting"][lane]
                values.append(
                    (
                        sum(v["correct"] for v in entry[key].values())
                        if metric == "accuracy"
                        else entry["stats"][key]["boxed"]
                    )
                    / 72
                    * 100
                )
            ax.plot(range(4), values, marker="o", color=color, linestyle=style, label=key.replace("_", " "))
        ax.set(
            title=title,
            xlabel="Fixed SFT model condition",
            ylabel="Fraction of 72 prompts (%)",
            ylim=(-3, 105),
            xticks=range(4),
            xticklabels=["endpoint", "off patch", "teacher", "top 1%"],
            xlim=(-0.15, 3.15),
        )
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=2)
    save(fig, "format-instruction-intervention", [source])
    response_path = ROOT / "generated-response-analysis-009.json"
    response = json.loads(response_path.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7), layout="constrained")
    for direction, color in zip(["final-learned", "final-off", "final-null"], colors, strict=True):
        energies = [
            response["conditions"][label]["all"][direction]["median_energy_over_gaussian"]
            for label in ["sft-base", "sft-early", "sft-endpoint"]
        ]
        axes[0].plot([0, 64, 320], energies, marker="o", color=color, label=direction.replace("final-", ""))
        ratios = [
            1,
            response["stage_ratios"]["early_over_base"][direction],
            response["stage_ratios"]["endpoint_over_base"][direction],
        ]
        axes[1].plot([0, 64, 320], ratios, marker="o", color=color, label=direction.replace("final-", ""))
    axes[0].set(
        xlabel="SFT checkpoint step",
        ylabel="Median energy / Gaussian",
        yscale="log",
        ylim=(10, 1000),
        title="New generated prefixes: direction advantage",
    )
    axes[1].set(
        xlabel="SFT checkpoint step",
        ylabel="Median paired energy / base",
        ylim=(0, 1.1),
        title="Same final directions: absolute response shrinks",
    )
    for ax in axes:
        ax.set_xticks([0, 64, 320])
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=3)
    save(fig, "new-prefix-direction-stages", [response_path])


if __name__ == "__main__":
    main()
