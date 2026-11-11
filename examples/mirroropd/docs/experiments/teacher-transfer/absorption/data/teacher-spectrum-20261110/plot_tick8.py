"""Inherited directional sensitivity and nonmonotone task retention under SVD compression."""

import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2] / "dotfiles/.agents/skills/figure-viz/assets/mpl"))
import fvstyle as fv  # noqa: E402


def save(fig, name, sources):
    out = ROOT / "figures-review-008"
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
    path = ROOT / "response-base-analysis-008.json"
    base = json.loads(path.read_text())
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), layout="constrained", sharey=True)
    for ax, (label, pair) in zip(axes, base["pairs"].items(), strict=True):
        for i, (direction, color) in enumerate(zip(["learned", "off", "null"], colors, strict=True)):
            d = pair["directions"][direction]
            ax.plot(
                [0, 1],
                [d["early_energy_over_gaussian_median"], d["final_energy_over_gaussian_median"]],
                marker="o",
                color=color,
                label=direction,
            )
        ax.axhline(1, color=fv.NEUTRALS["baseline"], linestyle="--", linewidth=1)
        ax.set(
            title=label,
            xlabel="Jacobian checkpoint",
            ylabel="Median energy / Gaussian",
            xticks=[0, 1],
            xticklabels=["base", "endpoint"],
            yscale="log",
            ylim=(0.7, 1000),
            xlim=(-0.15, 1.15),
        )
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=3)
    save(fig, "base-direction-response", [path])
    path = ROOT / "top-and-rollout-stratified-008.json"
    data = json.loads(path.read_text())["top"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), layout="constrained", sharey=True)
    for ax, (label, pair) in zip(axes, data.items(), strict=True):
        for domain, color in zip(["MATH-500", "AMC23"], colors, strict=False):
            values = [pair[k][domain]["correct"] / 16 * 100 for k in ["top1", "top5", "endpoint"]]
            ax.plot(range(3), values, marker="o", color=color, label=domain)
        ax.set(
            title=f"Qwen {label.upper()}",
            xlabel="Retained update singular components",
            ylabel="Accuracy (%)",
            xticks=range(3),
            xticklabels=["top 1%", "top 5%", "full"],
            ylim=(0, 100),
            xlim=(-0.2, 2.2),
        )
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=2)
    save(fig, "top-rank-task-readout", [path])
    path = ROOT / "top-and-rollout-matched-008.json"
    data = json.loads(path.read_text())["native_rollout_risk"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), layout="constrained", sharey=True)
    for ax, pair in zip(axes, ["qwen-grpo", "qwen-sft"], strict=True):
        banks = [b for b in data if b["pair"] == pair]
        for lane, color in zip(["endpoint", "patch-off-b1", "random-frames-seed0"], colors, strict=True):
            values = [b["groups"]["all"]["means"][lane] for b in banks]
            ax.plot(range(4), values, marker="o", color=color, label=lane)
        ax.set(
            title=pair,
            xlabel="Trajectory origin / window",
            ylabel="Mean reverse KL (matched, H128)",
            xticks=range(4),
            xticklabels=["end / first", "end / tail", "off / first", "off / tail"],
            yscale="log",
            ylim=(0.0003, 1),
            xlim=(-0.2, 3.2),
        )
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=3)
    save(fig, "new-trajectory-risk", [path])
    source = ROOT / "expanded-generation-complete-008.json"
    audit_source = ROOT / "supplemented-generation-analysis-008.json"
    canonical = json.loads(source.read_text())["pairs"]
    supplemented = json.loads(audit_source.read_text())["expanded72"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), layout="constrained", sharey=True)
    for ax, pair in zip(axes, ["rl", "sft"], strict=True):
        for data, label, color, marker, style in [
            (canonical, "canonical", colors[0], "o", "--"),
            (supplemented, "+ explicit ####", colors[1], "s", "-"),
        ]:
            values = [
                sum(d["correct"] for d in data[pair]["conditions"][lane].values()) / 72 * 100
                for lane in ["endpoint", "patch-off-b1", "teacher", "top1"]
            ]
            ax.plot(range(4), values, label=label, color=color, marker=marker, linestyle=style)
        ax.set(
            title=f"Qwen {pair.upper()}: 72 disjoint prompts",
            xlabel="Predeclared model condition",
            ylabel="Correct answers (%)",
            xticks=range(4),
            xticklabels=["endpoint", "off patch", "teacher", "top 1%"],
            ylim=(0, 100),
            xlim=(-0.2, 3.2),
        )
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=2)
    save(fig, "expanded-task-and-format", [source, audit_source])


if __name__ == "__main__":
    main()
