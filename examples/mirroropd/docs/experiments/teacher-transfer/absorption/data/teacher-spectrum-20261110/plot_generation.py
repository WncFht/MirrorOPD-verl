"""Matched generation evidence; scores and truncation share the same task panel."""

import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2] / "dotfiles/.agents/skills/figure-viz/assets/mpl"))
import fvstyle as fv  # noqa: E402

fv.apply_rc("paper", fontsize=10)
COLORS = fv.series_colors("paper", 4)
OUT = ROOT / "figures-review-007"
OUT.mkdir(exist_ok=True)


def main():
    sources = {}
    values = {}
    for pair, directory in [("RL", "generation-readout-rl-v1"), ("SFT", "generation-readout-v2")]:
        values[pair] = {}
        for condition in ["endpoint", "patch-off-b1", "random-frames-seed0", "teacher"]:
            path = ROOT / directory / condition / condition / "records.jsonl"
            rows = [json.loads(x) for x in path.read_text().splitlines()]
            assert len(rows) == 32
            sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
            values[pair][condition] = {
                domain: dict(
                    n=sum(r["data_source"] == domain for r in rows),
                    correct=sum(r["data_source"] == domain and bool(r["acc"]) for r in rows),
                    truncated=sum(r["data_source"] == domain and r["finish_reason"] == "length" for r in rows),
                )
                for domain in ["MATH-500", "AMC23"]
            }
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout="constrained", sharey=True)
    for column, pair in enumerate(["RL", "SFT"]):
        for row, domain in enumerate(["MATH-500", "AMC23"]):
            ax = axes[row, column]
            for index, (condition, color) in enumerate(zip(values[pair], COLORS, strict=True)):
                v = values[pair][condition][domain]
                ax.bar(index, v["correct"] / v["n"] * 100, color=color, label=condition)
            ax.set(
                title=f"{pair}: {domain}",
                ylim=(0, 100),
                ylabel="Correct (%)",
                xlabel="Model condition",
                xticks=range(4),
                xticklabels=["student", "off patch", "random", "teacher"],
            )
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside upper center", ncol=4)
    fig.canvas.draw()
    audits = {"task_accuracy": [str(x) for x in fv.audit(fig, print_slot=10)]}
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT / f"task_accuracy.{ext}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout="constrained", sharey=True)
    for ax, pair in zip(axes, ["RL", "SFT"], strict=True):
        for index, (condition, color) in enumerate(zip(values[pair], COLORS, strict=True)):
            total = sum(v["truncated"] for v in values[pair][condition].values())
            ax.bar(index, total / 32 * 100, color=color, label=condition)
        ax.set(
            title=pair,
            ylim=(0, 100),
            ylabel="Truncated responses (%)",
            xlabel="Model condition",
            xticks=range(4),
            xticklabels=["student", "off patch", "random", "teacher"],
        )
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=4)
    fig.canvas.draw()
    audits["truncation"] = [str(x) for x in fv.audit(fig, print_slot=10)]
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT / f"truncation.{ext}", dpi=180)
    plt.close(fig)
    (OUT / "data.json").write_text(json.dumps(values, indent=2))
    (OUT / "audit.json").write_text(json.dumps(audits, indent=2))
    sources[Path(__file__).name] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT / "provenance.json").write_text(json.dumps(sources, indent=2))
    print(audits)


if __name__ == "__main__":
    main()
