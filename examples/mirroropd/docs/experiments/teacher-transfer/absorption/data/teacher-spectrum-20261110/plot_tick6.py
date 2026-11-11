"""Auditable full-layer and response evidence, preserving older figure snapshots."""

import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2] / "dotfiles/.agents/skills/figure-viz/assets/mpl"))
import fvstyle as fv  # noqa: E402

fv.apply_rc("paper", fontsize=10)
OUT = ROOT / "figures-review-006"
OUT.mkdir(exist_ok=True)
DATA = json.loads((ROOT / "spectral-analysis-006.json").read_text())
LABELS = ["qwen-grpo", "qwen-sft", "r1-justrl"]
NAMES = ["Qwen GRPO", "Qwen SFT", "R1 JustRL"]
COLORS = fv.series_colors("paper", 4)
AUDITS = {}


def finish(fig, name):
    fig.canvas.draw()
    issues = fv.audit(fig, print_slot=float(fig.get_figwidth()))
    AUDITS[name] = [str(x) for x in issues]
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT / f"{name}.{ext}", dpi=180)
    plt.close(fig)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    for ax, label, name in zip(axes, LABELS, NAMES, strict=True):
        stages = DATA["weights"][label]
        xs = list(stages)
        for role, color in [("teacher", COLORS[0]), ("opd", COLORS[1])]:
            ax.plot(
                range(len(xs)),
                [stages[s]["roles"][role]["top16_median"] * 100 for s in xs],
                color=color,
                marker="o",
                label=role,
            )
        ax.set(
            title=name,
            xlabel="Student training step",
            ylabel="Median top-16 energy (%)",
            ylim=(0, 65),
            xticks=range(len(xs)),
            xticklabels=xs,
        )
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=2)
    finish(fig, "full-layer-concentration")

    summary = {}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for ax, label, name in zip(axes, LABELS, NAMES, strict=True):
        rows = json.loads((ROOT / f"response-probes-v3/{label}-full.json").read_text())["rows"]
        summary[label] = {}
        for i, direction in enumerate(["learned", "diag", "off", "null"]):
            ratios = np.array(
                [
                    r["directions"][direction]["unit_response_energy"]
                    / r["directions"]["gaussian"]["unit_response_energy"]
                    for r in rows
                ]
            )
            cos = np.array([r["directions"][direction]["request_cosine"] for r in rows])
            summary[label][direction] = dict(
                median_energy_ratio=float(np.median(ratios)),
                median_cosine=float(np.median(cos)),
                n=len(rows),
                positive_alignment=int((cos > 0).sum()),
            )
            # Paired site/layer ratios; no independent-sample confidence interval.
            ax.scatter(np.full(len(ratios), i), ratios, color=COLORS[i], alpha=0.25, s=10, label="_nolegend_")
            ax.scatter(i, np.median(ratios), color=COLORS[i], marker="D", s=50, label=direction)
        ax.axhline(1, color=fv.NEUTRALS["baseline"], linestyle="--", linewidth=1)
        ax.set(
            title=name,
            xlabel="Unit parameter direction",
            ylabel="Response energy / Gaussian",
            yscale="log",
            ylim=(0.001, 1e6),
            xticks=range(4),
            xticklabels=["learned", "diag", "off", "null"],
        )
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=True))
    fig.legend(list(unique.values()), list(unique), loc="outside upper center", ncol=4)
    finish(fig, "unit-response-energy")
    (OUT / "response-summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "audit.json").write_text(json.dumps(AUDITS, indent=2))
    sources = [
        ROOT / "spectral-analysis-006.json",
        *[ROOT / f"response-probes-v3/{k}-full.json" for k in LABELS],
        Path(__file__),
    ]
    (OUT / "provenance.json").write_text(
        json.dumps({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, indent=2)
    )


if __name__ == "__main__":
    main()
