"""Fixed-direction response ratios separate Jacobian change from direction change."""

import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2] / "dotfiles/.agents/skills/figure-viz/assets/mpl"))
import fvstyle as fv  # noqa: E402


def main():
    fv.apply_rc("paper", fontsize=10)
    colors = fv.series_colors("paper", 4)
    source = ROOT / "response-stage-analysis-007.json"
    data = json.loads(source.read_text())
    out = ROOT / "figures-review-007"
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), layout="constrained", sharey=True)
    for ax, (label, pair) in zip(axes, data["pairs"].items(), strict=True):
        for index, (name, color) in enumerate(zip(["learned", "diag", "off", "null"], colors, strict=True)):
            value = pair["directions"][name]["endpoint_over_early_energy_median"]
            ax.bar(index, value, color=color, label=name)
        ax.axhline(1, color=fv.NEUTRALS["baseline"], linestyle="--", linewidth=1)
        ax.set(
            title=label,
            xlabel="Fixed endpoint parameter direction",
            ylabel="Median energy: endpoint / early",
            ylim=(0, 1.3),
            xticks=range(4),
            xticklabels=["learned", "diag", "off", "null"],
        )
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=4)
    fig.canvas.draw()
    audit = [str(x) for x in fv.audit(fig, print_slot=12)]
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out / f"fixed-direction-stages.{ext}", dpi=180)
    (out / "stage-audit.json").write_text(json.dumps(audit, indent=2))
    (out / "stage-provenance.json").write_text(
        json.dumps(
            {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [source, Path(__file__)]},
            indent=2,
        )
    )
    print(audit)


if __name__ == "__main__":
    main()
