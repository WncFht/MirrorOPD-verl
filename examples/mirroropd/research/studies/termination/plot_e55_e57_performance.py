# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Plot frozen E55-E57 timing and CPU operator-path audits.

Semantic ledger: time bars partition measured complete-outer training and its
recorded validation. Token bars partition validation output by finish reason.
The operator panel counts logical large tensor results, not CUDA launches or
HBM traffic. Geometry ledger: zero-based bars, comparable within each panel;
covered outer ranges are explicit, so totals are not equal-budget rankings.
"""

import argparse
import json
from pathlib import Path

from examples.mirroropd.research.core.plotting import load_style


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.data.read_text())
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fv = load_style()
    fv.apply_rc("paper", fontsize=12)
    runs = [r for r in data["runs"] if r["validation"]]
    fig, axes = plt.subplots(1, 2, figsize=(9, 5.5))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.38, top=0.78, wspace=0.34)
    for i, r in enumerate(runs):
        s = r["sums"]
        actor = s["actor_seconds"] / 3600
        other = (s["train_seconds"] - s["actor_seconds"]) / 3600
        val = s["validation_seconds"] / 3600
        bottom = 0
        for label, value, hue in [
            ("Actor updates", actor, "vteal"),
            ("Other training", other, "vgray"),
            ("Validation", val, "vorange"),
        ]:
            axes[0].bar(i, value, bottom=bottom, width=0.48, color=fv.HUES[hue], label=label if i == 0 else None)
            bottom += value
        axes[0].text(i, bottom + 0.15, f"{bottom:.2f} h", ha="center", va="bottom")
        v = r["validation"][0]
        capped = v["capped_output_tokens"] / 1e6
        stopped = (v["output_tokens"] - v["capped_output_tokens"]) / 1e6
        axes[1].bar(i, stopped, width=0.48, color=fv.HUES["vgray"], label="Stopped output" if i == 0 else None)
        axes[1].bar(
            i, capped, bottom=stopped, width=0.48, color=fv.HUES["vcoral"], label="Capped output" if i == 0 else None
        )
        axes[1].text(
            i, capped + stopped + 2, f"{100 * v['capped_token_fraction']:.1f}% capped", ha="center", va="bottom"
        )
    axes[0].set(
        title="Covered wall time",
        ylabel="Hours",
        ylim=(0, 11),
        xticks=[0, 1],
        xticklabels=["E56: 19 outers", "E57: 18 outers"],
    )
    axes[1].set(
        title="One outer-10 validation",
        ylabel="Output tokens (millions)",
        ylim=(0, 135),
        xticks=[0, 1],
        xticklabels=["E56", "E57"],
    )
    for ax in axes:
        ax.set_xlabel("Received log coverage")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=1, frameon=False, fontsize=10)
    fig.suptitle("E55–E57 latency: validation dominates the received wall time", y=0.96)
    fig.text(
        0.5,
        0.015,
        "Validation: 630 questions x 16 samples; cap 31,744. Actor includes 16 reuse updates.\n"
        "No missing or in-progress outer was imputed; totals are not equal-budget speed rankings.",
        ha="center",
        fontsize=10,
    )
    fig.canvas.draw()
    issues = fv.audit(fig, ignore=["facet-ylim-mismatch"])
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "latency-audit.json").write_text(json.dumps(issues) + "\n")
    for ext in ("png", "pdf"):
        fig.savefig(args.out / f"latency.{ext}", dpi=200)
    print(issues)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.34, top=0.79)
    for offset, diagnostics, label, hue in [
        (-0.18, False, "Diagnostics off", "vgray"),
        (0.18, True, "Diagnostics on", "vteal"),
    ]:
        rows = [r for r in data["row_operator_audit"]["rows"] if r["diagnostics"] == diagnostics]
        ax.bar(
            [i + offset for i in range(3)],
            [r["large_result_ops"] for r in rows],
            width=0.34,
            label=label,
            color=fv.HUES[hue],
        )
    ax.set(
        xticks=[0, 1, 2],
        xticklabels=["0", ".5", "1"],
        xlabel="kappa",
        ylabel="Large tensor results per tile",
        ylim=(0, 600),
    )
    ax.grid(axis="y", alpha=0.2)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.55, 0.10), ncol=2, frameon=False)
    fig.suptitle("EOS MDPO path: repeated large-tensor operations", y=0.96)
    fig.text(
        0.5,
        0.02,
        "CPU operator-path audit; not CUDA timing, kernel counts, or measured HBM traffic.",
        ha="center",
        fontsize=8,
    )
    fig.canvas.draw()
    issues = fv.audit(fig)
    (args.out / "operators-audit.json").write_text(json.dumps(issues) + "\n")
    for ext in ("png", "pdf"):
        fig.savefig(args.out / f"operators.{ext}", dpi=200)
    print(issues)


if __name__ == "__main__":
    main()
