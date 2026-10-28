# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Frozen E54 diagnostics: length is per rollout batch; stop mass is per prefix.

Semantic ledger: E41/4B is teal, E54/30B violet, initial Base gray.
Geometry: equal panels; matched outer coordinates; probe IDs are categorical.
No interpolation across probe cases, smoothing, or missing-data imputation.
"""

import argparse
import json
from pathlib import Path

from ..common.plotting import load_style


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fv = load_style()
    fv.apply_rc("paper", fontsize=8)
    args.out.mkdir(parents=True, exist_ok=True)
    audits = {}

    def save(fig, name, caption):
        fig.text(0.5, 0.018, caption, ha="center", fontsize=7, color=fv.NEUTRALS["baseline"])
        audits[name] = fv.audit(fig)
        for ext in ["png", "pdf"]:
            fig.savefig(args.out / f"{name}.{ext}", dpi=240)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.8))
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.29, top=0.93, wspace=0.4)
    for name, path, color in [
        ("Base ← 4B (E41)", args.reference, fv.HUES["vteal"]),
        ("Base ← 30B (E54)", args.data / "metrics.jsonl", fv.HUES["vviolet"]),
    ]:
        with path.open() as stream:
            rows = [json.loads(line)["data"] for line in stream]
        rows = [r for r in rows if "rollout/response_length/mean" in r and r["step/outer"] <= 29]
        assert [r["step/outer"] for r in rows] == list(range(1, 30))
        x = [r["step/outer"] for r in rows]
        axes[0].plot(x, [r["rollout/response_length/mean"] / 1000 for r in rows], color=color, label=name)
        axes[1].plot(x, [r["rollout/finish_reason/length_count"] / 64 * 100 for r in rows], color=color)
    for ax, ylabel, limits in zip(
        axes, ["Mean length (k tokens)", "Length-capped (%)"], [(0, 7.5), (0, 100)], strict=True
    ):
        ax.set(xlabel="Outer step", ylabel=ylabel, ylim=limits)
        ax.grid(axis="y", alpha=0.4)
    fig.legend(
        *axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.06), ncol=2, frameon=False
    )
    save(fig, "early-length", "Same prompt batches, different sampled trajectories; response cap 7,168.")

    models = [
        ("initial_student", "Initial Base", "vgray"),
        ("teacher_4b", "4B teacher", "vteal"),
        ("teacher_30b", "30B teacher", "vviolet"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.9))
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.29, top=0.87, wspace=0.32)
    for ax, category, title in zip(
        axes,
        ["archived_terminal", "e41_correct_end"],
        ["Archived natural ends (n=10)", "Verifier-correct ends (n=7)"],
        strict=True,
    ):
        for offset, (name, label, hue) in zip([-0.18, 0, 0.18], models, strict=True):
            data = json.loads((args.data / "probes" / f"{name}.json").read_text())
            rows = [r for r in data if r["category"] == category and r["variant"] == "training_prompt"]
            ax.scatter(
                [i + 1 + offset for i in range(len(rows))],
                [r["p_stop"] * 100 for r in rows],
                color=fv.HUES[hue],
                label=label,
                s=15,
            )
        ax.set(
            title=title,
            xlabel="Probe case",
            ylabel="Total stop probability (%)",
            ylim=(-5, 105),
            xticks=range(1, len(rows) + 1),
        )
        ax.grid(axis="y", alpha=0.4)
    handles = [Line2D([], [], color=fv.HUES[hue], marker="o", linestyle="", label=label) for _, label, hue in models]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.06), ncol=3, frameon=False)
    save(fig, "stop-probes", "Frozen models on identical selected prefixes; not population estimates.")
    (args.out / "audit.json").write_text(json.dumps(audits, indent=2) + "\n")
    print(json.dumps(audits))


if __name__ == "__main__":
    main()
