#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""One renderer for the maintained precision report; measured data only.

Semantic ledger: color identifies compute precision in convergence plots;
attention plots compare actual conditional VJPs with final-cast error. Every
curve is one independent restore, without smoothing or invented uncertainty.
Geometry ledger: x means optimizer updates, not outer steps or wall time.
Objective ratios use each arm's own initial score. Fixed-motion points share
one saved update and measure displacement against each mode's own initial
policy; they are one scalar series on a categorical axis, with a log x scale.
"""

import argparse
import json
from pathlib import Path

from examples.mirroropd.research.core.paths import MIRROROPD
from examples.mirroropd.research.core.plotting import load_style


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-skill", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fv = load_style(args.figure_skill / "assets/mpl")

    root = MIRROROPD / "docs/experiments/teacher-transfer/optimizer-precision/mechanism"
    old = json.loads((root / "data/localization-20261027.json").read_text())
    new = json.loads((root / "data/opd-transfer-20261027.json").read_text())
    out = root / "figures/precision-report-20261027"
    out.mkdir(parents=True, exist_ok=True)
    fv.apply_rc("paper", fontsize=9)
    colors = {
        "bfloat16": fv.HUES["vcoral"],
        "bf16_weights_fp32_compute": fv.HUES["vteal"],
        "float32": fv.HUES["vviolet"],
    }
    names = {"bfloat16": "BF16", "bf16_weights_fp32_compute": "BF16 weights / FP32 compute", "float32": "FP32 model"}
    audits = {}

    def arm(data, run, **filters):
        found = [a for a in data["arms"] if a["run"] == run and all(a[k] == v for k, v in filters.items())]
        assert len(found) == 1, (run, filters)
        return found[0]

    def curve(a, column=1):
        key = {1: "distill/target_kl", 2: "distill/teacher_kl"}[column]
        values = np.asarray([[0, a["before"][key]]] + [[p[0], p[column]] for p in a["post_series"]])
        assert np.isfinite(values).all() and (values[:, 1] > 0).all()
        return values[:, 0], values[:, 1] / values[0, 1]

    def style(ax):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=fv.NEUTRALS["vmist"], alpha=0.4)
        ax.set_axisbelow(True)

    def save(fig, name, caption):
        fig.text(0.5, 0.012, caption, ha="center", color=fv.NEUTRALS["vink"], fontsize="small")
        fig.canvas.draw()
        audits[name] = fv.audit(fig, print_slot=fig.get_figwidth())
        for ext in ("png", "pdf"):
            fig.savefig(out / f"{name}.{ext}", dpi=200)
        plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(7.8, 3.8), sharex=True, sharey=True)
    for ax, runs, title in zip(
        axs,
        [("mdpo-canonical", "mdpo-canonical-fp32"), ("teacher-native-fixed", "teacher-canonical-fp32")],
        ["MDPO: eta=0.3", "Ordinary Full-KL: eta=1"],
        strict=True,
    ):
        for run in runs:
            a = arm(new, run, index=0)
            ax.plot(*curve(a), color=colors[a["precision"]], label=names[a["precision"]])
        fv.mark_neutral(ax.axhline(1, color=fv.NEUTRALS["baseline"], linestyle=":"))
        ax.set(xlabel="Optimizer updates", xlim=(0, 64), ylim=(0, 1.7), title=title)
        ax.set_xticks([0, 16, 32, 48, 64])
        style(ax)
    axs[0].set_ylabel("Objective KL / its initial value")
    h, labels = axs[0].get_legend_handles_labels()
    fig.legend(h, labels, loc="upper center", ncol=2, frameon=False, fontsize="small")
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.20, top=0.80, wspace=0.15)
    save(
        fig,
        "objective-comparison",
        "Repaired teacher; same state and SGD 0.003; ratios compare descent within each objective",
    )

    fig, axs = plt.subplots(1, 3, figsize=(8.8, 3.8), sharex=True, sharey=True)
    for ax, solver, title in zip(
        axs, ("adam_saved", "sgd", "muon"), ("Adam 1e-6", "SGD 0.003", "Muon+AdamW 1e-6"), strict=True
    ):
        for precision in colors:
            a = arm(old, "long-precision", solver=solver, precision=precision)
            ax.plot(*curve(a), color=colors[precision], label=names[precision], linewidth=1.1)
        fv.mark_neutral(ax.axhline(1, color=fv.NEUTRALS["baseline"], linestyle=":"))
        ax.set(xlabel="Optimizer updates", xlim=(0, 256), yscale="log", title=title)
        ax.set_xticks([0, 128, 256])
        style(ax)
    axs[0].set_ylabel("Target KL / initial (log scale)")
    h, labels = axs[0].get_legend_handles_labels()
    fig.legend(h, labels, loc="upper center", ncol=3, frameon=False, fontsize="small")
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.21, top=0.79, wspace=0.14)
    save(fig, "long-budget", "Historical teacher execution; frozen MDPO targets; each precision captures its own q")

    evaluations = old["forward_controls"]["evaluations"] + old["forward_followup"]["evaluations"]
    settings = [
        ("bfloat16", "Native BF16"),
        ("residual_stream_fp32", "FP32 residual stream only"),
        ("rope_table_preserved_fp32", "Preserved FP32 RoPE table"),
        ("attention_fp32_cast_back", "FP32 attention, BF16 output"),
        ("midstream_head_bf16", "FP32 midstream, BF16 final norm/head"),
        ("midstream_head_fp32", "FP32 midstream + final norm/head"),
        ("float32", "FP32 arithmetic, BF16-valued weights"),
    ]
    values = []
    for precision, _ in settings:
        found = [
            r["metrics"]["distill/mode_anchor_kl"]
            for r in evaluations
            if r["precision"] == precision and r["snapshot"] == "1e-07/1"
        ]
        assert found and max(found) - min(found) < 1e-12
        values.append(found[0])
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.scatter(values, range(len(values)), color=fv.HUES["vteal"], s=32)
    for i, value in enumerate(values):
        ax.annotate(
            f"{value:.3e}", (value, i), xytext=(8, 0), textcoords="offset points", va="center", fontsize="small"
        )
    ax.set_yticks(range(len(values)), [label for _, label in settings])
    ax.invert_yaxis()
    ax.set(xscale="log", xlim=(3e-8, 8e-4), xlabel="Policy displacement KL (nats, log scale)")
    style(ax)
    fig.subplots_adjust(left=0.40, right=0.98, bottom=0.18, top=0.97)
    save(fig, "fixed-displacement", "Same stored parameter update and BF16-valued weights; per-mode initial output")

    captured = old["attention_capture"]["rows"]
    total = sum(r["dq"]["error_energy"] for r in captured)
    observed, cast, energy = [], [], []
    for layer in range(28):
        rows = [r["dq"] for r in captured if r["layer"] == layer]
        ref, err = sum(r["reference_energy"] for r in rows), sum(r["error_energy"] for r in rows)
        observed.append(100 * (err / ref) ** 0.5)
        cast.append(100 * (sum(r["cast_only_error_energy"] for r in rows) / ref) ** 0.5)
        energy.append(100 * err / total)
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.7))
    axs[0].plot(range(28), observed, color=colors["bfloat16"], label="Native FA2 VJP")
    axs[0].plot(range(28), cast, color=fv.NEUTRALS["baseline"], label="FP64 reference cast to BF16")
    axs[0].set(yscale="log", ylabel="Relative dQ error (%)", title="Conditional VJP accuracy")
    axs[0].legend(frameon=False, fontsize="small", loc="upper right")
    axs[1].bar(range(28), energy, color=colors["bfloat16"])
    axs[1].set(ylabel="Share of dQ error energy (%)", ylim=(0, 105), title="Error concentration")
    for ax in axs:
        ax.set(xlabel="Transformer layer (zero-based)", xlim=(-0.5, 27.5))
        ax.set_xticks([0, 7, 14, 21, 27])
        style(ax)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.22, top=0.90, wspace=0.32)
    save(fig, "attention-gradients", "112 historical captures; activation-gradient error, not loss attribution")

    fig, axs = plt.subplots(1, 3, figsize=(8.8, 3.8), sharex=True, sharey=True)
    for ax, run, title in zip(
        axs,
        ("state-a", "adam-confirm", "state-b"),
        ("State A / SGD 0.003", "State A / Adam 1e-6", "State B / SGD 0.003"),
        strict=True,
    ):
        for a in new["arms"]:
            if a["run"] == run:
                ax.plot(*curve(a, 2), color=colors[a["precision"]], label=names[a["precision"]], linewidth=1.15)
        ax.set(xlabel="Optimizer updates", xlim=(0, 128), yscale="log", title=title)
        ax.set_xticks([0, 64, 128])
        style(ax)
    axs[0].set_ylabel("Teacher KL / initial (log scale)")
    h, labels = axs[0].get_legend_handles_labels()
    fig.legend(h, labels, loc="upper center", ncol=3, frameon=False, fontsize="small")
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.21, top=0.79, wspace=0.14)
    save(fig, "opd-transfer", "Ordinary Full-KL; pre-fix teacher held fixed; equal budget within each panel")
    (out / "figure-audit.json").write_text(json.dumps(audits, indent=2) + "\n")
    print(json.dumps(audits, indent=2))


if __name__ == "__main__":
    main()
