#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Measured long-budget and precision-factor plots, without smoothing.

Semantic ledger: long curves are independently restored runs, each normalized
by its own initial target KL. Factor dots are the final measured 16-step score,
not uncertainty intervals. Fixed-motion dots reuse the same stored parameter
update and measure each mode against its own initial output. No synthetic data.
Geometry ledger: update axes cover 0..256; KL-ratio axes are dimensionless.
The fixed-motion axis is logarithmic. Colors mean precision policies in the
long plot and intervention families in the factor plot, with explicit legends.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-skill", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.figure_skill / "assets/mpl"))
    import fvstyle as fv

    root = Path(__file__).resolve().parents[1]
    data = json.loads((root / "data/localization-20261027.json").read_text())
    out = root / "figures/precision-localization-20261027"
    out.mkdir(parents=True, exist_ok=True)
    fv.apply_rc("paper", fontsize=9)
    colors = [fv.HUES[k] for k in ("vcoral", "vteal", "vviolet")]
    audit = {}

    def style(ax):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=fv.NEUTRALS["vmist"], alpha=0.4)
        ax.set_axisbelow(True)

    def save(fig, name, caption):
        fig.text(0.5, 0.012, caption, ha="center", fontsize="small", color=fv.NEUTRALS["vink"])
        fig.canvas.draw()
        audit[name] = fv.audit(fig, print_slot=fig.get_figwidth())
        for extension in ("png", "pdf"):
            fig.savefig(out / f"{name}.{extension}", dpi=200)
        plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(8.8, 3.8), sharex=True, sharey=True)
    names = ["BF16", "BF16 weights / FP32 compute", "FP32"]
    policies = ["bfloat16", "bf16_weights_fp32_compute", "float32"]
    solvers = ["adam_saved", "sgd", "muon"]
    for i, (ax, solver) in enumerate(zip(axs, solvers, strict=True)):
        for j, policy in enumerate(policies):
            matches = [
                a
                for a in data["arms"]
                if a["run"] == "long-precision" and a["solver"] == solver and a["precision"] == policy
            ]
            assert len(matches) == 1
            arm = matches[0]
            values = np.asarray([[0, arm["before"]["distill/target_kl"]]] + [p[:2] for p in arm["post_series"]])
            assert len(values) == 257 and (values[:, 1] > 0).all()
            ax.plot(values[:, 0], values[:, 1] / values[0, 1], color=colors[j], label=names[j], linewidth=1.1)
        fv.mark_neutral(ax.axhline(1, color=fv.NEUTRALS["baseline"], linestyle=":"))
        ax.set(yscale="log", xlim=(0, 256), xlabel="Optimizer updates")
        ax.set_xticks([0, 128, 256])
        ax.set_title(["Adam 1e-6", "SGD 0.003", "Muon+AdamW 1e-6"][i], loc="left")
        style(ax)
    axs[0].set_ylabel("Target KL / initial (log scale)")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize="small")
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.21, top=0.78, wspace=0.14)
    save(fig, "01-long-precision", "State A; same rounded master start; each precision freezes its own q")

    labels = {
        "bfloat16": "Native BF16",
        "gemm_no_reduced": "Disable reduced GEMM reduction",
        "attention_vjp32": "Native forward, FP32 attention VJP",
        "attention_vjp64": "Native forward, FP64 attention VJP",
        "attention_deterministic": "Deterministic native attention backward",
        "residual_stream_fp32": "FP32 residual stream only",
        "norm_inner_fp32": "FP32 inner norm outputs",
        "norm_vjp32": "Native norm forward, FP32 VJP",
        "rope_math_fp32": "FP32 RoPE multiply/add",
        "rope_table_math_fp32": "FP32 RoPE table requested (recast)",
        "rope_table_preserved_fp32": "Preserve FP32 RoPE table",
        "midstream_head_bf16": "FP32 midstream, BF16 final norm/head",
        "midstream_head_fp32": "FP32 midstream + final norm/head",
        "mlp_pointwise_fp32": "FP32 MLP activation/product",
        "mlp_internal_fp32": "FP32 MLP internal outputs",
        "attention_fp32_cast_back": "FP32 attention, BF16 output",
    }
    factor_arms = [a for a in data["arms"] if a["run"] == "factors" and a["lr"] > 0]
    fig, ax = plt.subplots(figsize=(8.3, 5.6))
    groups = {}
    for i, arm in enumerate(factor_arms):
        mode = arm["precision"]
        family = (
            "Backward"
            if mode.startswith("attention_vjp") or mode in ("attention_deterministic", "norm_vjp32")
            else "GEMM setting"
            if mode == "gemm_no_reduced"
            else "Baseline"
            if mode == "bfloat16"
            else "Forward"
        )
        color = {
            "Backward": colors[2],
            "Forward": colors[1],
            "GEMM setting": fv.HUES["vorange"],
            "Baseline": fv.NEUTRALS["baseline"],
        }[family]
        value = arm["after"]["distill/target_kl"] / arm["before"]["distill/target_kl"]
        groups.setdefault((family, color), []).append((value, i))
        ax.annotate(
            f"{value:.3f}", (value, i), xytext=(7, 0), textcoords="offset points", va="center", fontsize="small"
        )
    for (family, color), points in groups.items():
        ax.scatter([p[0] for p in points], [p[1] for p in points], color=color, s=32, label=family)
    ax.set_yticks(range(len(factor_arms)), [labels[a["precision"]] for a in factor_arms])
    ax.invert_yaxis()
    fv.mark_neutral(ax.axvline(1, color=fv.NEUTRALS["baseline"], linestyle=":"))
    ax.set(xlabel="Final target KL / initial target KL", xlim=(0, max(2, ax.get_xlim()[1] * 1.2)))
    style(ax)
    handles = [
        Line2D([], [], linestyle="", marker="o", color=color, label=label)
        for label, color in [
            ("Forward", colors[1]),
            ("Backward", colors[2]),
            ("GEMM setting", fv.HUES["vorange"]),
            ("Baseline", fv.NEUTRALS["baseline"]),
        ]
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize="small")
    fig.subplots_adjust(left=0.40, right=0.98, bottom=0.14, top=0.90)
    save(fig, "02-factor-solves", "SGD 0.003; 16 updates; values below 1 improve on the initial target KL")

    forward = [
        r for r in data["forward_controls"]["evaluations"] if r["snapshot"] == "1e-07/1" and r["precision"] != "float32"
    ]
    forward += [
        r
        for r in data["forward_followup"]["evaluations"]
        if r["snapshot"] == "1e-07/1" and r["precision"] != "bfloat16"
    ]
    baseline = next(r["metrics"]["distill/mode_anchor_kl"] for r in forward if r["precision"] == "bfloat16")
    fig, ax = plt.subplots(figsize=(8.3, 6.0))
    ratios = [record["metrics"]["distill/mode_anchor_kl"] / baseline for record in forward]
    assert min(ratios) > 1e-4 and max(ratios) < 3
    ax.scatter(ratios, range(len(ratios)), color=colors[1], s=32)
    for i, ratio in enumerate(ratios):
        ax.annotate(
            f"{ratio:.3g}",
            (ratio, i),
            xytext=(-8, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize="small",
        )
    ax.set_yticks(
        range(len(forward)), [labels.get(r["precision"], "FP32 arithmetic, BF16-valued weights") for r in forward]
    )
    ax.invert_yaxis()
    fv.mark_neutral(ax.axvline(1, color=fv.NEUTRALS["baseline"], linestyle=":"))
    ax.set(xscale="log", xlim=(1e-4, 3), xlabel="Policy displacement KL / native BF16 displacement KL")
    style(ax)
    fig.subplots_adjust(left=0.40, right=0.98, bottom=0.16, top=0.97)
    save(fig, "03-fixed-factors", "Same stored parameter update; each mode measured against its own initial output")
    fig, ax = plt.subplots(figsize=(7.4, 4.1))
    selected = [
        ("long-precision", 8, "FP32 / LR=1e-6", colors[0]),
        ("muon-lr-controls", 0, "FP32 / LR=3e-7", colors[1]),
        ("muon-lr-controls", 1, "FP32 / LR=1e-7", colors[2]),
        ("long-precision", 5, "BF16 W + FP32 ops / LR=1e-6", fv.HUES["vorange"]),
    ]
    for run, index, label, color in selected:
        arm = next(a for a in data["arms"] if a["run"] == run and a["index"] == index)
        values = np.asarray([[0, arm["before"]["distill/target_kl"]]] + [p[:2] for p in arm["post_series"]])
        ax.plot(values[:, 0], values[:, 1] / values[0, 1], label=label, color=color, linewidth=1.2)
    ax.set(yscale="log", xlabel="Optimizer updates", ylabel="Target KL / initial", xlim=(0, 256))
    ax.set_xticks([0, 64, 128, 192, 256])
    style(ax)
    fig.legend(loc="upper center", ncol=2, frameon=False, fontsize="small")
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.19, top=0.78)
    save(fig, "04-muon-step-size", "Muon+AdamW; both optimizer-group learning rates change together")

    captures = data["attention_capture"]["rows"]
    total_error = sum(row["dq"]["error_energy"] for row in captures)
    layers, relative, cast, fractions = [], [], [], []
    for layer in range(28):
        group = [row["dq"] for row in captures if row["layer"] == layer]
        ref = sum(row["reference_energy"] for row in group)
        err = sum(row["error_energy"] for row in group)
        layers.append(layer)
        relative.append(100 * (err / ref) ** 0.5)
        cast.append(100 * (sum(row["cast_only_error_energy"] for row in group) / ref) ** 0.5)
        fractions.append(100 * err / total_error)
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.7))
    axs[0].plot(layers, relative, color=colors[0], label="Native FA2 VJP")
    axs[0].plot(layers, cast, color=fv.NEUTRALS["baseline"], label="FP64 reference cast to BF16")
    axs[0].set(yscale="log", ylabel="Relative dQ error (%)", title="Conditional VJP accuracy")
    axs[0].legend(frameon=False, fontsize="small", loc="upper right")
    axs[1].bar(layers, fractions, color=colors[0])
    axs[1].set(ylabel="Share of dQ error energy (%)", ylim=(0, 105), title="Error concentration")
    for axis in axs:
        axis.set(xlabel="Transformer layer (zero-based)", xlim=(-0.5, 27.5))
        axis.set_xticks([0, 7, 14, 21, 27])
        style(axis)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.22, top=0.90, wspace=0.32)
    save(fig, "05-attention-vjp", "112 actual captures; one backward; activation-gradient error, not loss attribution")
    (out / "figure-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
