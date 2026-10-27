#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Render measured MDPO trajectories; never generate synthetic training data.

Semantic ledger: each line is one independently restored solver arm. Step zero
is its own frozen-anchor score; subsequent points are post-update FP64 row
scores. Running minimum includes step zero and is not another training run.
Colors identify solvers in solver plots and precision policies in precision
plots; each figure has its own explicit legend. No smoothing or uncertainty
bands: the available runs do not form an independent-seed sample.
Geometry ledger: x is optimizer updates, not outer steps or elapsed time.
KL ratios are dimensionless; absolute KL plots use nats. Shared axes have the
same limits. The fixed-displacement figure uses a log axis and points (no
zero-origin bars). Figures are exported as PNG and PDF, without HTML.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-skill", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.figure_skill / "assets/mpl"))
    import fvstyle as fv

    root = Path(__file__).resolve().parents[1]
    out = args.out or root / "figures/precision-report-20261027"
    out.mkdir(parents=True, exist_ok=True)
    data = json.loads((root / "data/solver-precision-20261027.json").read_text())
    assert len(data["arms"]) == 62
    assert sum(a["steps"] for a in data["arms"]) == 3440
    assert data["post_series_columns"][:5] == ["update", "target_kl", "teacher_kl", "anchor_kl", "objective"]
    indexed = {(a["run"], a["index"]): a for a in data["arms"]}
    fv.apply_rc("paper", fontsize=9)
    solver_colors = [fv.HUES[k] for k in ("vteal", "vorange", "vcoral", "vviolet")]
    neutral = fv.NEUTRALS["baseline"]
    audit_results = {}

    def series(run, index):
        arm = indexed[run, index]
        b = arm["before"]
        rows = [
            [0, b["distill/target_kl"], b["distill/teacher_kl"], b["distill/anchor_kl"], b["distill/mdpo/objective"]]
        ]
        rows += [r[:5] for r in arm["post_series"]]
        a = np.asarray(rows)
        assert np.isfinite(a).all()
        assert np.allclose(a[:, 4], a[:, 2] + (7 / 3) * a[:, 3], rtol=1e-8, atol=1e-12)
        return a

    def axes_style(ax):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3, color=fv.NEUTRALS["vmist"])
        ax.set_axisbelow(True)

    def reference(ax):
        fv.mark_neutral(ax.axhline(1, color=neutral, linestyle=":", linewidth=1))

    def save(fig, name, caption):
        fig.text(0.5, 0.012, caption, ha="center", color=fv.NEUTRALS["vink"], fontsize="small")
        fig.canvas.draw()
        audit_results[name] = fv.audit(fig, print_slot=fig.get_figwidth())
        for problem in audit_results[name]:
            print(name, problem)
        for extension in ("png", "pdf"):
            fig.savefig(out / f"{name}.{extension}", dpi=200)
        plt.close(fig)

    # 1: same 16-update budget on the integrated production numerical repairs.
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    labels = ["Adam, 1e-6", "SGD, 0.003", "Muon+AdamW, 1e-6", "SGD, BF16 weights / FP32 compute"]
    for i, label in enumerate(labels):
        a = series("current-main-confirm", i)[:17]
        ax.plot(a[:, 0], a[:, 1] / a[0, 1], color=solver_colors[i], label=label, linewidth=1.3)
    reference(ax)
    ax.set(xlabel="Optimizer updates", ylabel="Target KL / initial target KL", xlim=(0, 16), ylim=(0, 2.25))
    ax.set_xticks([0, 4, 8, 12, 16])
    axes_style(ax)
    fig.legend(loc="upper center", ncol=2, frameon=False, fontsize="small")
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.18, top=0.79)
    save(fig, "01-short-solve", "State A; post-update scores; dotted line = initial score")

    # 2: an actual objective decomposition; all terms share nats and zero origin.
    a = series("bf16-grid", 0)
    fig, ax = plt.subplots(figsize=(6.6, 3.7))
    ax.plot(a[:, 0], a[:, 2] * 1e4, color=solver_colors[0], label="Teacher KL")
    ax.plot(a[:, 0], a[:, 3] * (7 / 3) * 1e4, color=solver_colors[1], label="Anchor penalty: (7/3) KL(p || q)")
    ax.plot(a[:, 0], a[:, 4] * 1e4, color=solver_colors[2], label="Objective J = teacher KL + penalty")
    fv.mark_neutral(ax.axhline(a[0, 4] * 1e4, color=neutral, linestyle=":"))
    ax.set(xlabel="Optimizer updates", ylabel="Value (units of 1e-4 nats)", xlim=(0, 16), ylim=(0, 13))
    ax.set_xticks([0, 4, 8, 12, 16])
    axes_style(ax)
    fig.legend(loc="upper center", ncol=1, frameon=False, fontsize="small")
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.19, top=0.74)
    save(fig, "02-objective-components", "State A; saved Adam 1e-6; original solver baseline fd5f4a479")

    # 3: noisy trajectories and running minima, independently restored arms.
    entries = [
        ("long-exact", 0, "Adam 1e-6"),
        ("long-exact", 1, "SGD 0.003"),
        ("long-exact", 2, "Muon+AdamW 1e-6"),
        ("adam-long-controls", 0, "Fresh Adam 1e-7"),
    ]
    fig, axs = plt.subplots(2, 2, figsize=(7.4, 5.7), sharex=True, sharey=True)
    for ax, (run, i, title), color in zip(axs.flat, entries, solver_colors, strict=True):
        a = series(run, i)
        y = a[:, 1] / a[0, 1]
        ax.plot(a[:, 0], y, color=color, linewidth=0.8, alpha=0.65, label=title)
        minimum = ax.plot(a[:, 0], np.minimum.accumulate(y), color=color, linestyle="--", linewidth=1.5)[0]
        fv.mark_neutral(minimum)
        reference(ax)
        ax.set_title(title, loc="left")
        ax.set(xlim=(0, 256), ylim=(0.65, 2.15))
        ax.set_xticks([0, 64, 128, 192, 256])
        axes_style(ax)
    for ax in axs[-1]:
        ax.set_xlabel("Optimizer updates")
    for ax in axs[:, 0]:
        ax.set_ylabel("Target KL / initial")
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.15, top=0.94, hspace=0.30, wspace=0.16)
    save(fig, "03-long-solves", "State A; solid = actual iterate; dashed = best so far (including step 0)")

    # 4: equal rounded starting weights, own q per precision, same solver budget.
    fig, axs = plt.subplots(1, 3, figsize=(8.8, 3.5), sharex=True, sharey=True)
    policies = [
        ("precision", 0, "BF16", solver_colors[2]),
        ("mixed-precision", 0, "BF16 weights / FP32 compute", solver_colors[0]),
        ("precision", 3, "FP32", solver_colors[3]),
    ]
    for i, ax in enumerate(axs):
        for run, offset, label, color in policies:
            a = series(run, i + offset)
            ax.plot(a[:, 0], a[:, 1] / a[0, 1], color=color, label=label, linewidth=1.15)
        reference(ax)
        ax.set_title(["Adam 1e-6", "SGD 0.003", "Muon+AdamW 1e-6"][i], loc="left")
        ax.set(xlabel="Optimizer updates", xlim=(0, 32), ylim=(0, 2.1))
        ax.set_xticks([0, 16, 32])
        axes_style(ax)
    axs[0].set_ylabel("Target KL / initial")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize="small")
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.22, top=0.77, wspace=0.14)
    save(fig, "04-precision-solves", "State A; same BF16-valued master start; each precision freezes its own q")

    # 5: same stored displacement, not newly computed optimization trajectories.
    def forward(run, precision):
        matches = [
            r
            for r in data["forward_controls"][run]["evaluations"]
            if r["precision"] == precision and r["snapshot"] == "1e-07/1"
        ]
        assert len(matches) == 1
        return matches[0]["metrics"]["distill/mode_anchor_kl"]

    settings = [
        ("mixed-fixed-step", "bfloat16", "Native BF16"),
        ("residual-fixed-step", "fp32_residual", "FP32 intermediate bundle"),
        ("projection-fixed-step", "bf16_mlp_fp32_outputs", "Bundle + MLP outputs FP32"),
        ("projection-fixed-step", "bf16_attention_fp32_outputs", "Bundle + attention projections FP32"),
        ("mixed-fixed-step", "bf16_gemm_fp32_outputs", "Bundle + all projections FP32"),
        ("mixed-fixed-step", "float32", "FP32 arithmetic, BF16-valued weights"),
    ]
    values = [forward(run, precision) for run, precision, _ in settings]
    fig, ax = plt.subplots(figsize=(7.8, 3.7))
    y = np.arange(len(settings))
    ax.scatter(values, y, color=solver_colors[0], s=34)
    for x, yi in zip(values, y, strict=True):
        ax.annotate(f"{x:.3e}", (x, yi), xytext=(8, 0), textcoords="offset points", va="center", fontsize="small")
    ax.set_yticks(y, [s[2] for s in settings])
    ax.invert_yaxis()
    ax.set(xscale="log", xlim=(3e-8, 8e-4), xlabel="Policy displacement KL (nats, log scale)")
    axes_style(ax)
    fig.subplots_adjust(left=0.40, right=0.98, top=0.96, bottom=0.20)
    save(fig, "05-fixed-displacement", "Same stored update; each mode compared with its own initial output")
    (out / "figure-audit.json").write_text(json.dumps(audit_results, indent=2) + "\n")
    print(f"Rendered {len(audit_results)} figures into {out}")


if __name__ == "__main__":
    main()
