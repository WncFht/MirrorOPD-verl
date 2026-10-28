#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Ledger: synthetic KL noise expectation vs quantization step; measured
arithmetic error vs true KL. Error bars are Monte Carlo mean 95% intervals,
not model/seed performance uncertainty. The error panel takes the maximum
across both synthetic profiles and shifts at each target. No trained model data.
"""

import argparse
import json
from pathlib import Path

import fvlocate

fvlocate.ensure()
import fvstyle as fv  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    d = json.loads(args.data.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    fv.apply_rc("paper", fontsize=10)
    colors = fv.series_colors("paper", 4)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for color, rho in zip(colors[:3], [0.0, 0.5, 0.9], strict=True):
        rows = [r for r in d["noise_model"] if r["profile"] == "peaked" and r["rho"] == rho]
        xs = [r["step"] for r in rows]
        axes[0].plot(xs, [r["predicted_mean"] for r in rows], color=color, lw=1.6, label=f"correlation {rho:g}")
        axes[0].errorbar(
            xs,
            [r["measured_mean"] for r in rows],
            yerr=[1.96 * r["mc_standard_error"] for r in rows],
            fmt="o",
            color=color,
            markersize=4,
            capsize=3,
        )
    axes[0].set(
        xscale="log",
        yscale="log",
        xlabel="Quantization step",
        ylabel="Expected KL (nats)",
        title="Noise model: lines predicted, points measured",
    )
    targets = sorted({r["target"] for r in d["arithmetic"] if r["target"]})
    for method, color, label in [
        ("naive_fp32", fv.NEUTRALS["baseline"], "Current FP32 formula"),
        ("stable_fp32", colors[3], "Stable identity (audit only)"),
    ]:
        vals = [max(r[method]["max_abs_error"] for r in d["arithmetic"] if r["target"] == t) for t in targets]
        axes[1].plot(targets, vals, color=color, marker="o", lw=1.6, label=label, markersize=4)
    axes[1].set(
        xscale="log",
        yscale="log",
        xlabel="Reference mean KL (nats)",
        ylabel="Maximum absolute row error (nats)",
        title="FP32 arithmetic resolution",
    )
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=fv.NEUTRALS["vmist"], lw=0.6)
    handles, labels = [], []
    for ax in axes:
        h, labels_here = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(labels_here)
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.13, 1, 0.80), w_pad=2)
    fig.text(
        0.5,
        0.025,
        "Synthetic inputs only. Left: 4096 trials per point. Right: vocabulary 151936; two profiles and shifts.",
        ha="center",
        fontsize=9,
        color=fv.NEUTRALS["baseline"],
    )
    fig.canvas.draw()
    issues = fv.audit(fig, print_slot=fig.get_figwidth())
    (args.out / "precision-model.audit.txt").write_text(
        "\n".join(fv.format_audit_lines(issues)) + "\n" if issues else "PASS\n"
    )
    if any(fv.problem_severity(i) == "error" for i in issues):
        raise RuntimeError(fv.format_audit_lines(issues))
    for ext in ["png", "pdf"]:
        fig.savefig(args.out / f"precision-model.{ext}", dpi=170, facecolor=fv.NEUTRALS["paper"])
    plt.close(fig)


if __name__ == "__main__":
    main()
