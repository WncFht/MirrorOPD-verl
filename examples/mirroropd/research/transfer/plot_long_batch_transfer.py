#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Ledger: twenty disjoint training blocks, all forty fixed banks reevaluated.
Risk heatmaps normalize each column by its own initial risk. Influence cells
are successive risk reductions divided by the same initial column risk, not
independent causal interventions. Black outlines mark the newly trained batch.
First-update projections use a separate fixed 64-prompt math diagnostic panel;
the finite change is observed in the same mixed-precision implementation.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle

from ..common.plotting import load_style


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    fv = load_style()
    a.out.mkdir(parents=True, exist_ok=True)
    read = lambda n: json.loads((a.data / n).read_text())
    stats = read("risk-summary.json")
    matrices = read("transfer-matrix.json")
    incoming = read("incoming-batches.json")
    fresh = read("fresh-summary.json")
    fv.apply_rc("paper", fontsize=11)
    audits = {}
    gray = fv.NEUTRALS["baseline"]
    teal = fv.HUES["vteal"]
    orange = fv.HUES["vorange"]
    colors = {"justrl": teal, "polaris": orange}

    def save(fig, name, caption):
        fig.text(0.5, 0.02, caption, ha="center", fontsize=10, color=gray)
        fig.canvas.draw()
        audits[name] = [str(x) for x in fv.audit(fig)]
        for ext in ["png", "pdf"]:
            fig.savefig(a.out / f"{name}.{ext}", dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.8))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.91, bottom=0.28, wspace=0.28)
    panels = [
        ("train_pool", "Train bank A–T", gray),
        ("held_math", "Held math", fv.HUES["vviolet"]),
        ("held_code", "Held code", fv.HUES["vblue"]),
        ("held_if", "Held IF", orange),
    ]
    for ax, t in zip(axes, ["justrl", "polaris"], strict=False):
        for panel, label, color in panels:
            rows = sorted(
                [r for r in stats if r["teacher"] == t and r["panel"] == panel], key=lambda r: r["completed_batches"]
            )
            ax.plot(
                [r["completed_batches"] for r in rows],
                [r["rkl"] / r["rkl_initial"] for r in rows],
                marker="o",
                markersize=3,
                color=color,
                label=label,
            )
        ax.set(
            title="JustRL" if t == "justrl" else "Polaris",
            xlabel="Completed training batches",
            ylabel="Fixed-bank RKL / initial",
            ylim=(-0.02, 1.08),
            xticks=[0, 5, 10, 15, 20],
        )
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.07),
        ncol=2,
        frameon=False,
        fontsize=10,
    )
    save(fig, "fixed-risk-curves", "All panels remain fixed; train-bank average includes both seen and future batches.")

    maximum = max(1.05, max(np.array(m["risk"]).max(axis=0).max() for m in matrices.values()))
    # Color limits are normalized-risk limits, shared across teachers.
    maximum = max(1.05, max(float((np.array(m["risk"]) / np.array(m["risk"])[0]).max()) for m in matrices.values()))
    cmap = LinearSegmentedColormap.from_list("remaining", [fv.NEUTRALS["paper"], orange])
    all_changes = []
    for m in matrices.values():
        z = np.array(m["risk"])
        all_changes.extend(((z[:-1] - z[1:]) / z[0] * 100).reshape(-1).tolist())
    lo = min(-5, math.floor(min(all_changes or [0]) / 5) * 5)
    hi = max(5, math.ceil(max(all_changes or [0]) / 5) * 5)
    diverging = LinearSegmentedColormap.from_list("influence", [fv.HUES["vcoral"], fv.NEUTRALS["paper"], teal])
    for t, m in matrices.items():
        z = np.array(m["risk"])
        groups = m["groups"]
        batches = m["completed_batches"]
        for kind in ["risk", "influence"]:
            values = z / z[0] if kind == "risk" else (z[:-1] - z[1:]) / z[0] * 100
            ticks = batches if kind == "risk" else batches[1:]
            fig, ax = plt.subplots(figsize=(9.5, 6.8))
            fig.subplots_adjust(left=0.10, right=0.86, top=0.93, bottom=0.24)
            im = ax.imshow(
                values,
                aspect="auto",
                cmap=cmap if kind == "risk" else diverging,
                norm=None if kind == "risk" else TwoSlopeNorm(vmin=lo, vcenter=0, vmax=hi),
                **(dict(vmin=0, vmax=maximum) if kind == "risk" else {}),
            )
            for row, b in enumerate(ticks):
                if b:
                    ax.add_patch(
                        Rectangle(
                            (b - 1 - 0.5, row - 0.5),
                            1,
                            1,
                            fill=False,
                            facecolor=fv.NEUTRALS["paper"],
                            edgecolor=fv.NEUTRALS["vink"],
                            linewidth=0.8,
                        )
                    )
            for boundary in [19.5, 31.5, 35.5]:
                ax.axvline(boundary, color=gray, lw=0.8)
            short = [g.replace("Code", "C").replace("IF", "I") if len(g) > 1 else g for g in groups]
            ax.set(
                title=("JustRL" if t == "justrl" else "Polaris")
                + (" — remaining risk" if kind == "risk" else " — improvement in each block"),
                xticks=range(40),
                xticklabels=short,
                yticks=range(len(ticks)),
                yticklabels=[str(b) if b == 0 else f"{b}: {chr(64 + b)}" for b in ticks],
                xlabel="A–T train | M01–M12 held math | C01–C04 code | I01–I04 IF",
                ylabel="Completed batch / active training batch",
            )
            ax.tick_params(axis="x", rotation=90)
            cax = fig.add_axes([0.9, 0.26, 0.025, 0.62])
            fig.colorbar(im, cax=cax, label="RKL / initial" if kind == "risk" else "Decrease (% of initial KL)")
            save(
                fig,
                f"{kind}-matrix-{t}",
                "Every cell is measured on fixed prefixes. Outline: newly trained batch; no batch revisits.",
            )

    if incoming:
        fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.8))
        fig.subplots_adjust(left=0.10, right=0.98, top=0.91, bottom=0.26, wspace=0.28)
        for ax, t in zip(axes, ["justrl", "polaris"], strict=False):
            rows = [r for r in incoming if r["teacher"] == t]
            for field, label, color in [
                ("risk_before_first_training", "Before first training", gray),
                ("risk_after_training", "After its 16 updates", teal),
            ]:
                ax.plot(
                    [r["training_batch"] for r in rows],
                    [r[field] / r["initial_risk"] for r in rows],
                    marker="o",
                    markersize=3,
                    color=color,
                    label=label,
                )
            ax.set(
                title="JustRL" if t == "justrl" else "Polaris",
                xlabel="New training batch index",
                ylabel="Active-batch RKL / its initial",
                ylim=(-0.02, 1.08),
                xticks=[1, 5, 10, 15, 20],
            )
        fig.legend(
            *axes[0].get_legend_handles_labels(),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.07),
            ncol=2,
            frameon=False,
            fontsize=10,
        )
        save(
            fig,
            "incoming-transfer",
            "Each batch is unseen before its turn; its initial risk is measured at the common starting model.",
        )
        fig, ax = plt.subplots(figsize=(6.9, 4.6))
        fig.subplots_adjust(left=0.16, right=0.97, top=0.94, bottom=0.26)
        for t in ["justrl", "polaris"]:
            rows = [r for r in incoming if r["teacher"] == t and r["held_drop_per_local_drop"] is not None]
            ax.plot(
                [r["training_batch"] for r in rows],
                [r["held_drop_per_local_drop"] for r in rows],
                marker="o",
                markersize=3,
                color=colors[t],
                label="JustRL" if t == "justrl" else "Polaris",
            )
        ax.axhline(0, color=gray, lw=0.8, label="Zero held-out gain")
        ax.set(xlabel="Training batch index", ylabel="Held-math drop / active-batch drop", xticks=[1, 5, 10, 15, 20])
        fig.legend(
            *ax.get_legend_handles_labels(),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.07),
            ncol=3,
            frameon=False,
            fontsize=9,
        )
        save(
            fig,
            "transfer-yield",
            "Same-state finite changes over 16 updates; this ratio is not a probability or a causal percentage.",
        )

    probes = {
        t: read(t + "/update-probes.json")
        for t in ["justrl", "polaris"]
        if (a.data / t / "update-probes.json").exists()
    }
    if len(probes) == 2:
        fig, axes = plt.subplots(3, 2, figsize=(9.3, 10.5))
        fig.subplots_adjust(left=0.12, right=0.97, top=0.95, bottom=0.14, hspace=0.45, wspace=0.35)
        span = (
            max(
                abs(r[k])
                for rows in probes.values()
                for r in rows
                for k in ["finite_held_change", "held_dot_actual_update"]
            )
            * 1.5
        )
        for col, t in enumerate(["justrl", "polaris"]):
            rows = probes[t]
            x = [r["training_batch"] for r in rows]
            axes[0, col].plot(x, [r["raw_train_held_cosine"] for r in rows], marker="o", markersize=3, color=teal)
            axes[0, col].set(
                title="JustRL" if t == "justrl" else "Polaris",
                ylabel="Raw train–held gradient cosine",
                ylim=(-0.05, 1.05),
            )
            axes[1, col].plot(x, [r["descent_alignment"] for r in rows], marker="o", markersize=3, color=orange)
            axes[1, col].set(ylabel="Actual-update descent alignment", ylim=(-0.01, 0.10))
            for key, label, color in [
                ("finite_held_change", "Observed first-step decrease", teal),
                ("held_dot_actual_update", "Linear prediction", orange),
            ]:
                axes[2, col].plot(x, [-r[key] for r in rows], marker="o", markersize=3, label=label, color=color)
            axes[2, col].set(yscale="symlog", ylabel="Probe RKL decrease (symlog)", ylim=(-span, span))
            axes[2, col].set_yscale("symlog", linthresh=1e-6)
            for row in range(3):
                axes[row, col].set(xlabel="Training batch index", xticks=[1, 5, 10, 15, 20])
        fig.legend(
            *axes[2, 0].get_legend_handles_labels(),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.055),
            ncol=2,
            frameon=False,
            fontsize=9,
        )
        save(
            fig,
            "actual-update-probes",
            (
                "First update of each block; same 64 held-out prompts. Linear "
                "prediction uses surrogate mixed-precision gradients."
            ),
        )
    if fresh:
        fig, axes = plt.subplots(2, 1, figsize=(6.9, 6.9))
        fig.subplots_adjust(left=0.16, right=0.97, top=0.95, bottom=0.20, hspace=0.42)
        for t in ["justrl", "polaris"]:
            rows = [r for r in fresh if r["teacher"] == t]
            if not rows:
                continue
            x = [r["completed_batches"] for r in rows]
            y = np.array([r["sequence_rkl"] / r["initial_sequence_rkl"] for r in rows])
            lower = np.array([r["ratio_paired_prompt_ci"][0] for r in rows])
            upper = np.array([r["ratio_paired_prompt_ci"][1] for r in rows])
            label = "JustRL" if t == "justrl" else "Polaris"
            axes[0].errorbar(x, y, yerr=[y - lower, upper - y], fmt="o-", color=colors[t], label=label, capsize=0)
            axes[1].plot(x, [r["mean_length"] for r in rows], marker="o", color=colors[t], label=label)
        axes[0].set(ylabel="Fresh sequence RKL / initial", ylim=(0, 1.2))
        axes[1].set(ylabel="Mean response tokens", ylim=(0, 1100))
        for ax in axes:
            ax.set(xlabel="Completed training batches", xticks=[0, 5, 10, 15, 20])
        fig.legend(
            *axes[1].get_legend_handles_labels(),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.065),
            ncol=2,
            frameon=False,
            fontsize=10,
        )
        save(
            fig,
            "fresh-risk",
            (
                "Unrestricted current-student rollout, 32 prompts × 2 streams; paired "
                "prompt ratio intervals, not training-seed uncertainty."
            ),
        )
    (a.out / "audit.json").write_text(json.dumps(audits, indent=2) + "\n")
    print(json.dumps(audits, indent=2))


if __name__ == "__main__":
    main()
