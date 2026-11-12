# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Figures for checkpoint geometry; numerical derivation lives in readouts."""

import numpy as np

from examples.mirroropd.research.core.plotting import load_style, save_figure


def render(data, destination):
    """Plot within-pair dynamics only on layers with all seven matrices available."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fv = load_style()
    fv.apply_rc("paper")
    colors = fv.series_colors("paper", 3)
    destination.mkdir(parents=True, exist_ok=True)
    for name, pair in data["pairs"].items():
        matrices = [m for m in pair["matrices"] if m["layer"] in pair["complete_layers"]]
        if not matrices:
            continue
        steps = [s["step"] for s in matrices[0]["stages"]]
        coverage = f"{len(matrices)}/{pair['expected']} matrices; {len(pair['complete_layers'])} complete layers"
        # Pair IDs map to full model names in the report; long names crowd axes.
        title = name
        caption = f"Median and interquartile range across matrices; {coverage}. Right update subspaces."
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), layout="constrained")
        fig.get_layout_engine().set(rect=(0, 0.08, 1, 0.9))
        for color, rank in zip(colors, (16, 64, 256), strict=True):
            rows = [
                [
                    next(
                        p
                        for p in m["pairs"]
                        if p["step_a"] == step and p["step_b"] == steps[-1] and p["requested_rank"] == rank
                    )
                    for step in steps
                ]
                for m in matrices
            ]
            overlap = np.array([[p["adjusted_overlap"] for p in row] for row in rows])
            captures = np.array(
                [
                    [
                        p["a_captures_b"]
                        / next(r["captured_energy"] for r in m["stages"][-1]["ranks"] if r["requested_rank"] == rank)
                        for p in row
                    ]
                    for m, row in zip(matrices, rows, strict=True)
                ]
            )
            for ax, values in zip(axes, (overlap, captures), strict=True):
                ax.plot(steps, np.median(values, axis=0), color=color, label=f"k={rank}")
                ax.fill_between(steps, *np.quantile(values, [0.25, 0.75], axis=0), color=color, alpha=0.15)
                ax.set_xlabel("Optimizer updates at anchor")
                ax.set_ylim(-0.03, 1.03)
        axes[0].set_ylabel("Overlap with final update (random-adjusted)")
        axes[1].set_ylabel("Final energy captured / optimal rank-k capture")
        axes[0].legend(loc="lower right")
        fig.suptitle(title)
        save_figure(fig, destination / f"{name}-early-final", caption)
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), layout="constrained")
        fig.get_layout_engine().set(rect=(0, 0.1, 1, 0.9))
        for ax, key, label, color in zip(
            axes,
            ("delta_sigma", "rho_sigma", "kappa_spec"),
            ("Full-weight spectral distance", "Spectral / update norm", "Spectral concentration"),
            colors,
            strict=True,
        ):
            values = np.array([[s["weight"][key] for s in m["stages"]] for m in matrices])
            ax.plot(steps, np.median(values, axis=0), color=color)
            ax.fill_between(steps, *np.quantile(values, [0.25, 0.75], axis=0), color=color, alpha=0.15)
            ax.set_xlabel("Optimizer updates")
            ax.set_ylabel(label)
        fig.suptitle(title)
        save_figure(
            fig,
            destination / f"{name}-weight-spectrum",
            f"Median and IQR; {coverage}. Initial weight is the reference.",
        )
        if not pair["teacher_parameter_comparison"]:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), layout="constrained")
        fig.get_layout_engine().set(rect=(0, 0.1, 1, 0.9))
        for ax, key, label, color in zip(
            axes,
            ("norm_ratio", "cosine", "residual_energy_ratio"),
            ("OPD / teacher update norm", "Update cosine", "Residual / teacher energy"),
            colors,
            strict=True,
        ):
            values = np.array([[s["teacher_comparison"][key] for s in m["stages"]] for m in matrices])
            ax.plot(steps, np.median(values, axis=0), color=color)
            ax.fill_between(steps, *np.quantile(values, [0.25, 0.75], axis=0), color=color, alpha=0.15)
            ax.set_xlabel("Optimizer updates")
            ax.set_ylabel(label)
        fig.suptitle(title)
        save_figure(
            fig,
            destination / f"{name}-parameter-residual",
            f"Median and IQR; {coverage}. Teacher update is relative to the same initial weight.",
        )
