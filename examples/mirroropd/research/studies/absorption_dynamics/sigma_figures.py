# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Trace-normalized spectrum figures from exported, self-contained readouts."""

import numpy as np

from examples.mirroropd.research.core.plotting import load_style, save_figure


def band(ax, x, values, color, label):
    values = np.asarray(values, dtype=float)
    ax.plot(x, np.median(values, axis=0), color=color, label=label)
    ax.fill_between(x, *np.quantile(values, [0.25, 0.75], axis=0), color=color, alpha=0.14)


def positive_log(ax):
    # A self-distance is exactly zero: final-state curves stay on linear axes.
    if any(np.any(np.asarray(line.get_ydata()) > 0) for line in ax.lines):
        ax.set_yscale("log")


def render(data, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fv = load_style()
    fv.apply_rc("paper")
    colors = fv.series_colors("paper", 3)
    for name, pair in data["pairs"].items():
        matrices = [m for m in pair["matrices"] if m["layer"] in pair["complete_layers"] and "sigma_variation" in m]
        if not matrices:
            continue
        coverage = f"{len(matrices)}/{pair['expected']} matrices; {len(pair['complete_layers'])} full layers"
        groups = [
            ("Attention", [m for m in matrices if m["module"] in ("q_proj", "k_proj", "v_proj", "o_proj")]),
            ("MLP", [m for m in matrices if m["module"] in ("gate_proj", "up_proj", "down_proj")]),
        ]
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
        fig.get_layout_engine().set(rect=(0, 0.08, 1, 0.94))
        for row, kind in enumerate(("weight", "update")):
            for color, (label, subset) in zip(colors, groups, strict=False):
                samples = [m["sigma_variation"][kind] for m in subset]
                intervals = samples[0]["intervals"]
                x = [p["end"] for p in intervals]
                band(axes[row, 0], x, [[p["sd_variation"] for p in s["intervals"]] for s in samples], color, label)
                band(axes[row, 1], x, [[p["net_speed"] for p in s["intervals"]] for s in samples], color, label)
                steps = [p["step"] for p in samples[0]["points"]]
                band(
                    axes[row, 2],
                    steps,
                    [[p["sd_variation"] for p in s["pairs"] if p["step_b"] == steps[-1]] for s in samples],
                    color,
                    label,
                )
            for col, label in enumerate(
                ("SD variation / interval", "Net SD change / optimizer step", "SD distance to final checkpoint")
            ):
                axes[row, col].set_xlabel("Optimizer updates (interval end)")
                axes[row, col].set_ylabel(("Full weight: " if row == 0 else "Cumulative update: ") + label)
                if col < 2:
                    positive_log(axes[row, col])
        axes[0, 0].legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2)
        fig.suptitle(f"{name}: spectral shape dynamics")
        save_figure(
            fig,
            destination / f"{name}-sigma-dynamics",
            f"Median and IQR; {coverage}. Unequal intervals; net speed is not path length.",
            caption_size=8,
        )

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), layout="constrained")
        fig.get_layout_engine().set(rect=(0, 0.1, 1, 0.9))
        for ax, kind in zip(axes, ("weight", "update"), strict=True):
            sample = matrices[0]["sigma_variation"][kind]
            steps = [p["step"] for p in sample["points"]]
            grid = np.zeros((len(steps), len(steps)))
            for i, a in enumerate(steps):
                for j, b in enumerate(steps):
                    lo, hi = sorted((a, b))
                    grid[i, j] = np.median(
                        [
                            next(
                                p["sd_variation"]
                                for p in m["sigma_variation"][kind]["pairs"]
                                if p["step_a"] == lo and p["step_b"] == hi
                            )
                            for m in matrices
                        ]
                    )
            image = ax.imshow(grid, cmap="cividis", vmin=0)
            ax.set_xticks(range(len(steps)), steps, rotation=60)
            ax.set_yticks(range(len(steps)), steps)
            ax.set_xlabel("Optimizer updates")
            ax.set_ylabel("Optimizer updates")
            ax.set_title("Full weight" if kind == "weight" else "Cumulative update")
            fig.colorbar(image, ax=ax, label="Median SD distance", shrink=0.8)
        fig.suptitle(f"{name}: checkpoint spectral distances")
        save_figure(fig, destination / f"{name}-sigma-pairwise", f"{coverage}. Each panel has its own color scale.")

        # Hellinger distance avoids the raw L2 scale shrinking with singular dimension.
        fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
        fig.get_layout_engine().set(rect=(0, 0.1, 1, 0.9))
        layers = pair["complete_layers"]
        for ax, kind in zip(axes, ("weight", "update"), strict=True):
            intervals = matrices[0]["sigma_variation"][kind]["intervals"]
            grid = np.array(
                [
                    [
                        np.median(
                            [
                                m["sigma_variation"][kind]["intervals"][i]["hellinger"] / p["optimizer_interval"]
                                for m in matrices
                                if m["layer"] == layer
                            ]
                        )
                        for i, p in enumerate(intervals)
                    ]
                    for layer in layers
                ]
            )
            image = ax.imshow(grid, aspect="auto", cmap="cividis", vmin=0)
            ax.set_xticks(range(len(intervals)), [f"{p['start']}-{p['end']}" for p in intervals], rotation=60)
            ticks = list(range(0, len(layers), max(1, len(layers) // 10)))
            ax.set_yticks(ticks, [layers[i] for i in ticks])
            ax.set_xlabel("Checkpoint interval (optimizer updates)")
            ax.set_ylabel("Transformer layer")
            ax.set_title("Full weight" if kind == "weight" else "Cumulative update")
            fig.colorbar(image, ax=ax, label="Hellinger net change / step", shrink=0.8)
        fig.suptitle(f"{name}: layer-resolved spectral dynamics")
        save_figure(
            fig,
            destination / f"{name}-sigma-layers",
            f"Median over seven projections per layer; {coverage}. Panel scales differ.",
        )

        if "teacher_distances" in matrices[0]["sigma_variation"]:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4.3), layout="constrained")
            fig.get_layout_engine().set(rect=(0, 0.1, 1, 0.9))
            for ax, key, metric, ylabel in zip(
                axes,
                ("teacher_distances", "teacher_distances", "teacher_update_distances"),
                ("sd_variation", "remaining_fraction", "sd_variation"),
                (
                    "Full weight: SD distance to teacher",
                    "Full weight: distance / initial distance",
                    "Cumulative update: SD distance to teacher",
                ),
                strict=True,
            ):
                for color, (label, subset) in zip(colors, groups, strict=False):
                    valid = [
                        m
                        for m in subset
                        if key in m["sigma_variation"]
                        and all(p.get(metric) is not None for p in m["sigma_variation"][key])
                    ]
                    if not valid:
                        continue
                    x = [p["step"] for p in valid[0]["sigma_variation"][key]]
                    band(ax, x, [[p[metric] for p in m["sigma_variation"][key]] for m in valid], color, label)
                ax.set_xlabel("Optimizer updates")
                ax.set_ylabel(ylabel)
            axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2)
            fig.suptitle(f"{name}: teacher spectral distance and remaining fraction")
            save_figure(
                fig,
                destination / f"{name}-sigma-teacher-distance",
                f"Median and IQR; {coverage}. Relative distance excludes near-zero baselines.",
            )

        fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
        fig.get_layout_engine().set(rect=(0, 0.09, 1, 0.92))
        events = pair["risks"]
        risksteps = [e["optimizer_steps"] for e in events]
        for ax, bank in zip(axes[0], ("public_initial", "public_teacher"), strict=True):
            for color, domain in zip(colors, ("all", "math", "code"), strict=True):
                ax.plot(
                    risksteps,
                    [e["metrics"][f"eval/rkl/{bank}/{domain}/question_mean"] for e in events],
                    color=color,
                    label=domain,
                )
            ax.set_xlabel("Optimizer updates")
            ax.set_ylabel(f"{bank}: absolute held-out RKL")
            ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=4)
        # IF is included in all-domain RKL; explicit IF curve uses a distinct linestyle.
        for ax, bank in zip(axes[0], ("public_initial", "public_teacher"), strict=True):
            ax.plot(
                risksteps,
                [e["metrics"][f"eval/rkl/{bank}/if/question_mean"] for e in events],
                color=colors[2],
                linestyle="--",
                label="if",
            )
            ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=4)
        for ax, kind, key in [
            (axes[1, 0], "weight", "teacher_distances"),
            (axes[1, 1], "update", "teacher_update_distances"),
        ]:
            if key not in matrices[0]["sigma_variation"]:
                ax.text(0.5, 0.5, "Teacher coordinates unavailable", ha="center", va="center", transform=ax.transAxes)
            else:
                steps = [p["step"] for p in matrices[0]["sigma_variation"][key]]
                for color, (label, subset) in zip(colors, groups, strict=False):
                    band(ax, steps, [[p["hellinger"] for p in m["sigma_variation"][key]] for m in subset], color, label)
                ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=4)
            ax.set_xlabel("Optimizer updates")
            ax.set_ylabel(f"{kind}: Hellinger distance to teacher")
        fig.suptitle(f"{name}: held-out behavior and teacher spectral distance")
        save_figure(
            fig,
            destination / f"{name}-sigma-teacher",
            f"Spectral median and IQR: {coverage}. RKL uses both full fixed banks.",
        )
