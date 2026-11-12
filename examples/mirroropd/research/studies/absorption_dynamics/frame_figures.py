# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Right-frame angles, temporal structure, rank dependence and future increments."""

import numpy as np

from .geometry_plotting import band, finish, grid, heatmap, legend, setup
from .geometry_probes import MODULES, RANKS, rank_record, relation


def render(data, destination):
    _, fv = setup()
    colors = fv.series_colors("paper", 5)
    for name, pair in data["pairs"].items():
        ms = [
            m
            for m in pair["matrices"]
            if m["layer"] in pair["complete_layers"] and "angle_distribution" in m["pairs"][0]
        ]
        if not ms:
            continue
        steps = [s["step"] for s in ms[0]["stages"]]
        final = steps[-1]
        anchors = [steps[0], steps[3], steps[5], steps[7]]
        fig, axes = grid(2, 3, height=7.5)
        for col, k in enumerate((16, 64, 256)):
            for color, t in zip(colors, anchors, strict=False):
                rows = [relation(m, t, final, k)["angle_distribution"] for m in ms]
                band(axes[0, col], rows[0]["probabilities"], [r["quantiles_deg"] for r in rows], color, f"t={t}")
                band(axes[1, col], rows[0]["ecdf_degrees"], [r["ecdf_fraction"] for r in rows], color, f"t={t}")
            axes[0, col].set_xlabel("Principal-angle quantile")
            axes[0, col].set_ylabel(f"k={k}: angle (degrees)")
            axes[0, col].set_ylim(0, 91)
            axes[1, col].set_xlabel("Principal angle (degrees)")
            axes[1, col].set_ylabel(f"k={k}: fraction of angles")
            axes[1, col].set_ylim(-0.02, 1.02)
        legend(axes[0, 0], 2)
        finish(
            fig,
            destination,
            name,
            "frame-angles",
            pair,
            ms,
            (
                "Principal-angle shapes and ECDFs",
                "Matrix medians and IQR; quantile curves interpolated at 33 points; ECDF exact at 5-degree grid.",
            ),
        )

        fig, axes = grid(1, 3, height=4.8)
        for ax, k in zip(axes[0], (16, 64, 256), strict=True):
            values = [[np.median([relation(m, a, b, k)["adjusted_overlap"] for m in ms]) for b in steps] for a in steps]
            heatmap(fig, ax, values, steps, steps, "Adjusted overlap", vmin=-0.05, vmax=1)
            ax.set_xlabel("Optimizer updates")
            ax.set_ylabel(f"k={k}: optimizer updates")
        finish(
            fig,
            destination,
            name,
            "frame-pairwise",
            pair,
            ms,
            (
                "All checkpoint frame similarities",
                "Median across matrices; common color scale. Diagonal is self-comparison.",
            ),
        )

        fig, axes = grid(1, 3, height=4.6)
        for ax, key, label in zip(
            axes[0],
            ("adjusted_overlap", "a_captures_b", "optimal"),
            ("Adjusted overlap", "Final energy captured", "Capture / optimal rank-k"),
            strict=True,
        ):
            values = []
            for k in RANKS:
                row = []
                for t in steps:
                    v = [
                        relation(m, t, final, k)["a_captures_b"] / rank_record(m["stages"][-1], k)["captured_energy"]
                        if key == "optimal"
                        else relation(m, t, final, k)[key]
                        for m in ms
                    ]
                    row.append(np.median(v))
                values.append(row)
            heatmap(fig, ax, values, steps, RANKS, label, vmin=-0.05 if key == "adjusted_overlap" else 0, vmax=1)
            ax.set_xlabel("Anchor optimizer updates")
            ax.set_ylabel("Requested rank k")
        finish(
            fig,
            destination,
            name,
            "frame-rank-time",
            pair,
            ms,
            (
                "When do important update directions form?",
                "All five ranks; adjusted overlap and energy coverage are different probes.",
            ),
        )

        fig, axes = grid(2, 4, width=15, height=7.6)
        for ax, module in zip(axes.flat, MODULES, strict=False):
            subset = [m for m in ms if m["module"] == module]
            for color, k in zip(colors, (16, 64, 256), strict=False):
                band(
                    ax,
                    steps,
                    [[relation(m, t, final, k)["adjusted_overlap"] for t in steps] for m in subset],
                    color,
                    f"k={k}",
                )
            ax.set_title(module)
            ax.set_xlabel("Optimizer updates")
            ax.set_ylabel("Adjusted overlap to final")
            ax.set_ylim(-0.05, 1.03)
        axes.flat[-1].set_axis_off()
        legend(axes[0, 0])
        finish(
            fig,
            destination,
            name,
            "frame-modules",
            pair,
            ms,
            (
                "Seven module trajectories",
                "Median and IQR across the same complete layers; cumulative-update right frames.",
            ),
        )

        fig, axes = grid(1, 3, height=5)
        layers = pair["complete_layers"]
        for ax, t in zip(axes[0], (16, 64, 192), strict=True):
            values = [
                [
                    np.median(
                        [
                            relation(m, t, final, 64)["adjusted_overlap"]
                            for m in ms
                            if m["layer"] == layer and m["module"] == module
                        ]
                    )
                    for module in MODULES
                ]
                for layer in layers
            ]
            heatmap(
                fig,
                ax,
                values,
                [x.replace("_proj", "") for x in MODULES],
                layers,
                "Adjusted overlap",
                vmin=-0.05,
                vmax=1,
            )
            ax.set_xlabel("Projection")
            ax.set_ylabel(f"t={t}, k=64: layer")
        finish(
            fig,
            destination,
            name,
            "frame-layer-modules",
            pair,
            ms,
            (
                "Depth and projection resolved frames",
                "Each cell is one measured matrix; same scale across checkpoints.",
            ),
        )

        increments = ms[0]["increments"]
        fig, axes = grid(1, 3, height=5)
        for ax, k in zip(axes[0], (16, 64, 256), strict=True):
            values = []
            for anchor in steps[:-1]:
                values.append(
                    [
                        np.median(
                            [
                                next(
                                    c["relative_to_optimal"]
                                    for c in m["increments"][i]["captures"]
                                    if c["anchor"] == anchor and c["requested_rank"] == k
                                )
                                for m in ms
                            ]
                        )
                        if anchor <= inc["start"]
                        else np.nan
                        for i, inc in enumerate(increments)
                    ]
                )
            heatmap(
                fig,
                ax,
                values,
                [f"{i['start']}-{i['end']}" for i in increments],
                steps[:-1],
                "Capture / optimal",
                vmin=0,
                vmax=1,
            )
            ax.set_xlabel("Future increment interval")
            ax.set_ylabel(f"k={k}: frame anchor")
        finish(
            fig,
            destination,
            name,
            "frame-increments",
            pair,
            ms,
            (
                "Do early frames cover later learning?",
                "Future anchors are blank; energy capture is divided by the optimal same-rank capture.",
            ),
        )

        fig, axes = grid(2, 3, height=7.5)
        x = [i["end"] for i in increments]
        for col, k in enumerate((16, 64, 256)):
            inside = []
            outside = []
            fraction = []
            for m in ms:
                captures = [
                    next(c["capture"] for c in i["captures"] if c["anchor"] == 16 and c["requested_rank"] == k)
                    for i in m["increments"]
                ]
                energy = [i["energy"] for i in m["increments"]]
                inside.append(np.asarray(energy) * captures)
                outside.append(np.asarray(energy) * (1 - np.asarray(captures)))
                fraction.append(1 - np.asarray(captures))
            band(axes[0, col], x, inside, colors[0], "inside")
            band(axes[0, col], x, outside, colors[1], "outside")
            axes[0, col].set_yscale("log")
            axes[0, col].set_xlabel("Increment interval end")
            axes[0, col].set_ylabel(f"k={k}: absolute increment energy")
            band(axes[1, col], x, fraction, colors[1], "outside fraction")
            axes[1, col].set_xlabel("Increment interval end")
            axes[1, col].set_ylabel(f"k={k}: energy outside step-16 frame")
            axes[1, col].set_ylim(0, 1)
        legend(axes[0, 0], 2)
        finish(
            fig,
            destination,
            name,
            "frame-outside-energy",
            pair,
            ms,
            (
                "New energy outside the early update frame",
                "Median and IQR; raw energy belongs to unequal-length intervals, not matched training budgets.",
            ),
        )

        fig, axes = grid(1, 3, height=4.5)
        for ax, k in zip(axes[0], (16, 64, 256), strict=True):
            for color, key in zip(colors, ("median_angle_deg", "p90_angle_deg", "max_angle_deg"), strict=False):
                band(
                    ax,
                    steps,
                    [[relation(m, t, final, k)[key] for t in steps] for m in ms],
                    color,
                    key.replace("_angle_deg", ""),
                )
            ax.set_xlabel("Optimizer updates")
            ax.set_ylabel(f"k={k}: angle to final (degrees)")
            ax.set_ylim(0, 91)
        legend(axes[0, 0])
        finish(
            fig,
            destination,
            name,
            "frame-angle-time",
            pair,
            ms,
            (
                "Typical and extreme directions evolve differently",
                "Matrix medians and IQR; median, P90 and maximum principal angle.",
            ),
        )

        if not pair["teacher_parameter_comparison"]:
            continue
        fig, axes = grid(1, 3, height=4.5)
        for ax, k in zip(axes[0], (16, 64, 256), strict=True):
            band(
                ax,
                steps,
                [
                    [
                        next(
                            f["adjusted_overlap"] for f in s["teacher_comparison"]["frames"] if f["requested_rank"] == k
                        )
                        for s in m["stages"]
                    ]
                    for m in ms
                ],
                colors[0],
                "teacher",
            )
            band(
                ax,
                steps,
                [[relation(m, t, final, k)["adjusted_overlap"] for t in steps] for m in ms],
                colors[1],
                "own final",
            )
            ax.set_xlabel("Optimizer updates")
            ax.set_ylabel(f"k={k}: adjusted overlap")
            ax.set_ylim(-0.05, 1.03)
        legend(axes[0, 0], 2)
        finish(
            fig,
            destination,
            name,
            "frame-teacher-final",
            pair,
            ms,
            (
                "Does the stable space resemble the teacher?",
                "Teacher and student updates share W0; cross-size teacher comparisons are omitted.",
            ),
        )
