# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Full singular spectra, update dimension, scale, spectral gaps and frame motion."""

import numpy as np

from .geometry_plotting import band, finish, grid, heatmap, legend, setup
from .geometry_probes import MODULES, rank_record, relation


def singular(point):
    return np.asarray(point["probabilities"]) * point["nuclear_norm"]


def render(data, destination):
    _, fv = setup()
    colors = fv.series_colors("paper", 5)
    for name, pair in data["pairs"].items():
        ms = [m for m in pair["matrices"] if m["layer"] in pair["complete_layers"] and "spectral_shape" in m]
        if not ms:
            continue
        steps = [s["step"] for s in ms[0]["stages"]]
        fig, axes = grid(2, 3, height=7.8)
        probes = [
            ("norm", "Update Frobenius norm"),
            ("relative_norm", "Update / initial weight norm"),
            ("stable_rank", "Update stable rank"),
            ("rank90", "Rank for 90% update energy"),
            ("rank95", "Rank for 95% update energy"),
            ("entropy", "Energy-entropy effective rank"),
        ]
        for ax, (key, label) in zip(axes.flat, probes, strict=True):
            for color, group in zip(colors, ("Attention", "MLP"), strict=False):
                subset = [m for m in ms if (m["module"] in MODULES[:4]) == (group == "Attention")]
                rows = []
                for m in subset:
                    if key == "norm":
                        v = [np.sqrt(s["energy"]) for s in m["stages"]]
                    elif key == "relative_norm":
                        v = [np.sqrt(s["energy"] / s["weight"]["base_energy"]) for s in m["stages"]]
                    elif key == "entropy":
                        v = [s["entropy_rank"] for s in m["spectral_shape"]["update"]]
                    else:
                        v = [s[key] for s in m["stages"]]
                    rows.append(v)
                band(ax, steps, rows, color, group)
            ax.set_xlabel("Optimizer updates")
            ax.set_ylabel(label)
        legend(axes[0, 0], 2)
        finish(
            fig,
            destination,
            name,
            "update-dimension",
            pair,
            ms,
            (
                "Update magnitude and intrinsic dimension",
                "Matrix medians and IQR; entropy uses normalized squared singular values.",
            ),
        )

        # Full spectra remain exact in data; curves use a common normalized-rank grid.
        rankgrid = np.geomspace(1e-3, 1, 160)
        fig, axes = grid(1, 3, height=4.9)
        for ax, kind in zip(axes[0], ("weight", "update", "increment"), strict=True):
            indices = (0, 3, 5, 8) if kind != "increment" else (0, 2, 4, 7)
            for color, index in zip(colors, indices, strict=False):
                rows = []
                for m in ms:
                    if kind == "increment":
                        source = m["increments"][index]["spectrum_cdf"]
                        x = np.asarray(source["ranks"]) / source["ranks"][-1]
                        y = source["energy_fraction"]
                    else:
                        points = m["sigma_variation"][kind]["points"]
                        point = points[index + (1 if kind == "weight" else 0)]
                        s = singular(point)
                        x = np.arange(1, len(s) + 1) / len(s)
                        y = np.cumsum(s * s) / np.sum(s * s)
                    rows.append(np.interp(rankgrid, x, y))
                label = (
                    f"t={steps[index]}"
                    if kind != "increment"
                    else f"{ms[0]['increments'][index]['start']}-{ms[0]['increments'][index]['end']}"
                )
                band(ax, rankgrid, rows, color, label)
            ax.set_xscale("log")
            ax.set_ylim(0, 1.02)
            ax.set_xlabel("Retained rank / full singular dimension")
            ax.set_ylabel(f"{kind}: cumulative energy fraction")
            legend(ax, 2)
        finish(
            fig,
            destination,
            name,
            "singular-energy-rank",
            pair,
            ms,
            (
                "How many directions carry the energy?",
                "Median and IQR; increments use stored 65-point energy CDF interpolation.",
            ),
        )

        # Deterministic representative layer: middle of the measured complete-layer set.
        layer = pair["complete_layers"][len(pair["complete_layers"]) // 2]
        representative = sorted([m for m in ms if m["layer"] == layer], key=lambda m: MODULES.index(m["module"]))
        fig, axes = grid(2, 4, width=15, height=7.8)
        for ax, m in zip(axes.flat, representative, strict=False):
            points = m["sigma_variation"]["weight"]["points"]
            for color, index in zip(colors, (0, 1, 4, 9), strict=False):
                point = points[index]
                s = singular(point)
                ax.plot(np.arange(1, len(s) + 1), s, color=color, label=f"t={point['step']}")
            teacher = m["sigma_variation"].get("teacher_weight")
            if teacher:
                s = singular(teacher)
                ax.plot(np.arange(1, len(s) + 1), s, color=colors[4], linestyle="--", label="teacher")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Singular rank")
            ax.set_ylabel("Full-weight singular value")
            ax.set_title(m["module"])
        axes.flat[-1].set_axis_off()
        legend(axes[0, 0], 3)
        finish(
            fig,
            destination,
            name,
            "singular-profiles",
            pair,
            ms,
            (
                "Actual full-weight singular spectra",
                f"Representative layer {layer}, chosen as median measured layer; overlapping curves are expected.",
            ),
        )

        fig, axes = grid(2, 4, width=15, height=7.8)
        for ax, m in zip(axes.flat, representative, strict=False):
            points = m["sigma_variation"]["update"]["points"]
            for color, index in zip(colors, (0, 3, 5, 8), strict=False):
                p = points[index]
                s = singular(p)
                positive = s > 0
                ax.plot(np.arange(1, len(s) + 1)[positive], s[positive], color=color, label=f"t={p['step']}")
            teacher = m["sigma_variation"].get("teacher_update")
            if teacher and teacher["probabilities"]:
                s = singular(teacher)
                positive = s > 0
                ax.plot(
                    np.arange(1, len(s) + 1)[positive], s[positive], color=colors[4], linestyle="--", label="teacher"
                )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Singular rank")
            ax.set_ylabel("Cumulative-update singular value")
            ax.set_title(m["module"])
        axes.flat[-1].set_axis_off()
        legend(axes[0, 0], 3)
        finish(
            fig,
            destination,
            name,
            "singular-update-profiles",
            pair,
            ms,
            (
                "Actual cumulative-update singular spectra",
                f"Representative layer {layer}; zero singular values omitted from log axes.",
            ),
        )

        fig, axes = grid(2, 4, width=15, height=7.8)
        for ax, m in zip(axes.flat, representative, strict=False):
            points = m["sigma_variation"]["weight"]["points"]
            base = singular(points[0])
            ranks = np.unique(np.rint(np.linspace(0, len(base) - 1, 80)).astype(int))
            values = np.array([(singular(p) - base)[ranks] / base[0] for p in points]).T
            limit = max(float(np.abs(values).max()), 1e-15)
            heatmap(
                fig,
                ax,
                values,
                [p["step"] for p in points],
                (ranks + 1).tolist(),
                "SV change / initial top SV",
                vmin=-limit,
                vmax=limit,
                cmap="coolwarm",
            )
            ax.set_title(m["module"])
            ax.set_xlabel("Optimizer updates")
            ax.set_ylabel("Singular rank (linear samples)")
        axes.flat[-1].set_axis_off()
        finish(
            fig,
            destination,
            name,
            "singular-rank-time",
            pair,
            ms,
            (
                "Which singular ranks actually change?",
                f"Representative layer {layer}; 80 exact rank samples. Signed color scales are panel-specific.",
            ),
        )

        fig, axes = grid(2, 3, height=7.4)
        for row, kind in enumerate(("weight", "update")):
            points = ms[0]["sigma_variation"][kind]["points"]
            x = [p["step"] for p in points]
            for col, key in enumerate(("scale", "shape_residual", "maximum_perturbation")):
                for color, group in zip(colors, ("Attention", "MLP"), strict=False):
                    subset = [m for m in ms if (m["module"] in MODULES[:4]) == (group == "Attention")]
                    band(axes[row, col], x, [[s[key] for s in m["spectral_shape"][kind]] for m in subset], color, group)
                axes[row, col].set_xlabel("Optimizer updates")
                axes[row, col].set_ylabel(f"{kind}: " + key.replace("_", " "))
        legend(axes[0, 0], 2)
        finish(
            fig,
            destination,
            name,
            "singular-scale-shape",
            pair,
            ms,
            (
                "Uniform scaling versus spectral reshaping",
                "Weight reference: step 0. Cumulative-update reference: step 16. Matrix medians and IQR.",
            ),
        )

        fig, axes = grid(2, 3, height=7.5)
        layerlabels = pair["complete_layers"]
        for col, (key, label) in enumerate(
            (("stable_rank", "Stable rank"), ("rank95", "Rank95"), ("norm", "Relative update norm"))
        ):
            for row, index in enumerate((0, 8)):
                values = []
                for layer in layerlabels:
                    values.append(
                        [
                            np.median(
                                [
                                    np.sqrt(m["stages"][index]["energy"] / m["stages"][index]["weight"]["base_energy"])
                                    if key == "norm"
                                    else m["stages"][index][key]
                                    for m in ms
                                    if m["layer"] == layer and m["module"] == module
                                ]
                            )
                            for module in MODULES
                        ]
                    )
                heatmap(fig, axes[row, col], values, [m.replace("_proj", "") for m in MODULES], layerlabels, label)
                axes[row, col].set_xlabel("Projection")
                axes[row, col].set_ylabel(f"t={steps[index]}: layer")
        finish(
            fig,
            destination,
            name,
            "singular-layer-modules",
            pair,
            ms,
            ("Depth-resolved update size and rank", "Each cell is one matrix; color scales are panel-specific."),
        )

        fig, axes = grid(1, 3, height=4.8)
        for ax, k in zip(axes[0], (16, 64, 256), strict=True):
            for color, index in zip(colors, (0, 3, 5, 7), strict=False):
                gaps = [rank_record(m["stages"][index], k)["boundary_relative_gap"] for m in ms]
                angles = [relation(m, steps[index], steps[index + 1], k)["chordal_distance"] for m in ms]
                mask = np.asarray([g is not None and g > 0 for g in gaps])
                ax.scatter(
                    np.asarray(gaps, dtype=float)[mask],
                    np.asarray(angles)[mask],
                    color=color,
                    s=10,
                    alpha=0.4,
                    label=f"{steps[index]}-{steps[index + 1]}",
                )
            ax.set_xscale("log")
            ax.set_xlabel("Starting update spectrum: relative gap")
            ax.set_ylabel(f"k={k}: adjacent-frame chordal distance")
        legend(axes[0, 0], 2)
        finish(
            fig,
            destination,
            name,
            "singular-gap-angles",
            pair,
            ms,
            (
                "Spectral gaps and boundary sensitivity",
                "Each point is a matrix; selected intervals shown. Association does not establish cause.",
            ),
        )

        fig, axes = grid(1, 2, height=5)
        for ax, kind in zip(axes[0], ("weight", "update"), strict=True):
            for color, index in zip(colors, (0, 2, 4, 7), strict=False):
                # Weight intervals start at zero; use the same 16+ intervals as update frames.
                offset = 1 if kind == "weight" else 0
                x = [m["sigma_variation"][kind]["intervals"][index + offset]["sd_variation"] for m in ms]
                y = [relation(m, steps[index], steps[index + 1], 64)["chordal_distance"] for m in ms]
                ax.scatter(x, y, s=12, color=color, alpha=0.45, label=f"{steps[index]}-{steps[index + 1]}")
            ax.set_xscale("log")
            ax.set_xlabel(f"{kind}: interval SD variation")
            ax.set_ylabel("Update frame chordal distance, k=64")
            legend(ax, 2)
        finish(
            fig,
            destination,
            name,
            "singular-frame-phase",
            pair,
            ms,
            (
                "Spectral shape and update frame motion",
                "Each point is a matrix; x=interval net shape change, y=cumulative-update subspace change.",
            ),
        )

        # Quantify full-weight versus cumulative-update frame motion on the same side.
        fig, axes = grid(1, 3, height=4.6)
        for ax, k in zip(axes[0], (16, 64, 256), strict=True):
            band(
                ax,
                steps,
                [
                    [
                        next(f["median_angle_deg"] for f in s["weight"]["frames"] if f["requested_rank"] == k)
                        for s in m["stages"]
                    ]
                    for m in ms
                ],
                colors[0],
                "W0 to Wt",
            )
            band(
                ax,
                steps,
                [[relation(m, steps[0], t, k)["median_angle_deg"] for t in steps] for m in ms],
                colors[1],
                "delta16 to deltat",
            )
            ax.set_xlabel("Optimizer updates")
            ax.set_ylabel(f"k={k}: median principal angle (degrees)")
        legend(axes[0, 0], 2)
        finish(
            fig,
            destination,
            name,
            "weight-update-frames",
            pair,
            ms,
            (
                "Full weights and updates have different frames",
                "Both use right subspaces. Different references: W0 versus cumulative update at step 16.",
            ),
        )

        if pair["teacher_parameter_comparison"]:
            fig, axes = grid(1, 3, height=5)
            for ax, key, label in zip(
                axes[0],
                ("norm_ratio", "cosine", "residual_energy_ratio"),
                ("OPD / teacher norm", "Update cosine", "Residual / teacher energy"),
                strict=True,
            ):
                values = [
                    [
                        next(
                            m["stages"][-1]["teacher_comparison"][key]
                            for m in ms
                            if m["layer"] == layer and m["module"] == module
                        )
                        for module in MODULES
                    ]
                    for layer in layerlabels
                ]
                heatmap(fig, ax, values, [m.replace("_proj", "") for m in MODULES], layerlabels, label)
                ax.set_xlabel("Projection")
                ax.set_ylabel("Transformer layer")
            finish(
                fig,
                destination,
                name,
                "residual-layer-modules",
                pair,
                ms,
                (
                    "Which layers retain teacher parameter residuals?",
                    "One matrix per cell; endpoint values, not functional importance. Panel scales differ.",
                ),
            )
