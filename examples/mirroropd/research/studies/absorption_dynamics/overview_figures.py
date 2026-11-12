# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Behavioral overview and complete-coverage cross-pair comparisons."""

import numpy as np

from examples.mirroropd.research.core.plotting import save_figure

from .geometry_plotting import band, grid, heatmap, legend, setup
from .geometry_probes import relation


def render(data, destination):
    _, fv = setup()
    colors = fv.series_colors("paper", 5)
    names = list(data["pairs"])
    domains = ("all", "math", "code", "if")
    fig, axes = grid(2, 2, width=11, height=8)
    for col, bank in enumerate(("public_initial", "public_teacher")):
        absolute = []
        relative = []
        for name in names:
            risks = data["pairs"][name]["risks"]
            a, b = risks[0]["metrics"], risks[-1]["metrics"]
            keys = [f"eval/rkl/{bank}/{d}/question_mean" for d in domains]
            absolute.append([np.log10(max(b[k], 1e-12)) for k in keys])
            relative.append([b[k] / a[k] for k in keys])
        heatmap(fig, axes[0, col], absolute, domains, names, "log10 absolute endpoint RKL")
        heatmap(
            fig, axes[1, col], relative, domains, names, "Endpoint / initial RKL", vmin=0, vmax=max(1, np.max(relative))
        )
        for row in (0, 1):
            axes[row, col].set_xlabel("Held-out domain")
            axes[row, col].set_ylabel(bank)
    fig.suptitle("Completed training: domain-resolved absorption")
    save_figure(
        fig,
        destination / "overview-rkl-endpoints",
        "All banks use complete valid responses; all = equal-weight mean of three domains. Panel scales differ.",
    )

    rows = (len(names) + 2) // 3
    fig, axes = grid(rows, 3, width=13, height=3.4 * rows)
    for ax in axes.flat[len(names) :]:
        ax.set_axis_off()
    for ax, name in zip(axes.flat, names, strict=False):
        pair = data["pairs"][name]
        events = pair["risks"]
        x = [e["optimizer_steps"] for e in events]
        for color, bank in zip(colors, ("public_initial", "public_teacher"), strict=False):
            for linestyle, domain in [("-", "all"), ("--", "math")]:
                key = f"eval/rkl/{bank}/{domain}/question_mean"
                ax.plot(
                    x,
                    [e["metrics"][key] for e in events],
                    color=color,
                    linestyle=linestyle,
                    label=f"{bank.replace('public_', '')} {domain}",
                )
        ax.set_title(name)
        ax.set_xlabel("Optimizer updates")
        ax.set_ylabel("Absolute RKL")
        ax.set_yscale("log")
    legend(axes[0, 0], 2)
    fig.suptitle("All-domain and Math held-out trajectories")
    save_figure(
        fig,
        destination / "overview-rkl-trajectories",
        "One seed per pair; fixed initial and teacher banks. Training is complete for all displayed pairs.",
    )

    complete = [(name, pair) for name, pair in data["pairs"].items() if pair["complete"]]
    if len(complete) < 2:
        return
    fig, axes = grid(2, 3, width=13, height=7.8)
    metrics = [
        ("overlap", "Early/final overlap k64"),
        ("capture", "Early capture / optimal k64"),
        ("rank", "Final stable rank"),
        ("norm", "Relative update norm"),
        ("sigma", "Full-weight SD drift"),
        ("residual", "Parameter residual energy ratio"),
    ]
    for ax, (key, label) in zip(axes.flat, metrics, strict=True):
        for i, (name, pair) in enumerate(complete):
            ms = pair["matrices"]
            vals = []
            for m in ms:
                if key == "overlap":
                    v = relation(m, 16, 640, 64)["adjusted_overlap"]
                elif key == "capture":
                    v = relation(m, 16, 640, 64)["a_captures_b"] / next(
                        r["captured_energy"] for r in m["stages"][-1]["ranks"] if r["requested_rank"] == 64
                    )
                elif key == "rank":
                    v = m["stages"][-1]["stable_rank"]
                elif key == "norm":
                    v = np.sqrt(m["stages"][-1]["energy"] / m["stages"][-1]["weight"]["base_energy"])
                elif key == "sigma":
                    v = next(
                        p["sd_variation"]
                        for p in m["sigma_variation"]["weight"]["pairs"]
                        if p["step_a"] == 0 and p["step_b"] == 640
                    )
                else:
                    if not pair["teacher_parameter_comparison"]:
                        continue
                    v = m["stages"][-1]["teacher_comparison"]["residual_energy_ratio"]
                vals.append(v)
            if not vals:
                continue
            events = pair["risks"]
            keyr = "eval/rkl/public_teacher/all/question_mean"
            y = events[-1]["metrics"][keyr]
            ax.scatter([np.median(vals)], [y], color=colors[i % len(colors)], s=35, label=name)
            # Readable pair IDs in the legend, not text over data.
        ax.set_xlabel(label)
        ax.set_ylabel("Teacher-bank endpoint average RKL")
        ax.set_yscale("log")
    legend(axes[0, 0], min(5, len(complete)))
    fig.suptitle("Complete pairs only: geometry and output matching")
    save_figure(
        fig,
        destination / "overview-geometry-rkl",
        f"{len(complete)} complete pairs; one point per pair. Descriptive association, without causal fitting.",
    )

    # Match matrix coordinates and layer coverage within each shared student family.
    families = {}
    for name, pair in complete:
        families.setdefault(pair["student"], []).append((name, pair))
    for family, items in families.items():
        if len(items) < 2:
            continue
        keys = set.intersection(*[{m["tensor"] for m in p["matrices"]} for _, p in items])
        fig, axes = grid(1, 3, height=4.6)
        for i, (name, pair) in enumerate(items):
            ms = [m for m in pair["matrices"] if m["tensor"] in keys]
            steps = [s["step"] for s in ms[0]["stages"]]
            band(
                axes[0, 0],
                steps,
                [[relation(m, t, 640, 64)["adjusted_overlap"] for t in steps] for m in ms],
                colors[i % 5],
                name,
            )
            band(axes[0, 1], steps, [[s["stable_rank"] for s in m["stages"]] for m in ms], colors[i % 5], name)
            band(
                axes[0, 2],
                steps,
                [[np.sqrt(s["energy"] / s["weight"]["base_energy"]) for s in m["stages"]] for m in ms],
                colors[i % 5],
                name,
            )
        for ax, label in zip(
            axes[0], ("Early/final overlap k64", "Cumulative-update stable rank", "Relative update norm"), strict=True
        ):
            ax.set_xlabel("Optimizer updates")
            ax.set_ylabel(label)
        legend(axes[0, 0], len(items))
        fig.suptitle("Shared student: " + family)
        save_figure(
            fig,
            destination / f"overview-shared-student-{items[0][0]}",
            f"{len(keys)} identical matrix coordinates, complete pairs only. Median and IQR across matrices.",
        )
