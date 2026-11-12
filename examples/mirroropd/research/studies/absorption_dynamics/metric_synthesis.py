# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Audit complete cohorts, spectral trajectories and paired sigma0 interventions.

Ledger: statistical units are pairs; matrix medians describe each pair. Heatmap
cells are one pair/rank or one feature/outcome correlation, never seed replicates.
All figures can be reproduced from the exported JSON without run directories.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from examples.mirroropd.research.core.io import dump
from examples.mirroropd.research.core.plotting import save_figure

from .cohort_comparison import screen, shape_distance, summarize
from .geometry_plotting import grid, heatmap, setup
from .geometry_probes import RANKS, spectrum_metrics


def spectral_point(values, step):
    values = np.asarray(values)
    return dict(step=step, probabilities=(values / values.sum()).tolist(), nuclear_norm=float(values.sum()))


def enrich(cohort, sources, swap):
    """Validate immutable matrix identities, then export scalar-only trajectories."""
    audit = dict(residual_identity_max_error=0.0, matrices=0)
    for pair in cohort["pairs"].values():
        if pair["measured"] != pair["expected"]:
            raise ValueError("Synthesis requires complete pair coverage")
    lookup = {(n, m["tensor"]): m for n, p in cohort["pairs"].items() for m in p["matrices"]}
    seen = set()
    for directory in sources:
        for path in sorted(directory.glob("matrix-*.json.gz")):
            payload = path.read_bytes()
            raw = json.loads(gzip.decompress(payload))
            task = raw["task"]
            identity = (task["condition"], task["tensor"])
            if identity in seen or identity not in lookup:
                raise ValueError("Duplicate or unexpected matrix")
            seen.add(identity)
            m = lookup[identity]
            if hashlib.sha256(payload).hexdigest() != m["source_sha256"]:
                raise ValueError("Matrix differs from exported cohort evidence")
            base = spectral_point(raw["initial_singular_values"], 0)
            teacher = raw.get("teacher")
            teacher_s = teacher["weight"]["singular_values"] if teacher else None
            baseline = shape_distance(raw["initial_singular_values"], teacher_s) if teacher else None
            trajectory = []
            for stage in raw["stages"]:
                point = spectral_point(stage["weight"]["singular_values"], stage["step"])
                probes = spectrum_metrics(point, base)
                probes.update(
                    update_energy=stage["energy"],
                    update_stable_rank=stage["stable_rank"],
                    update_rank90=stage["rank90"],
                    update_rank95=stage["rank95"],
                )
                if teacher:
                    probes["teacher_spectrum_remaining"] = (
                        shape_distance(stage["weight"]["singular_values"], teacher_s) / baseline
                        if baseline > 1e-12
                        else None
                    )
                    tc = stage["teacher_comparison"]
                    error = abs(
                        tc["residual_energy_ratio"] - (1 + tc["norm_ratio"] ** 2 - 2 * tc["norm_ratio"] * tc["cosine"])
                    )
                    audit["residual_identity_max_error"] = max(audit["residual_identity_max_error"], error)
                trajectory.append(probes)
            m["spectral_trajectory"] = trajectory
            final = raw["stages"][-1]
            values = m["values"]
            for key in ("scale", "shape_residual", "entropy_rank", "maximum_perturbation"):
                values[f"weight_{key}"] = trajectory[-1][key]
            values["student_rank95_fraction"] = final["rank95"] / min(raw["shape"])
            for label, stage in (("early", raw["stages"][0]), ("final", final)):
                for rank in (16, 64, 256):
                    item = next(p for p in stage["ranks"] if p["requested_rank"] == rank)
                    if item["boundary_relative_gap"] is not None:
                        values[f"{label}_gap{rank}"] = item["boundary_relative_gap"]
            for anchor in (16, 480):
                relation = next(
                    p
                    for p in raw["pairs"]
                    if p["step_a"] == anchor and p["step_b"] == 640 and p["requested_rank"] == 64
                )
                for key in ("mean_angle_deg", "max_angle_deg", "chordal_distance", "random_overlap"):
                    values[f"frame64_{anchor}_{key}"] = relation[key]
            increment = raw["increments"][-1]
            values["late_increment_norm_per_update"] = np.sqrt(increment["energy"]) / increment["optimizer_interval"]
            for anchor in (16, 320):
                capture = next(p for p in increment["captures"] if p["anchor"] == anchor and p["requested_rank"] == 64)
                values[f"late_increment64_fraction_anchor{anchor}"] = capture["capture"]
            if teacher:
                values["teacher_spectrum_remaining"] = trajectory[-1]["teacher_spectrum_remaining"]
            audit["matrices"] += 1
    if seen != set(lookup):
        raise ValueError("Missing matrix evidence")
    for pair in cohort["pairs"].values():
        pair["summary"] = summarize(pair["matrices"])
        pair["depths"] = {
            name: summarize([m for m in pair["matrices"] if lo <= m["layer"] < hi])
            for name, lo, hi in (("shallow", 0, 9), ("middle", 9, 19), ("deep", 19, 28))
        }
        pair["rkl_dynamics"] = {}
        for bank in ("initial", "teacher"):
            for domain in ("math", "code", "if", "all"):
                key = f"eval/rkl/public_{bank}/{domain}/question_mean"
                risks = pair["risks"]
                start, final = risks[0]["metrics"][key], risks[-1]["metrics"][key]
                pair["rkl_dynamics"][f"{bank}/{domain}"] = dict(
                    initial=start,
                    final=final,
                    remaining=final / start,
                    at128=next(r["metrics"][key] for r in risks if r["optimizer_steps"] == 128),
                    at320=next(r["metrics"][key] for r in risks if r["optimizer_steps"] == 320),
                    best=min(r["metrics"][key] for r in risks),
                )
    cohort["descriptive_correlations"] = screen(cohort["pairs"])
    interventions = {}
    for name, pair in swap["pairs"].items():
        records = []
        for stage in pair["stages"]:
            if stage["rows"] != 512 or not stage["no_edit_gate"]["passed"]:
                raise ValueError("Incomplete paired intervention")
            for bank in ("initial", "teacher"):
                for domain in ("math", "code", "if", "all"):
                    key = f"eval/rkl/public_{bank}/{domain}/question_mean"
                    a, b = stage["original"][key], stage["restored"][key]
                    records.append(
                        dict(
                            step=stage["step"],
                            bank=bank,
                            domain=domain,
                            original=a,
                            edited=b,
                            delta=b - a,
                            relative_change=(b - a) / a,
                        )
                    )
        interventions[name] = records
    cohort["sigma0_intervention"] = interventions
    cohort["audit"] = audit
    cohort["schema"] = "absorption-metric-synthesis-v1"
    return cohort


def render(data, destination):
    plt, _ = setup()
    destination.mkdir(parents=True, exist_ok=True)
    names = list(data["pairs"])
    fig, axes = grid(1, 3, width=14, height=6.7)
    for ax, prefix, title in zip(
        axes[0],
        ("overlap", "late", "capture"),
        ("Step16 to final: overlap", "Step480 to final: overlap", "Step16 captures final / optimal"),
        strict=True,
    ):
        values = []
        for name in names:
            s = data["pairs"][name]["summary"]
            keys = [
                f"overlap{k}_{480 if prefix == 'late' else 16}_final"
                if prefix != "capture"
                else f"capture_optimal{k}_16_final"
                for k in RANKS
            ]
            values.append([s[k]["median"] for k in keys])
        heatmap(fig, ax, values, RANKS, names, title, vmin=0, vmax=1)
        ax.set_xlabel("Rank k")
        ax.set_ylabel("Pair")
    fig.suptitle("Early alignment separates pairs better than late stabilization")
    save_figure(
        fig,
        destination / "frame-rank-comparison",
        "196 matrices/pair; medians; cumulative-update right subspaces; common 0–1 color scale.",
    )
    plt.close(fig)

    fig, axes = grid(1, 2, width=12, height=6.8)
    for ax, bank in zip(axes[0], ("initial", "teacher"), strict=True):
        values = [
            [np.log10(p["rkl_dynamics"][f"{bank}/{d}"]["final"]) for d in ("math", "code", "if")]
            for p in data["pairs"].values()
        ]
        heatmap(fig, ax, values, ["Math", "Code", "IF"], names, "log10 final RKL", vmin=-4, vmax=0.3)
        for i, row in enumerate(values):
            for j, v in enumerate(row):
                ax.text(
                    j,
                    i,
                    f"{10**v:.4g}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1),
                )
        ax.set_title(f"{bank} bank")
        ax.set_xlabel("Held-out domain")
        ax.set_ylabel("Pair")
    fig.suptitle("Domain-resolved absolute RKL after 640 updates")
    save_figure(
        fig,
        destination / "domain-rkl",
        "Each cell is a question-mean RKL; Math128/Code64/IF64; no task accuracy measurement.",
    )
    plt.close(fig)

    fields = [
        ("overlap64_16_final", "Early overlap k64"),
        ("overlap64_480_final", "Late overlap k64"),
        ("capture_optimal64_16_final", "Early energy capture / optimal"),
        ("teacher_global_amplitude", "Teacher relative update norm"),
        ("student_global_amplitude", "OPD relative update norm"),
        ("parameter_norm_ratio", "OPD / teacher update norm"),
        ("parameter_cosine", "Update cosine to teacher"),
        ("parameter_residual_energy_ratio", "Parameter residual energy"),
        ("teacher_update_overlap64", "Final frame / teacher frame"),
        ("teacher_kappa_spec", "Teacher kappa_spec"),
        ("student_kappa_spec", "OPD kappa_spec"),
        ("student_rank90_fraction", "OPD rank90 / q"),
        ("final_energy16", "OPD top16 energy fraction"),
        ("student_weight_sd", "Full-weight spectrum drift"),
        ("student_update_sd", "Update spectrum drift"),
        ("teacher_spectrum_remaining", "Remaining teacher spectrum distance"),
    ]
    fig, axes = grid(1, 2, width=13, height=8.2)
    for ax, bank in zip(axes[0], ("public_initial", "public_teacher"), strict=True):
        vals = [
            [
                next(
                    r["spearman"]
                    for r in data["descriptive_correlations"]
                    if r["feature"] == key and r["bank"] == bank and r["domain"] == domain and not r["relative"]
                )
                for domain in ("math", "all")
            ]
            for key, _ in fields
        ]
        heatmap(
            fig,
            ax,
            vals,
            ["Math RKL", "All-domain RKL"],
            [label for _, label in fields],
            "Spearman rho",
            vmin=-1,
            vmax=1,
            cmap="coolwarm",
        )
        ax.set_yticks(range(len(fields)), [label for _, label in fields], fontsize=8)
        for i, row in enumerate(vals):
            for j, v in enumerate(row):
                ax.text(
                    j,
                    i,
                    f"{v:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1),
                )
        ax.set_title(bank.replace("public_", "") + " bank")
        ax.set_xlabel("Final absolute RKL outcome")
        ax.set_ylabel("Pair-level geometric feature")
    fig.suptitle("Descriptive feature screen: correlation is not causation")
    save_figure(
        fig,
        destination / "metric-screen",
        "n=11 pairs, or n=10 for teacher parameter features (cross-size E102 excluded); no multiple-testing inference.",
    )
    plt.close(fig)

    fig, axes = grid(2, 2, width=13, height=10)
    bins = [f"p{a:02d}_{a + 10:02d}" for a in range(0, 100, 10)]
    for row, domain in enumerate(("math", "if")):
        for col, bank in enumerate(("initial", "teacher")):
            values = [
                [
                    np.log10(p["risks"][-1]["metrics"][f"eval/rkl/public_{bank}/{domain}/position/{b}/mean"])
                    for b in bins
                ]
                for p in data["pairs"].values()
            ]
            ax = axes[row, col]
            heatmap(
                fig,
                ax,
                values,
                [f"{a}–{a + 10}" for a in range(0, 100, 10)],
                names,
                "log10 token-mean RKL",
                vmin=-4,
                vmax=0.3,
            )
            ax.set_title(f"{bank} bank / {domain}")
            ax.set_xlabel("Relative response position (%)")
            ax.set_ylabel("Pair")
    fig.suptitle("Residual mismatch is not uniformly a late-position problem")
    save_figure(
        fig,
        destination / "position-rkl",
        "Relative response deciles; token-weighted within bins; different aggregation from question-mean RKL.",
    )
    plt.close(fig)

    steps = sorted({r["step"] for rows in data["sigma0_intervention"].values() for r in rows})
    fig, axes = grid(1, 2, width=13, height=6.7)
    for ax, relative in zip(axes[0], (False, True), strict=True):
        values = []
        for name in names:
            rows = data["sigma0_intervention"][name]
            values.append(
                [
                    np.log10(
                        max(
                            abs(r["relative_change"] if relative else r["delta"])
                            for r in rows
                            if r["step"] == step and r["domain"] != "all"
                        )
                    )
                    for step in steps
                ]
            )
        heatmap(fig, ax, values, steps, names, "log10 max |relative change|" if relative else "log10 max |RKL change|")
        ax.set_xlabel("Optimizer updates")
        ax.set_ylabel("Pair")
    fig.suptitle("Sigma0 intervention: absolute and relative effects")
    save_figure(
        fig,
        destination / "sigma0-effects",
        "Each cell takes the largest paired effect over 2 banks × 3 domains; signs retained in exported data.",
    )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path)
    parser.add_argument("--source", type=Path, nargs="+")
    parser.add_argument("--swap", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--figures", type=Path)
    args = parser.parse_args()
    if args.data:
        data = json.loads(args.data.read_text())
    else:
        if not all((args.cohort, args.source, args.swap, args.output)):
            parser.error("Collection requires --cohort, --source, --swap and --output")
        data = enrich(json.loads(args.cohort.read_text()), args.source, json.loads(args.swap.read_text()))
        dump(args.output, data)
    if args.figures:
        render(data, args.figures)


if __name__ == "__main__":
    main()
