# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""ISO measured figures. Ledger: each marker is a completed all-layer pair-role median;
rows identify actual teachers, color identifies teacher/student. Axes carry distinct
metrics, not shared scales. Training lines represent one seed/method on the same
fixed heldout Math subset; no token-level pseudo-replication or smoothing.
"""

import argparse
import os
import sys
from pathlib import Path

from examples.mirroropd.research.core.io import dump, read


def render(summary, output):
    sys.path.insert(0, os.environ["FIGURE_VIZ_MPL"])
    import fvstyle as fv
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fv.apply_rc("paper", fontsize=12)
    colors = fv.series_colors("paper", 4)
    audits = []

    def save(fig, name):
        fig.canvas.draw()
        problems = fv.audit(fig)
        audits.append(dict(figure=name, problems=[str(p) for p in problems]))
        fig.savefig(output / f"{name}.png", dpi=180)
        fig.savefig(output / f"{name}.pdf")
        plt.close(fig)

    rows = summary["spectra"]
    names = {r["job_id"]: f"{r['family']} / {r['teacher']} / {r['job_id']}" for r in rows}
    if rows:
        ids = sorted({r["job_id"] for r in rows})
        lookup = {(r["job_id"], r["role"]): r for r in rows}
        fig, axes = plt.subplots(1, 3, figsize=(9, 7.5), sharey=True)
        for ax, key, title, log in zip(
            axes,
            ["delta_sigma_median", "kappa_median", "top16_median"],
            ["Full-weight\nspectral drift", "Diagonal\nenrichment", "Update top-16\nenergy"],
            [True, True, False],
            strict=True,
        ):
            for index, role in enumerate(["teacher", "endpoint"]):
                values = [lookup[job, role][key] for job in ids]
                positions = [i + (index - 0.5) * 0.22 for i in range(len(ids))]
                ax.plot(
                    values,
                    positions,
                    "o" if index == 0 else "s",
                    color=colors[index],
                    label=role,
                    markersize=4,
                    linestyle="none",
                )
            if log:
                ax.set_xscale("log")
            else:
                ax.set_xlim(0, 1)
            ax.set_xlabel(title)
            ax.grid(axis="x", alpha=0.2)
        axes[0].set_yticks(range(len(ids)), [names[i] for i in ids])
        axes[0].set_ylabel("Same-source teacher condition")
        axes[0].invert_yaxis()
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
        fig.text(
            0.5,
            0.02,
            "All seven projections, all layers; each marker is a matrix-wise median.",
            ha="center",
            fontsize=10,
            color=fv.NEUTRALS["baseline"],
        )
        fig.subplots_adjust(left=0.39, right=0.98, bottom=0.19, top=0.92, wspace=0.25)
        save(fig, "all-layer-spectra")
        fig, axes = plt.subplots(1, 2, figsize=(8, 7.5), sharey=True)
        for ax, key, title in zip(
            axes,
            ["radial_update_fraction_median", "spectrum_after_scale_fraction_median"],
            ["Radial update\nenergy fraction", "Spectral residual\nafter scalar fit"],
            strict=True,
        ):
            ax.plot(
                [lookup[job, "teacher"][key] for job in ids],
                range(len(ids)),
                "o",
                color=colors[0],
                markersize=4,
                label="teacher",
            )
            ax.set_xlabel(title)
            ax.set_xlim(-0.04, 1.04)
            ax.set_xticks([0, 0.5, 1])
            ax.grid(axis="x", alpha=0.2)
        axes[0].set_yticks(range(len(ids)), [names[i] for i in ids])
        axes[0].set_ylabel("Teacher condition")
        axes[0].invert_yaxis()
        fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", frameon=False)
        fig.text(
            0.5,
            0.02,
            "Teacher updates; scalar projections from saved FP64 norms and spectra.",
            ha="center",
            fontsize=10,
            color=fv.NEUTRALS["baseline"],
        )
        fig.subplots_adjust(left=0.43, right=0.98, bottom=0.19, top=0.92, wspace=0.3)
        save(fig, "scale-and-shape")
    functional = [
        row
        for row in summary["functional"]
        if row["origin"] == "initial-student"
        and row["split"] == "heldout"
        and row["domain"] == "math"
        and row["condition"]["role"] == "student"
        and "scale" not in row["task"]
    ]
    if functional:
        ids = sorted({int(r["task"].split("-")[-1]) for r in functional})
        lookup = {
            (int(r["task"].split("-")[-1]), r["condition"]["kind"], r["condition"]["alpha"]): r for r in functional
        }
        fig, ax = plt.subplots(figsize=(8, 7.5))
        for index, (kind, alpha, label) in enumerate(
            [
                ("restore", 1.0, "endpoint"),
                ("restore", 0.0, "base spectrum / learned frames"),
                ("reverse", 1.0, "learned spectrum / base frames"),
            ]
        ):
            ax.plot(
                [lookup[i, kind, alpha]["rkl"]["absolute_mean"] for i in ids],
                [j + (index - 1) * 0.2 for j in range(len(ids))],
                ["o", "s", "^"][index],
                color=colors[index],
                markersize=4,
                label=label,
            )
        ax.set_xscale("log")
        ax.set_yticks(range(len(ids)), [names[i] for i in ids])
        ax.invert_yaxis()
        ax.set_ylabel("Student condition")
        ax.set_xlabel("Absolute RKL to original teacher (fixed H128)")
        ax.grid(axis="x", alpha=0.2)
        fig.legend(*ax.get_legend_handles_labels(), loc="upper center", frameon=False, fontsize=10)
        fig.text(
            0.5,
            0.02,
            "Initial-student prefixes, heldout Math; identical questions within each condition.",
            ha="center",
            fontsize=10,
            color=fv.NEUTRALS["baseline"],
        )
        fig.subplots_adjust(left=0.43, right=0.98, bottom=0.14, top=0.85)
        save(fig, "functional-spectrum-swap")
    geometry = [
        r
        for r in summary.get("geometry_all_layer", [])
        if r["transition"] == "base-to-teacher" and r["rank_fraction"] == 0.9 and r["job_id"] in names
    ]
    if geometry:
        ids = sorted({r["job_id"] for r in geometry})
        lookup = {(r["job_id"], r["restriction"]): r for r in geometry}
        fig, ax = plt.subplots(figsize=(8, 7.5))
        for index, kind in enumerate(["mix", "left", "right", "iso"]):
            ax.plot(
                [lookup[i, kind]["median"] for i in ids],
                [j + (index - 1.5) * 0.16 for j in range(len(ids))],
                ["o", "s", "^", "D"][index],
                color=colors[index],
                markersize=4,
                label=kind,
            )
        ax.set_yticks(range(len(ids)), [names[i] for i in ids])
        ax.invert_yaxis()
        ax.set_xlim(-0.04, 1.04)
        ax.set_ylabel("Teacher condition")
        ax.set_xlabel("Unexplained rank-r update ratio (median)")
        ax.grid(axis="x", alpha=0.2)
        fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False)
        fig.text(
            0.5,
            0.02,
            "r = floor(0.9q); ratios use Frobenius norm, not squared energy.",
            ha="center",
            fontsize=10,
            color=fv.NEUTRALS["baseline"],
        )
        fig.subplots_adjust(left=0.43, right=0.98, bottom=0.14, top=0.92)
        save(fig, "rank90-transport")
    for key, figure_name, caption in [
        (
            "training",
            "online-opd-curves",
            "12-prompt diagnostic pilot (not DAPO training); fixed Math eval; one line per seed.",
        ),
        (
            "dapo_training",
            "dapo-opd-curves",
            "DAPO train / fixed Math heldout; 64 updates; one line per seed; equal nominal LR.",
        ),
    ]:
        training = summary.get(key, [])
        if not training:
            continue
        energies = [
            v["rkl"]
            for row in training
            for v in row["scores"]
            if v["origin"] == "initial-student" and v["split"] == "heldout" and v["domain"] == "math"
        ]
        job_ids = [2, 7, 8, 11] if key == "training" else [2, 7, 8, 11, 19, 20]
        nrows = (len(job_ids) + 1) // 2
        fig, axes = plt.subplots(nrows, 2, figsize=(10, 3.5 * nrows), squeeze=False)
        for ax, job_id in zip(axes.flat, job_ids, strict=True):
            for row in training:
                if row["protocol"]["job"]["job_id"] != job_id:
                    continue
                mode = row["protocol"]["mode"]
                selected = [
                    v
                    for v in row["scores"]
                    if v["origin"] == "initial-student" and v["split"] == "heldout" and v["domain"] == "math"
                ]
                selected.sort(key=lambda v: v["step"])
                ax.plot(
                    [v["step"] for v in selected],
                    [v["rkl"] for v in selected],
                    color=colors[["weights", "frames", "spectrum"].index(mode)],
                    label=mode,
                    alpha=0.75,
                    marker="o",
                    markersize=3,
                )
            completed = sum(row["protocol"]["job"]["job_id"] == job_id for row in training)
            ax.set_title(f"Condition {job_id}: {completed}/6 complete")
            ax.set_xlabel("Online OPD update")
            ax.set_ylabel("Absolute heldout RKL (fixed subset)")
            ax.set_yscale("log")
            ax.set_ylim(min(energies) * 0.8, max(energies) * 1.2)
        unique = {}
        for ax in axes.flat:
            handles, labels = ax.get_legend_handles_labels()
            unique.update(zip(labels, handles, strict=True))
        fig.legend(unique.values(), unique.keys(), loc="upper center", ncol=3, frameon=False)
        fig.text(
            0.5,
            0.02,
            caption,
            ha="center",
            fontsize=10,
            color=fv.NEUTRALS["baseline"],
        )
        fig.subplots_adjust(top=0.88, hspace=0.5, wspace=0.35, bottom=0.17)
        save(fig, figure_name)
    dump(output / "figure-audit.json", audits)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    render(read(args.summary), args.output)


if __name__ == "__main__":
    main()
