# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Recompute historical report analyses using only eligible model identities.

Ledger: spectral rows are exact k/v matrices at first/middle/last layers.
Distance dots show teacher or OPD weights relative to the same initial matrix.
Temporal lines summarize three layers, not all layers. Every curve is a
measured checkpoint trajectory; overlap and captured energy are distinct.
"""

import argparse
import hashlib
import json
import runpy
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullFormatter

from examples.mirroropd.research.core.plotting import load_style, save_figure
from examples.mirroropd.research.studies.spectral.absorption.cohort import report_eligible


def invoke(script, root, output):
    previous = sys.argv
    sys.argv = [str(script), "--root", str(root), "--output", str(output)]
    try:
        if script.name == "analyze_direct_pairs.py":
            source = script.read_text().replace(
                "MiMo and controlled-bank risks use separate protocols", "MiMo macro risks use a separate protocol"
            )
            exec(compile(source, str(script), "exec"), {"__name__": "__main__", "__file__": str(script)})
        else:
            runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = previous


def historical(source, output, scratch):
    inv = json.loads((source / "direct-pairs-v1/inventory.json").read_text())
    inv["pairs"] = [p for p in inv["pairs"] if report_eligible(p)]
    inv["conditions"] = [c for c in inv["conditions"] if report_eligible(c)]
    direct = scratch / "direct-pairs-v1"
    direct.mkdir()
    (direct / "inventory.json").write_text(json.dumps(inv, indent=2))
    (output / "inventory.json").write_text(json.dumps(inv, indent=2))
    for name in ("coordinate-configs.json", "matrices", "direction-controls"):
        (direct / name).symlink_to((source / "direct-pairs-v1" / name).resolve(), target_is_directory=True)
    for name in ("curve-spectrum-v1", "spectrum-expanded-v4"):
        (scratch / name).symlink_to((source / name).resolve(), target_is_directory=True)
    target = output / "direct-pairs"
    scripts = source / "sources/direct-cohort"
    invoke(scripts / "analyze_direct_pairs.py", scratch, target)
    (scratch / "direct-pairs-analysis-v4").symlink_to(target.resolve(), target_is_directory=True)
    invoke(scripts / "analyze_direct_associations.py", scratch, target)
    invoke(scripts / "plot_direct_associations.py", scratch, target)
    invoke(scripts / "plot_direct_dynamics.py", scratch, target)
    # No active report table carries excluded identities, including supplements.
    for subdir, filename in [
        ("curve-spectrum-v1/band-analysis-v3", "teacher-band-summary.csv"),
        ("curve-spectrum-v1/band-analysis-v3", "teacher-frame-summary.csv"),
        ("historical-dynamics-20261110", "historical-curves.csv"),
        ("historical-dynamics-20261110", "historical-windows.csv"),
    ]:
        frame = pd.read_csv(source / subdir / filename)
        keep = frame.apply(lambda r: report_eligible(r.to_dict()), axis=1)
        frame[keep].to_csv(output / filename, index=False)
    ancillary = json.loads((source / "spectrum-expanded-v4/ancillary-complete.json").read_text())
    ancillary["conditions"] = [c for c in ancillary["conditions"] if report_eligible(c)]
    (output / "ancillary.json").write_text(json.dumps(ancillary, indent=2))
    manifest = json.loads((source / "expansion-v1/manifest.json").read_text())
    manifest["jobs"] = [job for job in manifest["jobs"] if report_eligible(job)]
    (output / "expansion-manifest.json").write_text(json.dumps(manifest, indent=2))
    shard_audit = json.loads((source / "expansion-v1/checkpoint-shard-audit.json").read_text())
    (output / "checkpoint-shard-audit.json").write_text(
        json.dumps([row for row in shard_audit if report_eligible(row)], indent=2)
    )


def spectral(source, output):
    profiles = json.loads((source / "spectrum-expanded-v4/model-profiles.json").read_text())
    records, provenance = [], []
    for profile in profiles:
        if not report_eligible(profile) or not profile["compatible"]:
            continue
        for path in sorted((source / "spectrum-expanded-v4/runs" / profile["run_id"]).glob("*.json")):
            data = json.loads(path.read_text())
            provenance.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            roles = [("teacher", None, data["teacher_delta"])]
            roles.extend(("opd", stage["step"], stage["delta"]) for stage in data["stages"])
            for role, step, values in roles:
                records.append(
                    dict(
                        run_id=profile["run_id"],
                        teacher=profile["teacher"],
                        student=profile["student"],
                        role=role,
                        step=step,
                        tensor=data["tensor"],
                        delta_sigma=values["rho_sigma"] * values["norm_over_base"],
                        rho_sigma=values["rho_sigma"],
                        kappa_spec=values["kappa_spec"],
                    )
                )
    frame = pd.DataFrame(records)
    frame.to_csv(output / "exact-spectral-matrices.csv", index=False)
    summary = (
        frame.groupby(["run_id", "teacher", "student", "role", "step"], dropna=False)[
            ["delta_sigma", "rho_sigma", "kappa_spec"]
        ]
        .median()
        .reset_index()
    )
    summary.to_csv(output / "exact-spectral-summary.csv", index=False)
    (output / "spectral-coverage.json").write_text(
        json.dumps(
            {
                "compatible_conditions": int(frame.run_id.nunique()),
                "matrix_role_stage_records": len(frame),
                "condition_role_stage_summaries": len(summary),
                "scope": "Six first/middle/last k/v matrices; compatible pairs only; per-condition medians",
            },
            indent=2,
        )
    )
    (output / "spectral-provenance.json").write_text(json.dumps(provenance, indent=2))
    fv = load_style()
    fv.apply_rc("paper")
    # Identical weight pairs with different prompt modes remain distinct in CSV,
    # but are shown once for weight-only geometry.
    teacher = summary[summary.role == "teacher"].drop_duplicates(["teacher", "delta_sigma", "rho_sigma"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 10))
    fig.subplots_adjust(left=0.23, right=0.98, bottom=0.20, top=0.94, wspace=0.28)
    for ax, metric in zip(axes, ("delta_sigma", "rho_sigma", "kappa_spec"), strict=True):
        original, endpoint = [], []
        for row in teacher.itertuples():
            opd = summary[(summary.run_id == row.run_id) & (summary.role == "opd")].sort_values("step").iloc[-1]
            original.append(getattr(row, metric))
            endpoint.append(opd[metric])
        ax.plot([], [], color=fv.HUES["vgray"], label="Pair link")
        ax.hlines(range(len(teacher)), original, endpoint, color=fv.HUES["vgray"], alpha=0.6)
        ax.scatter(original, range(len(teacher)), color=fv.HUES["vorange"], label="Teacher")
        ax.scatter(endpoint, range(len(teacher)), color=fv.HUES["vteal"], label="OPD endpoint")
        ax.set(xscale="log", xlabel=metric, ylabel="Weight pair", ylim=(len(teacher) - 0.4, -0.6))
        ax.set_yticks(
            range(len(teacher)), [f"{r.student}: {r.teacher}" for r in teacher.itertuples()] if ax is axes[0] else []
        )
        ax.grid(axis="x", alpha=0.2)
        ax.xaxis.set_minor_formatter(NullFormatter())
        if metric == "rho_sigma":
            ax.set_xticks([0.02, 0.05, 0.1, 0.5], ["0.02", "0.05", "0.1", "0.5"])
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.6, 0.06), ncol=2)
    save_figure(
        fig,
        output / "spectral-distances",
        "Exact SVD; first/middle/last k/v matrices; per-pair median; OPD budgets differ",
    )


def temporal(source, output):
    rows = []
    arms = ("qwen-grpo", "qwen-sft", "r1-justrl")
    for arm in arms:
        for path in sorted((source / "temporal-svd").glob(f"{arm}-l*.json")):
            data = json.loads(path.read_text())
            initial = np.array(data["base_singular_values"])
            for stage in data["stages"]:
                drift = np.linalg.norm(np.array(stage["weight_singular_values"]) - initial)
                rows.append(
                    dict(
                        arm=arm,
                        tensor=data["tensor"],
                        step=stage["step"],
                        delta_sigma=drift / np.linalg.norm(initial),
                        rho_sigma=drift / np.sqrt(stage["update_energy"]),
                        own_top16=stage["subspaces"]["16"]["own_top_energy"],
                        early_capture=stage["subspaces"]["16"]["early_right_energy"],
                        early_overlap=stage["subspaces"]["16"]["early_right_overlap"],
                    )
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "temporal-exact.csv", index=False)
    grouped = frame.groupby(["arm", "step"]).median(numeric_only=True).reset_index()
    fv = load_style()
    fv.apply_rc("paper")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.20, top=0.92, hspace=0.40, wspace=0.32)
    for col, arm in enumerate(arms):
        g = grouped[grouped.arm == arm]
        axes[0, col].plot(g.step, g.rho_sigma, marker="o", color=fv.HUES["vteal"])
        axes[0, col].set(title=arm, xlabel="OPD update", ylabel="Exact spectral residual rho_sigma", ylim=(0, 0.06))
        for metric, label, color in [
            ("own_top16", "Own top16 energy", "vteal"),
            ("early_capture", "Early-space capture", "vorange"),
            ("early_overlap", "Early/current overlap", "vviolet"),
        ]:
            axes[1, col].plot(g.step, g[metric], marker="o", label=label, color=fv.HUES[color])
        axes[1, col].set(xlabel="OPD update", ylabel="Fraction", ylim=(0, 1.05))
    fig.legend(*axes[1, 0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.07), ncol=3)
    save_figure(
        fig,
        output / "temporal-dynamics",
        "Exact three-layer k_proj medians; Qwen has two checkpoints, JustRL has five; no interpolation",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="report-cohort-", dir=args.scratch_root) as scratch:
        historical(args.source_root, args.output_dir, Path(scratch))
    spectral(args.source_root, args.output_dir)
    temporal(args.source_root, args.output_dir)


if __name__ == "__main__":
    main()
