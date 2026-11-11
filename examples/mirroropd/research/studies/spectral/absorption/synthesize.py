# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Synthesize complete cross-family interventions and stage gradients on CPU.

Ledger: one row is one manifest condition, never a training replicate. Colors
identify initial/endpoint/off/null consistently. Absolute risks use log axes;
retention uses linear axes. Depth cells are endpoint-normalized improvements,
not additive attributions. Gradient panels separate vector coherence from the
signed cosine of a fixed direction with descent. Each marker is a domain on a
frozen state bank. Layers are disjoint coordinates, not cancelling vectors.
"""

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from examples.mirroropd.research.core.plotting import load_style, save_figure
from examples.mirroropd.research.studies.spectral.absorption.cohort import report_eligible


def read(path, sources):
    payload = path.read_bytes()
    sources.append({"path": str(path), "sha256": hashlib.sha256(payload).hexdigest()})
    return json.loads(payload)


def source_in_cohort(source, ids):
    parts = Path(source["path"]).parts
    return "jobs" not in parts or int(parts[parts.index("jobs") + 1]) in ids


def compact_scores(run, sources):
    jobs = read(run / "manifest.json", sources)["jobs"]
    result = []
    for job in jobs:
        if not report_eligible(job):
            continue
        folder = run / "jobs" / f"{job['job_id']:03d}" / "attempt-001"
        if not (folder / "interventions-complete.json").exists():
            continue
        conditions = {}
        for path in sorted((folder / "scores").glob("*.json")):
            if path.name.endswith(".progress.json"):
                continue
            score = read(path, sources)
            conditions[path.stem] = [
                {k: r[k] for k in ("origin", "row_id", "split", "domain", "tokens", "rkl", "fkl", "js")}
                for r in score["rows"]
            ]
        selection = {}
        for band in ("off", "null"):
            candidates = ["endpoint"] + [f"missing-{band}-beta-{b}" for b in (0.25, 0.5, 1.0)]
            selection[band] = min(
                candidates,
                key=lambda key: np.mean([r["rkl"] for r in conditions[key] if r["split"] == "calibration"]),
            )
        result.append({k: job[k] for k in ("job_id", "run_id", "family", "teacher", "cohort", "step")})
        result[-1].update(conditions=conditions, calibration_selection=selection)
    return result


def risk(job, condition, origin="initial-student", metric="rkl", domain=None):
    return float(
        np.mean(
            [
                r[metric]
                for r in job["conditions"][condition]
                if r["origin"] == origin and r["split"] == "heldout" and (domain is None or r["domain"] == domain)
            ]
        )
    )


def render(jobs, gradients, output):
    fv = load_style()
    fv.apply_rc("paper")
    modes = {8: "SFT thinking", 10: "SFT non-thinking", 20: "Zero from SFT [branch]"}
    labels = [f"{j['job_id']:02d} {j['family']} / {modes.get(j['job_id'], j['teacher'])}" for j in jobs]
    y = np.arange(len(jobs))
    fig, axes = plt.subplots(1, 2, figsize=(14, 9))
    fig.subplots_adjust(left=0.25, right=0.98, bottom=0.17, top=0.93, wspace=0.27)
    for ax, origin, metric, title in zip(
        axes,
        ("initial-student", "teacher"),
        ("rkl", "fkl"),
        ("Initial-student prefix RKL", "Teacher prefix FKL"),
        strict=True,
    ):
        for band, caption, color in [
            ("base", "Initial", "vgray"),
            ("endpoint", "OPD", "vteal"),
        ]:
            values = [risk(j, j["calibration_selection"].get(band, band), origin, metric) for j in jobs]
            ax.scatter(values, y, color=fv.HUES[color], label=caption, s=35, zorder=3)
        for index, job in enumerate(jobs):
            initial = risk(job, "base", origin, metric)
            endpoint = risk(job, "endpoint", origin, metric)
            ax.plot([initial, endpoint], [index, index], color=fv.HUES["vgray"], alpha=0.5, linewidth=2)
        ax.set(xscale="log", xlabel="Absolute KL / H128", title=title, ylim=(len(jobs) - 0.4, -0.6))
        ax.set_yticks(y, labels if ax is axes[0] else [])
        ax.set_ylabel("Condition" if ax is axes[0] else "Same condition order")
        ax.grid(axis="x", alpha=0.2)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.6, 0.08), ncol=2)
    save_figure(
        fig,
        output / "absolute-risk",
        f"{len(jobs)} conditions; 36 heldout prompts; linked dots show initial-to-OPD change on fixed states",
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 9))
    fig.subplots_adjust(left=0.25, right=0.98, bottom=0.17, top=0.93, wspace=0.27)
    for ax, origin, metric, title in zip(
        axes,
        ("initial-student", "teacher"),
        ("rkl", "fkl"),
        ("Initial-student prefix RKL", "Teacher prefix FKL"),
        strict=True,
    ):
        for offset, band, caption, color in [
            (-0.16, "endpoint", "OPD", "vteal"),
            (0, "off", "Selected off", "vorange"),
            (0.16, "null", "Selected null", "vviolet"),
        ]:
            values = [
                risk(j, j["calibration_selection"].get(band, band), origin, metric)
                / risk(j, "endpoint", origin, metric)
                for j in jobs
            ]
            ax.scatter(values, y + offset, color=fv.HUES[color], label=caption, s=25)
        ax.axvline(1, color=fv.HUES["vgray"], linestyle="--", linewidth=1)
        ax.set(
            xlabel="Risk / unedited OPD risk (lower is better)",
            title=title,
            ylim=(len(jobs) - 0.4, -0.6),
            xlim=(0, 1.12),
        )
        ax.set_yticks(y, labels if ax is axes[0] else [])
        ax.set_ylabel("Condition" if ax is axes[0] else "Same condition order")
        ax.grid(axis="x", alpha=0.2)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.6, 0.08), ncol=3)
    save_figure(
        fig, output / "patch-relative", "Beta selected only on calibration RKL of both banks; no heldout selection"
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 10))
    fig.subplots_adjust(left=0.25, right=0.98, bottom=0.16, top=0.95, wspace=0.22)
    for ax, restore, title in zip(
        axes, (False, True), ("Original update amplitude", "Restored per-matrix update norm"), strict=True
    ):
        for offset, fraction, color in [(-0.09, 0.01, "vteal"), (0.09, 0.05, "vorange")]:
            values = [
                (risk(j, "base") - risk(j, f"top-{fraction}-restore-{restore}"))
                / (risk(j, "base") - risk(j, "endpoint"))
                for j in jobs
            ]
            ax.scatter(values, y + offset, color=fv.HUES[color], s=25, label=f"Top {fraction:.0%}")
        ax.axvline(1, color=fv.HUES["vgray"], linestyle="--", linewidth=1)
        ax.set(xlabel="Retained initial-to-OPD RKL improvement", title=title, ylim=(len(jobs) - 0.4, -0.6))
        ax.set_yticks(y, labels if ax is axes[0] else [])
        ax.set_ylabel("Condition" if ax is axes[0] else "Same condition order")
        ax.grid(axis="x", alpha=0.2)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.6, 0.07), ncol=2)
    save_figure(
        fig, output / "compression", "Per-matrix SVD of student-minus-initial updates; initial-student bank, H128"
    )

    fig, ax = plt.subplots(figsize=(9, 10))
    fig.subplots_adjust(left=0.39, right=0.85, bottom=0.12, top=0.95)
    values = np.array(
        [
            [(risk(j, "endpoint") - risk(j, f"missing-off-depth-{d}")) / risk(j, "endpoint") for d in range(3)]
            for j in jobs
        ]
    )
    limit = float(np.max(np.abs(values)))
    from matplotlib.colors import LinearSegmentedColormap

    palette = LinearSegmentedColormap.from_list("signed", [fv.HUES["vviolet"], "white", fv.HUES["vorange"]])
    image = ax.imshow(values, aspect="auto", cmap=palette, vmin=-limit, vmax=limit)
    ax.set_yticks(y, labels)
    ax.set_xticks(range(3), ["Early layers", "Middle layers", "Late layers"])
    for row in range(len(jobs)):
        for col in range(3):
            ax.text(col, row, f"{values[row, col]:+.0%}", ha="center", va="center", fontsize=10)
    fig.colorbar(image, ax=ax, fraction=0.06, pad=0.06, label="Fraction of endpoint RKL removed")
    save_figure(
        fig, output / "depth", "Each block is edited separately at original amplitude; block effects are not additive"
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 10))
    fig.subplots_adjust(left=0.25, right=0.98, bottom=0.17, top=0.95, wspace=0.25)
    selected = [
        r
        for r in gradients
        if r["origin"] == "initial-student"
        and r["split"] == "heldout"
        and r["stage"] == "endpoint"
        and r["scope"] == "all"
    ]
    for domain, color, offset in [("math", "vteal", -0.14), ("code", "vorange", 0), ("if", "vviolet", 0.14)]:
        rows = [next(r for r in selected if r["job_id"] == j["job_id"] and r["domain"] == domain) for j in jobs]
        axes[0].scatter(
            [r["vector_coherence"] for r in rows], y + offset, color=fv.HUES[color], label=domain.upper(), s=22
        )
        axes[1].scatter(
            [r["directions"]["off"]["descent_direction_cosine"] for r in rows], y + offset, color=fv.HUES[color], s=22
        )
    for ax in axes:
        ax.set_ylim(len(jobs) - 0.4, -0.6)
        ax.set_yticks(y, labels if ax is axes[0] else [])
        ax.set_ylabel("Condition" if ax is axes[0] else "Same condition order")
        ax.grid(axis="x", alpha=0.2)
    axes[0].set(
        xlabel="Norm of summed gradients / sum of gradient norms", title="Cross-state vector coherence", xlim=(0, 1.05)
    )
    axes[1].set(
        xlabel="Cosine with descent (positive = locally helpful)", title="Alignment of missing off", xscale="symlog"
    )
    axes[1].set_xscale("symlog", linthresh=1e-4)
    axes[1].set_xlim(-0.01, 0.01)
    axes[1].axvline(0, color=fv.HUES["vgray"], linestyle="--", linewidth=1)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.6, 0.07), ncol=3)
    save_figure(
        fig,
        output / "gradient-alignment",
        "Endpoint, fixed initial-student states; 2 heldout questions x first/last position per domain",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--run-root", type=Path)
    inputs.add_argument("--archive-dir", type=Path, help="Replot compressed JSON without the run directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sources = []
    if args.archive_dir:
        jobs = json.loads(gzip.decompress((args.archive_dir / "question-risks.json.gz").read_bytes()))
        jobs = [j for j in jobs if report_eligible(j)]
        gradients = json.loads(gzip.decompress((args.archive_dir / "gradients.json.gz").read_bytes()))
        paired = json.loads(gzip.decompress((args.archive_dir / "paired.json.gz").read_bytes()))
        sources = json.loads((args.archive_dir / "provenance.json").read_text())
    else:
        jobs = compact_scores(args.run_root, sources)
        gradients = read(args.run_root / "analysis-final/gradients.json", sources)
        paired = read(args.run_root / "analysis-final/interventions/paired.json", sources)
    ids = {j["job_id"] for j in jobs}
    gradients["rows"] = [r for r in gradients["rows"] if r["job_id"] in ids]
    gradients["coverage"] = [r for r in gradients["coverage"] if r["job_id"] in ids]
    gradients["sources"] = [r for r in gradients["sources"] if source_in_cohort(r, ids)]
    sources = [r for r in sources if source_in_cohort(r, ids)]
    paired = [r for r in paired if r["job_id"] in ids and r["split"] == "heldout" and r["normalization"] == "H128"]
    for name, data in [("question-risks", jobs), ("gradients", gradients), ("paired", paired)]:
        (args.output_dir / f"{name}.json.gz").write_bytes(gzip.compress(json.dumps(data).encode(), mtime=0))
    table = []
    for j in jobs:
        row = {k: j[k] for k in ("job_id", "family", "teacher", "cohort", "step")}
        for condition in (
            "base",
            "endpoint",
            "missing-off-beta-1.0",
            "missing-null-beta-1.0",
            "top-0.01-restore-False",
            "top-0.05-restore-False",
        ):
            row[condition] = risk(j, condition)
        for band in ("off", "null"):
            row[f"selected-{band}"] = j["calibration_selection"][band]
            row[f"selected-{band}-rkl"] = risk(j, j["calibration_selection"][band])
        table.append(row)
    with (args.output_dir / "summary.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=table[0])
        writer.writeheader()
        writer.writerows(table)
    (args.output_dir / "provenance.json").write_text(json.dumps(sources, indent=2))
    (args.output_dir / "coverage.json").write_text(
        json.dumps(
            {
                "conditions": len(jobs),
                "stages": len({(r["job_id"], r["stage"]) for r in gradients["rows"]}),
                "gradient_rows": len(gradients["rows"]),
                "score_files": sum(len(j["conditions"]) for j in jobs),
            },
            indent=2,
        )
    )
    render(jobs, gradients["rows"], args.output_dir)
    print(args.output_dir)


if __name__ == "__main__":
    main()
