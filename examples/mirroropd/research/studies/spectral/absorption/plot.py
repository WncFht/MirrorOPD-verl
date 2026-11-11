# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Plot heldout functional risks from immutable score files.

Ledger: each row is one predefined student/teacher condition, in manifest order.
Dots are equal-domain means of equal-question fixed-H128 KL on frozen states.
Panels separate state origins/metrics. Colors identify interventions throughout.
Vertical offsets separate interventions within a model, never change model order.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from examples.mirroropd.research.core.plotting import load_style


def mean(score, origin, metric):
    values = [r[metric] for r in score["rows"] if r["origin"] == origin and r["split"] == "heldout"]
    return float(np.mean(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baselines-only", action="store_true")
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    global fv
    fv = load_style()
    fv.apply_rc("paper")
    jobs = json.loads((args.run_root / "manifest.json").read_text())["jobs"]
    scores, selected, provenance = {}, [], []
    for job in jobs:
        folder = args.run_root / "jobs" / f"{job['job_id']:03d}" / "attempt-001"
        if not args.baselines_only and not (folder / "interventions-complete.json").exists():
            continue
        data = {}
        for p in sorted((folder / "scores").glob("*.json")):
            if p.name.endswith(".progress.json"):
                continue
            data[p.stem] = json.loads(p.read_text())
            provenance.append(dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        if "endpoint" not in data or "base" not in data:
            continue
        selected.append(job)
        scores[job["job_id"]] = data
    if not selected:
        raise ValueError("No complete baseline comparison")
    labels = [f"{j['job_id']:02d}  {j['family']} / {j['teacher']}" for j in selected]
    axespecs = [("initial-student", "rkl", "Initial-student prefix RKL"), ("teacher", "fkl", "Teacher prefix FKL")]
    configurations = [("absolute-baselines", [("base", "Initial", "vgray"), ("endpoint", "OPD", "vteal")])]
    if not args.baselines_only:
        configurations += [
            (
                "top-truncation",
                [
                    ("endpoint", "OPD", "vgray"),
                    ("top-0.01-restore-False", "Top 1%", "vteal"),
                    ("top-0.05-restore-False", "Top 5%", "vorange"),
                    ("top-0.01-restore-True", "Top 1% + norm", "vviolet"),
                    ("top-0.05-restore-True", "Top 5% + norm", "vgreen"),
                ],
            ),
            (
                "missing-off-controls",
                [
                    ("endpoint", "OPD", "vgray"),
                    ("missing-off-beta-1.0", "True off", "vteal"),
                    ("missing-off-gaussian", "Gaussian mean", "vorange"),
                    ("missing-off-random_frames", "Frame mean", "vviolet"),
                ],
            ),
            (
                "missing-null-controls",
                [
                    ("endpoint", "OPD", "vgray"),
                    ("missing-null-beta-1.0", "True null", "vteal"),
                    ("missing-null-gaussian", "Gaussian mean", "vorange"),
                    ("missing-null-random_frames", "Frame mean", "vviolet"),
                ],
            ),
        ]
    audits, plotted = {}, []
    for name, variants in configurations:
        for origin, metric, xlabel in axespecs:
            fig, ax = plt.subplots(figsize=(8, max(7, 0.43 * len(selected) + 2)))
            fig.subplots_adjust(left=0.4, right=0.97, bottom=0.33, top=0.97)
            for index, (condition, label, color) in enumerate(variants):
                values, lows, highs = [], [], []
                for job in selected:
                    data = scores[job["job_id"]]
                    names = [condition] if condition in data else [f"{condition}-{seed}" for seed in range(3)]
                    readings = [mean(data[k], origin, metric) for k in names]
                    if min(readings) <= 0:
                        raise ValueError("Nonpositive risk cannot be drawn on logarithmic axis")
                    values.append(float(np.mean(readings)))
                    lows.append(min(readings))
                    highs.append(max(readings))
                    plotted.append(
                        dict(
                            figure=name,
                            job_id=job["job_id"],
                            origin=origin,
                            metric=metric,
                            condition=condition,
                            mean=values[-1],
                            min=lows[-1],
                            max=highs[-1],
                        )
                    )
                offset = (index - (len(variants) - 1) / 2) * 0.13
                y = np.arange(len(selected)) + offset
                ax.scatter(values, y, s=20, color=fv.HUES[color], label=label)
                if any(high > low for high, low in zip(highs, lows, strict=True)):
                    ax.hlines(y, lows, highs, color=fv.HUES[color], linewidth=1, label="_nolegend_")
            ax.set_xscale("log")
            ax.set_xlabel(xlabel + "\nAbsolute KL / H128")
            ax.set_ylim(len(selected) - 0.3, -0.7)
            ax.set_yticks(np.arange(len(selected)), labels)
            ax.grid(axis="x", color=fv.NEUTRALS["vmist"], alpha=0.5)
            ax.set_axisbelow(True)
            ax.set_ylabel("Student / teacher condition")
            fig.legend(
                *ax.get_legend_handles_labels(),
                loc="lower center",
                bbox_to_anchor=(0.6, 0.09),
                ncol=min(3, len(variants)),
                frameon=False,
                fontsize=11,
            )
            caption = (
                f"{len(selected)}/19 usable conditions; heldout Math/Code/IF, 12 questions each\n"
                "Fixed manifest order; control whiskers: three-seed range"
            )
            if args.baselines_only:
                caption = caption.replace("control whiskers: three-seed range", "preliminary baseline coverage")
            fig.text(0.5, 0.012, caption, ha="center", fontsize=11, color=fv.HUES["vgray"])
            fig.canvas.draw()
            problems = fv.audit(fig)
            stem = name + "-" + origin
            audits[stem] = [str(x) for x in problems]
            for extension in ["png", "pdf", "svg"]:
                fig.savefig(args.output_dir / f"{stem}.{extension}", dpi=180)
            plt.close(fig)
    (args.output_dir / "audit.json").write_text(json.dumps(audits, indent=2))
    (args.output_dir / "plot-data.json").write_text(json.dumps(plotted, indent=2))
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps(dict(conditions=len(selected), figures=len(audits), audit=audits)))


if __name__ == "__main__":
    main()
