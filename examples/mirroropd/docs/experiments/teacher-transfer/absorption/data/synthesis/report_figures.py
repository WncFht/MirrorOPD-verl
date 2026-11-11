# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Rebuild the curated absorption report from adjacent evidence, without run access.

Each row is a pair-condition. Risk comparisons use identical questions within a
bank; H128 cross-model and historical H1024 dynamics are distinct experiments.
ISO ratios change one full-weight component at a time and compare to alpha=1.
"""

import argparse
import csv
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def build(root):
    root = Path(root)
    sys.path.insert(0, os.environ["FIGURE_VIZ_MPL"])
    import fvstyle as fv
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fv.apply_rc("paper", fontsize=13)
    colors = fv.series_colors("paper", 4)
    output = root / "figures/synthesis"
    output.mkdir(parents=True, exist_ok=True)
    data = root / "data/synthesis"
    data.mkdir(parents=True, exist_ok=True)
    sources = [
        root / "figures/teacher-spectrum-clean/summary.csv",
        root / "data/synthesis/iso-inputs.json.gz",
        root / "data/teacher-spectrum-20261110/report-clean-v1/historical-curves.csv",
    ]
    rows = sorted(read_csv(sources[0]), key=lambda r: int(r["job_id"]))
    with gzip.open(sources[1], "rt") as stream:
        iso = json.load(stream)
    histories = read_csv(sources[2])
    names = {
        0: "R1 · DeepCoder [64]",
        1: "R1 · DeepScaleR",
        2: "R1 · JustRL",
        3: "R1 · MiniMath SFT",
        4: "R1 · BroRL",
        5: "R1 · Nemotron",
        6: "R1-7B · Light SFT [64]",
        7: "Qwen · GRPO",
        8: "Qwen · Samsung SFT (think)",
        9: "Qwen · e3",
        10: "Qwen · Samsung SFT (nonthink)",
        11: "Llama · CoReward-I",
        12: "Llama · GT",
        13: "Llama · Arithmetic SFT",
        19: "MiMo SFT → SFT+RL",
        20: "MiMo SFT → RL-Zero [branch]",
    }
    audits = []

    def save(fig, name, caption):
        fig.text(0.5, 0.018, caption, ha="center", fontsize=12, color=fv.NEUTRALS["baseline"])
        fig.canvas.draw()
        problems = [str(p) for p in fv.audit(fig)]
        audits.append(dict(figure=name, problems=problems))
        for ext in ("png", "pdf", "svg"):
            fig.savefig(output / f"{name}.{ext}", dpi=180)
        plt.close(fig)

    y = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(13, 8), sharey=True)
    fig.subplots_adjust(left=0.32, right=0.97, bottom=0.13, top=0.88, wspace=0.3)
    for index, (key, label) in enumerate(
        [("base", "Initial"), ("endpoint", "OPD"), ("selected-off-rkl", "OPD + missing off")]
    ):
        x = [float(r[key]) for r in rows]
        axes[0].plot(
            x,
            y + (index - 1) * 0.14,
            ["o", "s", "D"][index],
            color=colors[index],
            label=label,
            linestyle="none",
            markersize=4,
        )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Absolute RKL (H128, 36 heldout questions)")
    for i, (key, label) in enumerate([("endpoint", "OPD"), ("selected-off-rkl", "OPD + missing off")]):
        x = [100 * float(r[key]) / float(r["base"]) for r in rows]
        axes[1].plot(
            x, y + (i - 0.5) * 0.14, ["s", "D"][i], color=colors[i + 1], label=label, linestyle="none", markersize=4
        )
    axes[1].set_xlabel("Remaining initial gap (%)")
    axes[1].set_xlim(0, 55)
    axes[0].set_yticks(y, [names[int(r["job_id"])] for r in rows])
    axes[0].invert_yaxis()
    for ax in axes:
        ax.grid(axis="x", alpha=0.18)
        ax.set_ylabel("Pair condition")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False)
    save(
        fig,
        "model-absorption",
        "Initial-student prefixes; off amplitude selected on calibration only; [64] marks short runs.",
    )

    selected = [
        r
        for r in iso["functional"]
        if r["condition"]["role"] == "student"
        and r["origin"] == "initial-student"
        and r["split"] == "heldout"
        and r["domain"] == "math"
        and (
            (r["condition"]["kind"] == "restore" and r["condition"]["alpha"] in [0.0, 1.0])
            or r["condition"]["kind"] == "reverse"
        )
    ]
    lookup = {(int(r["task"].split("-")[-1]), r["condition"]["kind"], r["condition"]["alpha"]): r for r in selected}
    ids = sorted({key[0] for key in lookup})
    exported = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 8), sharey=True)
    fig.subplots_adjust(left=0.32, right=0.97, bottom=0.13, top=0.86, wspace=0.3)
    for i, job in enumerate(ids):
        base = lookup[job, "restore", 1.0]["rkl"]["absolute_mean"]
        restore = lookup[job, "restore", 0.0]["rkl"]["absolute_mean"]
        reverse = lookup[job, "reverse", 1.0]["rkl"]["absolute_mean"]
        exported.append(
            dict(
                job_id=job,
                name=names[job],
                endpoint=base,
                base_spectrum=restore,
                base_frames=reverse,
                spectrum_ratio=restore / base,
                frames_ratio=reverse / base,
            )
        )
    for ax, key in zip(axes, ["spectrum_ratio", "frames_ratio"], strict=True):
        ax.axvline(1, color=fv.NEUTRALS["baseline"], linestyle="--", linewidth=1)
        ax.plot(
            [r[key] for r in exported],
            np.arange(len(ids)),
            "o",
            color=colors[0 if key == "spectrum_ratio" else 2],
            markersize=5,
            linestyle="none",
            label="Restore spectrum" if key == "spectrum_ratio" else "Restore frames",
        )
        ax.grid(axis="x", alpha=0.18)
        ax.set_xlabel("Edited RKL / endpoint RKL")
        ax.set_ylabel("Pair condition")

    fig.legend(handles=[axes[0].lines[-1], axes[1].lines[-1]], loc="upper center", ncol=2, frameon=False, fontsize=12)
    axes[0].set_title("Keep learned frames", pad=15)
    axes[1].set_title("Keep learned spectrum", pad=15)
    axes[0].set_xlim(0.88, 1.12)
    axes[1].set_xscale("log")
    axes[1].set_xlim(1, 150)
    axes[0].set_yticks(np.arange(len(ids)), [names[i] for i in ids])
    axes[0].invert_yaxis()
    save(
        fig,
        "spectrum-versus-frames",
        "Same Math12 questions, initial-student prefixes, H128. Panel scales differ; dashed line = no change.",
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.14, top=0.80, hspace=0.45, wspace=0.29)
    sets = [
        ["GRPO-KL thinking", "Samsung SFT thinking", "Thinking-Space SFT thinking", "JustRL fixed-S0"],
        ["Samsung SFT nonthinking", "Thinking-Space SFT nonthinking"],
    ]
    short = {
        "GRPO-KL thinking": "GRPO",
        "Samsung SFT thinking": "Samsung SFT",
        "Thinking-Space SFT thinking": "Thinking-Space SFT",
        "JustRL fixed-S0": "JustRL",
        "Samsung SFT nonthinking": "Samsung SFT",
        "Thinking-Space SFT nonthinking": "Thinking-Space SFT",
    }
    used_colors = {"GRPO": colors[0], "Samsung SFT": colors[1], "Thinking-Space SFT": colors[2], "JustRL": colors[3]}
    for col, branches in enumerate(sets):
        for branch in branches:
            records = sorted(
                [r for r in histories if r["branch"] == branch and r["panel"] == "held_all"],
                key=lambda r: int(r["step"]),
            )
            for row_index, key in enumerate(["rkl", "remaining"]):
                axes[row_index, col].plot(
                    [int(r["step"]) for r in records],
                    [float(r[key]) for r in records],
                    color=used_colors[short[branch]],
                    label=short[branch],
                    linewidth=1.6,
                )
        axes[0, col].set_title("Thinking / fixed-S0 · 0–320" if col == 0 else "Non-thinking · 0–1280", pad=12)
        for ax in axes[:, col]:
            ax.set_yscale("log")
            ax.set_xlabel("Optimizer update")
            ax.grid(alpha=0.18)
        axes[0, col].set_ylabel("Absolute heldout RKL (H1024)")
        axes[1, col].set_ylabel("Remaining initial gap")
    for ax in axes[0, :]:
        ax.set_ylim(0.0007, 1.0)
    for ax in axes[1, :]:
        ax.set_ylim(0.04, 1.2)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False, fontsize=12)
    save(
        fig,
        "learning-dynamics",
        "Fixed historical bank: Math64 + Code32 + IF32; real checkpoints only; no smoothing or cross-run splicing.",
    )

    def write_table(path, values):
        with path.open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)

    write_table(data / "model-absorption.csv", rows)
    write_table(data / "spectrum-versus-frames.csv", exported)
    write_table(data / "learning-dynamics.csv", histories)
    (output / "figure-audit.json").write_text(json.dumps(audits, indent=2))
    provenance = dict(
        inputs=[
            dict(path=str(p.relative_to(root)), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources
        ],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        unit="question for risk; pair-condition for plotted rows; no training-seed confidence claim",
    )
    (data / "provenance.json").write_text(json.dumps(provenance, indent=2))
    shutil_source = data / "report_figures.py"
    shutil_source.write_bytes(Path(__file__).read_bytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, required=True)
    build(parser.parse_args().docs)
