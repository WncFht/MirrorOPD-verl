# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Plot P/A/C and per-record diagnostics from ``run_pac.py`` outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load(paths):
    rows = []
    for path in paths:
        data = json.loads(Path(path).read_text())
        label = Path(path).stem
        rows.append((label, data))
    if not rows:
        raise ValueError("no result files")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    results = load(args.inputs)
    args.out.mkdir(parents=True, exist_ok=True)
    labels = [x[0].replace("-", "\n") for x in results]
    summaries = [x[1]["summary"] for x in results]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    metrics = ["P", "A", "C", "rho", "Xi", "loss_mean"]
    for ax, metric in zip(axes.flat, metrics, strict=True):
        values = [s.get(metric, x[1].get(metric, np.nan)) for x, s in zip(results, summaries, strict=True)]
        ax.bar(np.arange(len(values)), values, color="#386cb0")
        ax.set_title(metric)
        ax.set_xticks(np.arange(len(values)), labels, rotation=25, ha="right")
        if metric in {"P", "A", "Xi"}:
            ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(args.out / "pac_summary.png", dpi=180)
    fig.savefig(args.out / "pac_summary.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for label, data in results:
        records = data["records"]
        pressure = [r["pressure"] for r in records]
        grad = [r["gradient_norm_sq"] for r in records]
        loss = [r["loss"] for r in records]
        axes[0].scatter(loss, pressure, label=label, s=28, alpha=0.8)
        axes[1].scatter(pressure, grad, label=label, s=28, alpha=0.8)
    axes[0].set(xlabel="record RKL loss", ylabel="record pressure P")
    axes[1].set(xlabel="record pressure P", ylabel="gradient norm squared")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.savefig(args.out / "record_geometry.png", dpi=180)
    fig.savefig(args.out / "record_geometry.pdf")
    plt.close(fig)

    audit = {
        "inputs": [str(p) for p in args.inputs],
        "results": [
            {"label": label, "summary": data["summary"], "records": len(data["records"])} for label, data in results
        ],
        "figures": ["pac_summary.png", "pac_summary.pdf", "record_geometry.png", "record_geometry.pdf"],
    }
    (args.out / "plot_audit.json").write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()
