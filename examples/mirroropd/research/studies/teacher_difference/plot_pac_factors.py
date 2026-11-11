# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Compare fixed-bank PAC factors using completed results only.

Semantic ledger: color = teacher; column = fixed train/heldout math population;
rows = loss, pressure per loss, trajectory accessibility/coherence, self rate.
Geometry ledger: x = saved optimizer updates; y = stated scalar, logarithmic.
Each trajectory contributes its fixed-H averaged gradient. The train panel
contains both seen and future batches. Neither row measures actual Adam rate.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .report_stages import DEFAULT_STYLE, frame
from .results import write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--style", type=Path, default=DEFAULT_STYLE)
    args = parser.parse_args()
    sys.path.insert(0, str(args.style))
    import fvstyle as fv

    fv.apply_rc("paper")
    root = args.manifest.parent
    manifest = json.loads(args.manifest.read_text())
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    rows, sources = [], {}
    for task in manifest["tasks"]:
        if task["model"] != "r1" or "-target-" in task["id"]:
            continue
        path = root / "results" / task["id"] / "result.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        if not data.get("finished") or data["signature"]["task"] != task:
            raise ValueError(f"Incomplete or mismatched result: {path}")
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        for split in ["train/math", "heldout/math"]:
            s = data["summaries"][split]
            loss = s["loss_mean"]
            rate = s["P"] / loss * s["A"] * s["C"]
            rows.append(
                dict(
                    task=task["id"],
                    target=task["target"],
                    stage=task["stage"],
                    split=split,
                    loss=loss,
                    P=s["P"],
                    P_over_L=s["P"] / loss,
                    A=s["A"],
                    C=s["C"],
                    Xi=s["Xi"],
                    PXi=s["P"] * s["Xi"],
                    PXi_over_L=rate,
                )
            )
    write_csv(out / "r1_pac_factors.csv", rows)
    metrics = [
        ("loss", "RKL / fixed H"),
        ("P_over_L", "P / L"),
        ("A", "Trajectory A"),
        ("C", "Trajectory C"),
        ("PXi_over_L", "Self signal: (P/L) A C"),
    ]
    fig, axes = plt.subplots(5, 2, figsize=(7, 10), sharey="row")
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.11, top=0.95, hspace=0.7, wspace=0.5)
    colors = dict(justrl=fv.HUES["vteal"], polaris=fv.HUES["vorange"])
    for row, split in enumerate(["train/math", "heldout/math"]):
        for ax, (key, label) in zip(axes[:, row], metrics, strict=True):
            for target in colors:
                selected = sorted(
                    [r for r in rows if r["split"] == split and r["target"] == target], key=lambda r: r["stage"]
                )
                ax.plot(
                    [r["stage"] for r in selected],
                    [r[key] for r in selected],
                    "o-",
                    color=colors[target],
                    label=target,
                    markersize=3,
                )
            ax.set_yscale("log")
            ax.set_xticks([0, 160, 320])
            frame(ax, "Optimizer updates", label)
        axes[0, row].set_title(f"{split}: {'160' if row == 0 else '96'} fixed trajectories", loc="left")
    axes[0, 0].legend(fontsize=9)
    fig.text(
        0.5,
        0.035,
        "Same S0 prefixes, H1024; train includes seen/future batches.\n"
        "Self gradient-flow signal; actual transfer and Adam rates require separate measurements.",
        ha="center",
        color=fv.NEUTRALS["baseline"],
        fontsize=9,
    )
    fig.canvas.draw()
    audit = [str(problem) for problem in fv.audit(fig)]
    for suffix in ["png", "pdf"]:
        fig.savefig(out / f"r1_pac_factors.{suffix}", dpi=220)
    plt.close(fig)
    (out / "r1_pac_factors.audit.json").write_text(json.dumps(dict(sources=sources, audit=audit), indent=2))


if __name__ == "__main__":
    main()
