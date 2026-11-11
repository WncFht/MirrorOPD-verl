# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Compare independent token-position seeds without pooling repeated questions."""

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .report_stages import DEFAULT_STYLE, frame
from .results import write_csv
from .runtime import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--style", type=Path, default=DEFAULT_STYLE)
    args = parser.parse_args()
    rows, sources = [], {}
    for index, root in enumerate(args.roots):
        path = root / "analysis/transfer.csv"
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        for row in csv.DictReader(path.open()):
            if row["branch"] == row["target"]:
                rows.append(dict(panel=root.name, sampling_index=index, **row))
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "sampling-transfer.csv", rows)
    sys.path.insert(0, str(args.style))
    import fvstyle as fv

    fv.apply_rc("paper")
    colors = dict(justrl=fv.HUES["vteal"], polaris=fv.HUES["vorange"])
    audits = {}
    for role in ["all-train", "future-train"]:
        fig, axes = plt.subplots(2, 2, figsize=(8, 7.5))
        fig.subplots_adjust(left=0.13, right=0.98, bottom=0.23, top=0.94, hspace=0.6, wspace=0.5)
        for i, mode in enumerate(["live", "frozen_B0"]):
            for j, (key, label) in enumerate(
                [
                    ("cross_inner", "Cross gradient inner product"),
                    ("cross_cosine", "Train / heldout mean-gradient cosine"),
                ]
            ):
                ax = axes[i, j]
                for index, root in enumerate(args.roots):
                    for branch in colors:
                        selected = sorted(
                            [
                                r
                                for r in rows
                                if r["panel"] == root.name
                                and r["branch"] == branch
                                and r["mode"] == mode
                                and r["source_role"] == role
                            ],
                            key=lambda r: int(r["stage"]),
                        )
                        if not selected:
                            continue
                        stages = [int(r["stage"]) for r in selected]
                        values = [float(r[key]) for r in selected]
                        style = "-" if index == 0 else "--"
                        marker = "o" if index == 0 else "s"
                        ax.plot(
                            stages,
                            values,
                            style,
                            marker=marker,
                            markersize=4,
                            color=colors[branch],
                            label=f"{branch} sample {index + 1}",
                        )
                        ax.fill_between(
                            stages,
                            [float(r[key + "_low"]) for r in selected],
                            [float(r[key + "_high"]) for r in selected],
                            color=colors[branch],
                            alpha=0.1,
                        )
                ax.set_title(mode, fontsize=11)
                frame(ax, "Optimizer updates", label)
        for column in range(2):
            limits = [axes[row, column].get_ylim() for row in range(2)]
            for row in range(2):
                axes[row, column].set_ylim(min(v[0] for v in limits), max(v[1] for v in limits))
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=11, frameon=False)
        fig.canvas.draw()
        audits[role] = [str(issue) for issue in fv.audit(fig)]
        for ext in ["png", "pdf"]:
            fig.savefig(args.out / f"sampling_{role}.{ext}", dpi=200)
        plt.close(fig)
    write_json(
        args.out / "sampling-audit.json",
        dict(
            sources=sources,
            figure_audits=audits,
            note="Same questions, independent positions; no pooling or doubled question count.",
        ),
    )


if __name__ == "__main__":
    main()
