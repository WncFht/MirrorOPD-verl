# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Plot paired AIME changes from Base for the common 64-response initial block.

Semantic ledger: each point is one completed kappa arm minus its matching Base
decoder; bars are paired question-bootstrap intervals, not training-seed errors.
Geometry ledger: both panels share percentage-point scales and kappa positions.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from examples.mirroropd.research.core.plotting import load_style


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data_bytes = args.data.read_bytes()
    data = json.loads(data_bytes)
    rows = []
    for protocol in ["local", "paper"]:
        for kappa, tag in [(0, "k0"), (0.5, "k05"), (1, "k1")]:
            right = f"reuse64x16-{tag}-s42-block1/{protocol}"
            condition = data["conditions"][right]
            assert condition["identity"]["n"] == 4 and condition["identity"]["input_variant"] == "local"
            item = next(c for c in data["contrasts"] if c["left"] == f"base/{protocol}" and c["right"] == right)
            value = item["right_minus_left"]["AIME macro"]["accuracy"]
            low, high = value["question_bootstrap95"]
            assert low <= value["value"] <= high
            rows.append(
                {
                    "protocol": protocol,
                    "kappa": kappa,
                    "delta_pp": 100 * value["value"],
                    "low_pp": 100 * low,
                    "high_pp": 100 * high,
                }
            )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "data.json").write_bytes(data_bytes)
    with (args.out / "points.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fv = load_style()
    fv.apply_rc("paper", fontsize=11)
    colors = [fv.HUES["vteal"], fv.HUES["vorange"], fv.HUES["vviolet"]]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.8), sharey=True)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.30, top=0.74, wspace=0.20)
    for ax, protocol, title in zip(
        axes,
        ["local", "paper"],
        ["Local decoder\nT=0.7, p=0.9, cap=31,744", "Paper decoder\nT=1, p=0.7, cap=16,384"],
        strict=True,
    ):
        subset = [row for row in rows if row["protocol"] == protocol]
        for index, row in enumerate(subset):
            point = row["delta_pp"]
            ax.errorbar(
                index,
                point,
                yerr=[[point - row["low_pp"]], [row["high_pp"] - point]],
                fmt=["o", "s", "^"][index],
                ms=7,
                capsize=5,
                lw=1.5,
                color=colors[index],
                label=rf"$\kappa={row['kappa']:g}$",
            )
        ax.axhline(0, color=fv.NEUTRALS["vink"], ls=":", lw=1)
        ax.set(
            title=title,
            xlabel=r"Target parameter $\kappa$",
            xticks=[0, 1, 2],
            xticklabels=["0", "0.5", "1"],
            xlim=(-0.5, 2.5),
            ylim=(-2.5, 3.5),
        )
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Native AIME avg@4 change (pp)")
    fig.suptitle("Qwen3-1.7B-Base <- Qwen3-30B-A3B-Instruct-2507\nInitial block: paired changes from Base", y=0.96)
    fig.legend(
        *axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.12), ncol=3, frameon=False
    )
    fig.text(
        0.5,
        0.035,
        "Same 64 responses, 16 updates; paired question 95% CIs; one training seed.",
        ha="center",
        fontsize=9,
        color=fv.NEUTRALS["vink"],
    )
    fig.canvas.draw()
    # Error-bar caps and stems are separate artists, but each panel has only three semantic series.
    problems = fv.audit(fig, ignore=["palette-overflow"])
    (args.out / "task-controls-audit.json").write_text(json.dumps(problems, indent=2) + "\n")
    assert not any(fv.problem_severity(problem) == "error" for problem in problems), problems
    for extension in ["png", "pdf"]:
        fig.savefig(args.out / f"initial-task-controls.{extension}", dpi=200)
    (args.out / "provenance.json").write_text(
        json.dumps(
            {
                "source_sha256": hashlib.sha256(data_bytes).hexdigest(),
                "plot_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "audit_exception": (
                    "palette-overflow counts error-bar caps/stems separately; three kappa series per panel"
                ),
                "scope": __doc__,
            },
            indent=2,
        )
        + "\n"
    )
    print(problems)


if __name__ == "__main__":
    main()
