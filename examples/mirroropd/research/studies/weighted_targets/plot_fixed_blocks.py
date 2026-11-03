# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Plot the nine completed initial-anchor blocks, never substitute KL for accuracy."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from examples.mirroropd.research.core.plotting import load_style

ARMS = ["reuse64x16", "fresh1024x1", "multi128x8"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    input_bytes = args.data.read_bytes()
    data = json.loads(input_bytes)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "data.json").write_bytes(input_bytes)
    rows = []
    for i, (k, tag) in enumerate([(0, "k0"), (0.5, "k05"), (1, "k1")]):
        for j, arm in enumerate(ARMS):
            b = data["blocks"][f"{arm}-{tag}-s42-block1"]
            assert b["protocol"]["updates"] == 16 and b["protocol"]["batch_responses"] == 64
            rows.append(
                {
                    "kappa": k,
                    "arm": arm,
                    "teacher_kl": b["heldout"]["teacher_kl"],
                    "anchor_kl": b["heldout"]["anchor_kl"],
                    "token_visits": b["protocol"]["token_visits"],
                }
            )
    with (out / "points.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fv = load_style()
    fv.apply_rc("paper", fontsize=11)
    colors = [fv.HUES["vteal"], fv.HUES["vorange"], fv.HUES["vviolet"]]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.8))
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.33, top=0.79, wspace=0.32)
    labels = ["64 questions x 1\n16 passes", "1024 questions x 1\n1 pass", "128 questions x 8\n1 pass"]
    for ax, key, ylabel in zip(
        axes, ["teacher_kl", "anchor_kl"], [r"Heldout $D_{KL}(p\Vert T)$", r"Heldout $D_{KL}(p\Vert q)$"], strict=True
    ):
        for i, k in enumerate([0, 0.5, 1]):
            subset = [r for r in rows if r["kappa"] == k]
            ax.scatter(
                [j + (i - 1) * 0.16 for j in range(3)],
                [r[key] for r in subset],
                color=colors[i],
                marker=["o", "s", "^"][i],
                s=48,
                label=rf"$\kappa={k:g}$",
            )
        ax.set(xticks=range(3), xticklabels=labels, xlim=(-0.5, 2.5), ylabel=ylabel, xlabel="Coverage condition")
        ax.grid(axis="y", alpha=0.2)
    initial = data["blocks"]["reuse64x16-k0-s42-block1"]["complete"]["heldout_teacher_kl_before"]
    axes[0].axhline(initial, color=fv.NEUTRALS["vink"], ls="--", lw=1, label="Initial policy")
    axes[0].set_ylim(0.375, 0.414)
    axes[1].axhline(0, color=fv.NEUTRALS["vink"], ls="--", lw=1)
    axes[1].set_ylim(-0.00005, 0.0017)
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    fig.legend(
        *axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.11), ncol=4, frameon=False
    )
    fig.suptitle("Qwen3-1.7B-Base <- Qwen3-30B-A3B-Instruct-2507\nOne frozen-anchor MDPO block, eta=0.1", y=0.97)
    fig.text(
        0.5,
        0.018,
        "128 heldout Base trajectories; 16 updates x 64 responses; one bank/seed.\n"
        "Fixed-prefix KL measurements; independent task evaluations are still required.",
        ha="center",
        fontsize=10,
        color=fv.NEUTRALS["vink"],
    )
    fig.canvas.draw()
    issues = fv.audit(fig, ignore=["facet-ylim-mismatch"])
    (out / "audit.json").write_text(json.dumps(issues, indent=2) + "\n")
    assert not any(fv.problem_severity(x) == "error" for x in issues), issues
    for ext in ["png", "pdf"]:
        fig.savefig(out / f"fixed-block-risks.{ext}", dpi=180)
    (out / "provenance.json").write_text(
        json.dumps(
            {
                "input": "data.json",
                "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "scope": __doc__,
            },
            indent=2,
        )
        + "\n"
    )
    print(issues)


if __name__ == "__main__":
    main()
