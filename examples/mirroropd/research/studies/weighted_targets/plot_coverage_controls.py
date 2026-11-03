# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Plot the complete initial 3x3 coverage/kappa task screen.

Each color and marker identifies kappa in both panels. Left: observed AIME
avg@4. Right: same-question differences from reuse64, with question bootstrap
intervals. All nine conditions must be complete; no imputed or partial scores.
"""

import argparse
import csv
import hashlib
import json
import math
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
    raw = args.data.read_bytes()
    data = json.loads(raw)
    arms = ["reuse64x16", "fresh1024x1", "multi128x8"]
    rows = []
    for kappa, tag in [(0, "k0"), (0.5, "k05"), (1, "k1")]:
        reference = f"reuse64x16-{tag}-s42-block1/paper"
        for arm in arms:
            label = f"{arm}-{tag}-s42-block1/paper"
            condition = data["conditions"][label]
            assert condition["identity"]["n"] == 4 and condition["identity"]["input_variant"] == "local"
            mean = (
                sum(condition["summary"]["datasets"][ds]["observed_mean"] for ds in ["AIME24", "AIME25", "AIME26"]) / 3
            )
            delta, low, high = 0, 0, 0
            if arm != arms[0]:
                item = next(c for c in data["contrasts"] if c["left"] == reference and c["right"] == label)
                estimate = item["right_minus_left"]["AIME macro"]["accuracy"]
                delta = 100 * estimate["value"]
                low, high = [100 * value for value in estimate["question_bootstrap95"]]
                assert low <= delta <= high
            rows.append(
                {
                    "kappa": kappa,
                    "arm": arm,
                    "accuracy_percent": 100 * mean,
                    "delta_pp": delta,
                    "low_pp": low,
                    "high_pp": high,
                }
            )
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "data.json").write_bytes(raw)
    with (args.out / "points.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fv = load_style()
    fv.apply_rc("paper", fontsize=11)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.1))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.30, top=0.77, wspace=0.32)
    colors = [fv.HUES["vteal"], fv.HUES["vorange"], fv.HUES["vviolet"]]
    for index, kappa in enumerate([0, 0.5, 1]):
        subset = [r for r in rows if r["kappa"] == kappa]
        marker = ["o", "s", "^"][index]
        axes[0].plot(
            range(3),
            [r["accuracy_percent"] for r in subset],
            color=colors[index],
            marker=marker,
            ms=6,
            label=rf"$\kappa={kappa:g}$",
        )
        for x, row in enumerate(subset[1:]):
            value = row["delta_pp"]
            axes[1].errorbar(
                x + (index - 1) * 0.18,
                value,
                yerr=[[value - row["low_pp"]], [row["high_pp"] - value]],
                fmt=marker,
                color=colors[index],
                ms=6,
                capsize=4,
                label=rf"$\kappa={kappa:g}$" if x == 0 else None,
            )
    base = data["conditions"]["base/paper"]["summary"]["datasets"]
    base_mean = 100 * sum(base[ds]["observed_mean"] for ds in ["AIME24", "AIME25", "AIME26"]) / 3
    axes[0].axhline(base_mean, color=fv.NEUTRALS["vink"], ls=":", lw=1, label="Base (left)")
    axes[1].axhline(0, color=fv.NEUTRALS["vink"], ls=":", lw=1)
    axes[0].set(
        title="Observed AIME accuracy",
        ylabel="Native avg@4 (%)",
        xlabel="Training coverage",
        xticks=[0, 1, 2],
        xticklabels=["Reuse 64", "Fresh 1024", "128 x 8"],
        xlim=(-0.3, 2.3),
        ylim=(0, max(5, math.ceil(max(r["accuracy_percent"] for r in rows)) + 1)),
    )
    axes[1].set(
        title="Paired change from reuse64",
        ylabel="Accuracy change (pp)",
        xlabel="Training coverage",
        xticks=[0, 1],
        xticklabels=["Fresh 1024", "128 x 8"],
        xlim=(-0.5, 1.5),
        ylim=(
            min(-2, math.floor(min(r["low_pp"] for r in rows)) - 0.5),
            max(4, math.ceil(max(r["high_pp"] for r in rows)) + 0.5),
        ),
    )
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    fig.legend(
        *axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.13), ncol=4, frameon=False
    )
    fig.suptitle("Qwen3-1.7B-Base <- Qwen3-30B-A3B-Instruct-2507\nInitial coverage screen: 16 updates, batch64", y=0.96)
    fig.text(
        0.5,
        0.035,
        "Native input; T=1, p=0.7, cap=16,384. One training seed.\n"
        "95% paired question CIs; no training-seed or multiplicity adjustment.",
        ha="center",
        fontsize=10,
        color=fv.NEUTRALS["vink"],
    )
    fig.canvas.draw()
    # Three semantic kappa series; errorbar caps/stems are extra artists.
    issues = fv.audit(fig, scoped_ignore=[("palette-overflow", 1)])
    (args.out / "coverage-audit.json").write_text(json.dumps(issues, indent=2) + "\n")
    assert not any(fv.problem_severity(problem) == "error" for problem in issues), issues
    for ext in ["png", "pdf"]:
        fig.savefig(args.out / f"coverage-controls.{ext}", dpi=200)
    provenance = {
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "plot_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "audit_exception": "Right-panel errorbar caps/stems overcount the three kappa series.",
        "scope": __doc__,
    }
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(issues)


if __name__ == "__main__":
    main()
