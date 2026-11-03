# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Render the completed TOP-D online coverage/kappa endpoints, without seed pooling.

Semantic ledger: each point is one training seed and one completed arm at outer8.
AIME is the equal mean of AIME24/25/26; macro5 also includes AMC23 and MATH64.
Each task question has four responses. Seeds share evaluation prompts and seeds;
therefore points are descriptive replications, not independent question samples.
Geometry ledger: horizontal slots identify arms, offsets identify training seeds.
Each panel uses a percent scale starting at zero; no interval pools training seeds.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from examples.mirroropd.research.core.plotting import load_style

ARMS = ("reuse-k0", "reuse-k05", "reuse-k1", "fresh-k1")
DATASETS = ("AIME24", "AIME25", "AIME26", "AMC23", "MATH-500")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw = args.data.read_bytes()
    data = json.loads(raw)
    rows = []
    for seed in (42, 43, 44):
        for arm in ARMS:
            prefix = "online-" if seed == 42 else f"online-seed{seed}-"
            label = f"{prefix}{arm}-outer08/paper"
            condition = data["conditions"][label]
            assert condition["identity"]["n"] == 4
            summary = condition["summary"]["datasets"]
            assert all(
                summary[d]["questions"] == (64 if d == "MATH-500" else 40 if d == "AMC23" else 30) for d in DATASETS
            )
            row = dict(seed=seed, arm=arm, completion_sha256=condition["completion_sha256"])
            for name, field in (("native", "observed_mean"), ("official", "official_verifier_observed_mean")):
                row[f"{name}_aime_percent"] = 100 * float(np.mean([summary[d][field] for d in DATASETS[:3]]))
                row[f"{name}_macro5_percent"] = 100 * float(np.mean([summary[d][field] for d in DATASETS]))
            row["aime_mean_length"] = float(np.mean([summary[d]["mean_response_length"] for d in DATASETS[:3]]))
            row["aime_cap_percent"] = 100 * float(np.mean([summary[d]["cap_rate"] for d in DATASETS[:3]]))
            row["verifier_disagreements"] = len(condition["disagreements"])
            rows.append(row)
    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out / "endpoints.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fv = load_style()
    fv.apply_rc("paper", fontsize=11)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    colors = [fv.HUES[c] for c in ("vteal", "vorange", "vviolet")]
    for ax, metric, title in zip(axes, ("aime", "macro5"), ("AIME24/25/26", "Five-set macro (MATH64)"), strict=True):
        for i, seed in enumerate((42, 43, 44)):
            values = [r[f"native_{metric}_percent"] for r in rows if r["seed"] == seed]
            ax.plot(
                np.arange(4) + (i - 1) * 0.14,
                values,
                ls="none",
                marker=("o", "s", "^")[i],
                ms=7,
                color=colors[i],
                label=f"Training seed {seed}",
            )
        ax.set(
            xticks=range(4),
            xticklabels=["Reuse\nk=0", "Reuse\nk=0.5", "Reuse\nk=1", "Fresh\nk=1"],
            xlim=(-0.45, 3.45),
            ylim=(0, 10 if metric == "aime" else 30),
            xlabel="Online training arm",
            ylabel="Native avg@4 accuracy (%)",
            title=title,
        )
        ax.grid(axis="y", alpha=0.2)
    fig.legend(
        *axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.99)
    )
    fig.subplots_adjust(top=0.80, bottom=0.28, left=0.08, right=0.98, wspace=0.27)
    fig.text(
        0.5,
        0.045,
        "Outer8: 128 updates and 8,192 response visits per arm; three training seeds, no pooled CI.",
        ha="center",
        fontsize=9,
        color=fv.NEUTRALS["baseline"],
    )
    fig.canvas.draw()
    # Separate metrics intentionally use different scales, each starting at zero.
    problems = fv.audit(fig, print_slot=fig.get_figwidth(), ignore=["facet-ylim-mismatch"])
    (args.out / "figure-audit.json").write_text(json.dumps(fv.format_audit_lines(problems), indent=2) + "\n")
    assert not any(fv.problem_severity(p) == "error" for p in problems), problems
    for ext in ("png", "pdf", "svg"):
        fig.savefig(args.out / f"online-endpoints.{ext}", dpi=180, facecolor=fv.NEUTRALS["paper"])
    (args.out / "provenance.json").write_text(
        json.dumps(
            dict(
                audit_exception="facet-ylim-mismatch: different metrics; AIME 0..10%, macro5 0..30%",
                source_sha256=hashlib.sha256(raw).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope=__doc__,
            ),
            indent=2,
        )
        + "\n"
    )
    print(fv.format_audit_lines(problems))


if __name__ == "__main__":
    main()
