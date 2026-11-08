# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Reaggregate archived teacher trajectories in the chain-rule direction.

Semantic ledger: each line is one Qwen teacher construction arm. Each point
uses 128 held-out questions/domain and the same frozen teacher paths at all
student checkpoints. The left panel is conditional student||teacher token
means on teacher paths; the right is teacher||student conditional sums, the
chain-rule-aligned truncated sequence estimator. Different arms have different
teacher paths. Neither panel uses current-student paths. One training seed.
Geometry ledger: two equal panels, identical linear residual axes and colors;
one global legend and one caption. No confidence band is inferred from seeds.
"""

import json
import math
from collections import defaultdict
from pathlib import Path

from examples.mirroropd.research.studies.fixed_transfer.compare_sft_teachers import (
    DOMAINS,
    QWEN,
    ROOT,
    digest,
    write_csv,
)

DOCS = ROOT / "examples/mirroropd/docs/experiments/teacher-transfer"
DATA = DOCS / "data/teacher-construction-20261108/sequence-direction-v1"
FIGURES = DOCS / "figures/teacher-construction-20261108/sequence-direction-v1"
ARMS = ("RAW", "CCS-rho0", "CCS-rho1")
METRICS = ("rkl_question_token_mean", "fkl_question_token_mean", "fkl_sequence_sum", "fkl_pooled_token_mean")


def aggregate(records):
    """Keep question mean, mean sum and length-pooled mean as distinct objects."""
    groups = defaultdict(list)
    for row in records:
        if row["tokens"] <= 0 or not all(math.isfinite(row[k]) for k in ("rkl", "fkl")):
            raise ValueError("Invalid conditional divergence or trajectory length")
        groups[row["step"], row["domain"]].append(row)
    result = []
    for (step, domain), values in sorted(groups.items()):
        if len({r["id"] for r in values}) != len(values):
            raise ValueError("Duplicate question")
        total = math.fsum(r["fkl"] * r["tokens"] for r in values)
        tokens = sum(r["tokens"] for r in values)
        result.append(
            dict(
                step=step,
                domain=domain,
                questions=len(values),
                mean_tokens=tokens / len(values),
                rkl_question_token_mean=math.fsum(r["rkl"] for r in values) / len(values),
                fkl_question_token_mean=math.fsum(r["fkl"] for r in values) / len(values),
                fkl_sequence_sum=total / len(values),
                fkl_pooled_token_mean=total / tokens,
            )
        )
    return result


def render(rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from examples.mirroropd.research.core.plotting import load_style, save_figure

    fv = load_style()
    fv.apply_rc("paper", fontsize=9)
    colors = fv.series_colors("paper", 3)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))
    for ax, metric, title in zip(
        axes,
        ("rkl_question_token_mean", "fkl_sequence_sum"),
        ("Conditional RKL on teacher paths", "Teacher-to-student KL sums"),
        strict=True,
    ):
        for arm, color, marker in zip(ARMS, colors, ("o", "s", "^"), strict=True):
            values = sorted((r for r in rows if r["arm"] == arm and r["domain"] == "macro4"), key=lambda r: r["step"])
            ax.plot(
                [r["step"] for r in values],
                [r[metric + "_residual_percent"] for r in values],
                label=arm,
                color=color,
                marker=marker,
                markersize=4,
            )
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("OPD optimizer updates")
        ax.set_ylabel("Four-domain residual (%)")
        ax.set_ylim(0, 105)
        ax.set_xlim(-5, 261)
        ax.set_xticks((0, 64, 128, 192, 256))
        ax.grid(alpha=0.18)
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=3, loc="upper center", frameon=False)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.78, bottom=0.23, wspace=0.30)
    save_figure(
        fig,
        FIGURES / "direction-residual",
        "Held-out own-teacher paths; H256; single seed. Exact conditional KLs; empirical path averages.",
        caption_size=8,
    )


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    receipt = json.loads((QWEN / "archive-receipt.json").read_text())
    rows, sources = [], []
    for arm in ARMS:
        name = f"inputs/opd/{arm}/heldout-teacher-bank.jsonl"
        path = QWEN / name
        if digest(path) != receipt["artifacts"][name]:
            raise ValueError("Archived input changed")
        sources.append(path)
        raw = [json.loads(line) for line in path.read_text().splitlines()]
        paths = {(r["domain"], r["id"]): r["tokens"] for r in raw if r["step"] == 0}
        if any(paths.get((r["domain"], r["id"])) != r["tokens"] for r in raw):
            raise ValueError("Question or trajectory length changed across checkpoints")
        values = aggregate(raw)
        if {(r["step"], r["domain"]) for r in values} != {(s, d) for s in (0, 16, 64, 128, 256) for d in DOMAINS}:
            raise ValueError("Incomplete checkpoint/domain grid")
        if any(r["questions"] != 128 for r in values):
            raise ValueError("Expected 128 questions/domain")
        for step in (0, 16, 64, 128, 256):
            subset = [r for r in values if r["step"] == step]
            values.append(
                dict(
                    step=step,
                    domain="macro4",
                    questions=512,
                    **{key: math.fsum(r[key] for r in subset) / 4 for key in (*METRICS, "mean_tokens")},
                )
            )
        initial = {r["domain"]: r for r in values if r["step"] == 0}
        for row in values:
            row["arm"] = arm
            for key in METRICS:
                row[key + "_residual_percent"] = 100 * row[key] / initial[row["domain"]][key]
        rows.extend(values)
    write_csv(DATA / "curves.csv", rows)
    write_csv(DATA / "endpoints.csv", [r for r in rows if r["step"] == 256])
    render(rows)
    (DATA / "provenance.json").write_text(
        json.dumps(
            dict(
                sources={str(p.relative_to(ROOT)): digest(p) for p in sources},
                generator={str(Path(__file__).relative_to(ROOT)): digest(Path(__file__))},
                interpretation="H256 teacher-path conditional FKL sums; chain-rule-aligned finite-sample estimate",
                execution="Original HF FP32 head/BF16 backbone; full head support; execution drift not corrected",
                macro="Equal-domain mean of domain metrics, then ratio to step zero; not a mixed-domain sequence KL",
                limitations="One path/question; truncation; different own-teacher measures; no current-student bank",
            ),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps([r for r in rows if r["step"] == 256 and r["domain"] == "macro4"], indent=2))


if __name__ == "__main__":
    main()
