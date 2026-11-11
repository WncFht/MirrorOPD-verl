# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Export and plot the W&B snapshots for E85 and E86.

Semantic ledger: color and marker identify the aggregation variant. Validation
curves use complete avg@16 nodes. Optimization curves use the logged token
mean objective and teacher KL; raw actor loss remains run-specific because
E85 uses ``token-sum`` while E86 uses ``seq-mean-token-sum``. Bold points are
means of the 16 updates in one outer. No interpolation or extrapolation is
performed. Geometry ledger: x is optimizer update, with 16 updates per outer.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone

from examples.mirroropd.research.studies.weighted_targets.plot_e81_validation import (
    ROOT,
    source,
    style,
)
from examples.mirroropd.research.studies.weighted_targets.plot_validation_comparison import save

SNAPSHOT = ROOT / "runs/wandb-e85-e86-20261109"
OUT = ROOT / "examples/mirroropd/docs/experiments/learning-signal/figures/e85-e86-comparison-20261109"
RUNS = {"E85": "sjdbuh0p", "E86": "sh3mrf3r"}
BENCHMARKS = ("math500", "amc23", "aime24", "aime25", "aime26")
LABELS = {
    "math500": "MATH-500",
    "amc23": "AMC23",
    "aime24": "AIME24",
    "aime25": "AIME25",
    "aime26": "AIME26",
}


def read_history(run):
    return [json.loads(line) for line in (SNAPSHOT / RUNS[run] / "history.jsonl").open()]


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def refresh():
    validations, updates, outers, audits = [], [], [], {}
    for run in RUNS:
        rows = read_history(run)
        eval_rows = [r for r in rows if all(f"eval/{bench}/reward/mean" in r for bench in BENCHMARKS)]
        for r in eval_rows:
            outer = int(round((r["_step"] + 1) / 17))
            values = {bench: float(r[f"eval/{bench}/reward/mean"]) for bench in BENCHMARKS}
            validations.append(
                {
                    "run": run,
                    "outer": outer,
                    "optimizer_update": outer * 16,
                    **values,
                    "macro5": statistics.mean(values.values()),
                    "aime_mean": statistics.mean(values[k] for k in ("aime24", "aime25", "aime26")),
                }
            )
        train = [r for r in rows if "step/update" in r and "optim/actor/update_applied" in r]
        for r in train:
            updates.append(
                {
                    "run": run,
                    "outer": int(r["step/outer"]),
                    "optimizer_update": int(r["step/update"]),
                    "actor_loss": float(r["optim/actor/loss"]),
                    "objective": float(r["distill/mdpo/objective"]),
                    "teacher_kl": float(r["distill/teacher_kl"]),
                    "anchor_kl": float(r.get("distill/anchor_kl", float("nan"))),
                    "target_kl": float(r.get("distill/target_kl", float("nan"))),
                    "grad_norm": float(r.get("optim/actor/grad_norm_preclip", float("nan"))),
                    "update_applied": int(r.get("optim/actor/update_applied", 0)),
                    "loss_nonfinite": int(r.get("optim/actor/loss_nonfinite", 0)),
                }
            )
        grouped = defaultdict(list)
        for r in train:
            grouped[int(r["step/outer"])].append(r)
        for outer, group in sorted(grouped.items()):
            if len(group) != 16:
                continue
            summary = next(
                (r for r in reversed(rows) if r.get("step/outer") == outer and "rollout/response_length/mean" in r),
                None,
            )
            if summary is None:
                continue

            def avg(key, group=group):
                return statistics.mean(float(r[key]) for r in group if key in r)

            outers.append(
                {
                    "run": run,
                    "outer": outer,
                    "optimizer_update": outer * 16,
                    "actor_loss": avg("optim/actor/loss"),
                    "objective": avg("distill/mdpo/objective"),
                    "teacher_kl": avg("distill/teacher_kl"),
                    "grad_norm": avg("optim/actor/grad_norm_preclip"),
                    "anchor_kl": avg("distill/anchor_kl"),
                    "target_kl": avg("distill/target_kl"),
                    "norm_above_one": statistics.mean(
                        float(r.get("optim/actor/grad_norm_preclip", 0)) > 1 for r in group
                    ),
                    "train_length": float(summary["rollout/response_length/mean"]),
                    "length_fraction": float(summary.get("rollout/finish_reason/length_fraction", 0)),
                    "repetition_fraction": float(summary.get("rollout/finish_reason/repetition_fraction", 0)),
                    "stop_fraction": float(summary.get("rollout/finish_reason/stop_fraction", 0)),
                }
            )
        audits[run] = {
            "wandb_run": RUNS[run],
            "state": "failed",
            "history_rows": len(rows),
            "last_history_step": max(int(r.get("_step", -1)) for r in rows),
            "last_training_outer": max(int(r["step/outer"]) for r in train),
            "last_optimizer_update": max(int(r["step/update"]) for r in train),
            "last_complete_validation_outer": max(int(r["outer"]) for r in validations if r["run"] == run),
            "update_applied": sum(int(r.get("optim/actor/update_applied", 0)) for r in train),
            "loss_nonfinite": sum(int(r.get("optim/actor/loss_nonfinite", 0)) for r in train),
        }
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "validation.csv", validations)
    write_csv(OUT / "updates.csv", updates)
    write_csv(OUT / "outers.csv", outers)
    sources = []
    for run, run_id in RUNS.items():
        for name in ("history.jsonl", "config.json", "summary.json", "source.json"):
            path = SNAPSHOT / run_id / name
            sources.append(source(path, path.read_bytes()))
    (OUT / "source-audit.json").write_text(json.dumps(audits, indent=2) + "\n")
    (OUT / "provenance.json").write_text(
        json.dumps(
            {
                "snapshot_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "sources": sources,
                "settings": {
                    "E85": "future=none, token-sum",
                    "E86": "future=none, seq-mean-token-sum",
                },
                "validation": "Five benchmarks, avg@16, validation every 5 outer; macro5 weights benchmarks equally.",
                "limits": (
                    "Both W&B runs are marked failed; complete validation curves stop at outer90/80. "
                    "One run per condition; changing on-policy prefixes."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


def load(name):
    with (OUT / f"{name}.csv").open() as stream:
        return list(csv.DictReader(stream))


def plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator, NullLocator

    fv = style()
    theme = fv.apply_rc("paper")
    palette = fv.series_colors("paper", 9)
    colors, markers = {"E85": palette[0], "E86": palette[1]}, {"E85": "D", "E86": "P"}
    valid, updates, outers = load("validation"), load("updates"), load("outers")
    for row in valid + updates + outers:
        for key, value in row.items():
            if key not in ("run",):
                try:
                    row[key] = float(value)
                except (ValueError, TypeError):
                    pass
    handles = [
        Line2D(
            [],
            [],
            color=colors[r],
            marker=markers[r],
            label=f"{r} | {'token-sum' if r == 'E85' else 'seq-mean-token-sum'}",
        )
        for r in RUNS
    ]
    max_update = max(r["optimizer_update"] for r in outers)

    def finish(fig, stem, caption):
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.055), ncol=2, frameon=False)
        fig.text(0.5, 0.015, caption, ha="center", color=fv.NEUTRALS["baseline"], fontsize="small")
        fig.tight_layout(rect=(0, 0.13, 1, 1))
        save(fig, OUT, stem, fv, independent=True)

    def frame(ax, title, ylabel):
        ax.set(title=title, xlabel="Optimizer updates", ylabel=ylabel, xlim=(0, max_update * 1.04))
        fv.apply_frame(ax, theme.frame)
        fv.apply_grid(ax, theme.grid)

    def auto_ylim(ax, floor=0.0, pad=0.08):
        values = [value for line in ax.lines for value in line.get_ydata() if math.isfinite(value)]
        if not values:
            return
        lower, upper = min(values), max(values)
        span = max(upper - lower, abs(upper) * 0.05, 1.0)
        ax.set_ylim(max(floor, lower - pad * span), upper + pad * span)

    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.8))
    for ax, key, title in zip(
        axes.flat,
        ("macro5", "math500", "amc23", "aime24", "aime25", "aime26"),
        ("Macro-5", *[LABELS[x] for x in BENCHMARKS]),
        strict=True,
    ):
        for run in RUNS:
            points = [r for r in valid if r["run"] == run]
            ax.plot(
                [r["optimizer_update"] for r in points],
                [100 * r[key] for r in points],
                color=colors[run],
                marker=markers[run],
            )
        frame(ax, title, "Validation accuracy (%)")
        auto_ylim(ax)
    finish(
        fig,
        "validation-accuracy",
        "Complete avg@16 nodes; Macro-5 gives equal weight to five benchmarks; no interpolation",
    )
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.8))
    for ax, key, title in zip(
        axes.flat,
        ("objective", "teacher_kl", "actor_loss", "grad_norm", "anchor_kl", "target_kl"),
        (
            "MDPO objective",
            "Teacher KL",
            "Raw actor loss (run scale)",
            "Preclip gradient norm",
            "Anchor KL",
            "Target KL",
        ),
        strict=True,
    ):
        for run in RUNS:
            points = [r for r in outers if r["run"] == run]
            ax.plot(
                [r["optimizer_update"] for r in points],
                [r[key] for r in points],
                color=colors[run],
                marker=markers[run],
            )
        frame(ax, title, "Logged value")
        if key in ("actor_loss", "grad_norm"):
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
            ax.yaxis.set_minor_locator(NullLocator())
        ax.margins(y=0.1)
    finish(
        fig,
        "optimization",
        "Objective and KL are token means; raw actor loss uses each run's aggregation and is not directly rankable",
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.4))
    for ax, key, title, ylabel in zip(
        axes,
        ("train_length", "length_fraction", "repetition_fraction"),
        ("Training response length", "Length-stop fraction", "Repetition-stop fraction"),
        ("Tokens", "Fraction (%)", "Fraction (%)"),
        strict=True,
    ):
        for run in RUNS:
            points = [r for r in outers if r["run"] == run]
            vals = [r[key] * (100 if "fraction" in key else 1) for r in points]
            ax.plot(
                [r["optimizer_update"] for r in points],
                vals,
                color=colors[run],
                marker=markers[run],
            )
        frame(ax, title, ylabel)
        if "fraction" in key:
            ax.set_ylim(0, 100)
    finish(
        fig,
        "length-stopping",
        "Training rollout diagnostics; validation lengths are available in the raw W&B eval history",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    if args.refresh or not (OUT / "validation.csv").exists():
        refresh()
    plot()


if __name__ == "__main__":
    main()
