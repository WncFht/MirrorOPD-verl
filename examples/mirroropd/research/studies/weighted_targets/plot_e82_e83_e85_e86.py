# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Compare E82/E83/E85/E86 on one optimizer-update axis.

Semantic ledger: each color identifies one aggregation rule; card count is
metadata in the legend. Validation uses complete avg@16 nodes. Objective and
teacher KL are logged token means on each run's changing on-policy prefix.
Raw actor loss uses the run's own reduction and is shown on a log axis only to
show scale, never to rank the four runs. No interpolation or extrapolation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from datetime import datetime, timezone

from examples.mirroropd.research.studies.weighted_targets.plot_e81_validation import (
    ROOT,
    source,
    style,
)
from examples.mirroropd.research.studies.weighted_targets.plot_validation_comparison import save

OLD = ROOT / "examples/mirroropd/docs/experiments/learning-signal/figures/e82-e83-comparison-20261107"
NEW = ROOT / "examples/mirroropd/docs/experiments/learning-signal/figures/e85-e86-comparison-20261109"
OUT = ROOT / "examples/mirroropd/docs/experiments/learning-signal/figures/e82-e83-e85-e86-comparison-20261109"
SNAPSHOT = ROOT / "runs/wandb-e85-e86-20261109"
RUNS = {
    "E82": {"label": "E82 | token-mean | 4 GPU", "color": 0, "marker": "D"},
    "E83": {"label": "E83 | seq-mean-token-mean | 4 GPU", "color": 1, "marker": "P"},
    "E85": {"label": "E85 | token-sum | 8 GPU", "color": 2, "marker": "o"},
    "E86": {"label": "E86 | seq-mean-token-sum | 8 GPU", "color": 3, "marker": "s"},
}
BENCHMARKS = ("MATH-500", "AMC23", "AIME24", "AIME25", "AIME26")
SLUGS = {
    "MATH-500": "math500",
    "AMC23": "amc23",
    "AIME24": "aime24",
    "AIME25": "aime25",
    "AIME26": "aime26",
}


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def number(value):
    return float(value) if value not in (None, "") else float("nan")


def old_data():
    valid = []
    for row in read_csv(OLD / "validation.csv"):
        valid.append(
            {
                "run": row["run"],
                "outer": int(row["outer"]),
                "optimizer_update": int(row["optimizer_update"]),
                "benchmark": row["benchmark"],
                "accuracy": float(row["accuracy"]),
                "mean_length": float(row["mean_length"]),
                "length_rate": float(row["length_rate"]),
                "repetition_rate": float(row["repetition_rate"]),
                "stop_rate": float(row["stop_rate"]),
            }
        )
    outer = []
    for row in read_csv(OLD / "outers.csv"):
        outer.append(
            {
                "run": row["run"],
                "outer": int(row["outer"]),
                "optimizer_update": int(row["optimizer_update"]),
                "actor_loss": float(row["actor_loss"]),
                "objective": float(row["token_objective"]),
                "teacher_kl": float(row["teacher_kl"]),
                "grad_norm": float(row["grad_norm"]),
                "norm_above_one": float(row["norm_above_one"]),
                "train_length": float(row["train_length"]),
            }
        )
    return valid, outer


def wandb_data():
    valid, outer = [], []
    for run, run_id in (("E85", "sjdbuh0p"), ("E86", "sh3mrf3r")):
        rows = [json.loads(line) for line in (SNAPSHOT / run_id / "history.jsonl").open()]
        for row in rows:
            if not all(f"eval/{SLUGS[b]}/reward/mean" in row for b in BENCHMARKS):
                continue
            outer_id = int(round((row["_step"] + 1) / 17))
            for bench in BENCHMARKS:
                prefix = f"eval/{SLUGS[bench]}/"
                valid.append(
                    {
                        "run": run,
                        "outer": outer_id,
                        "optimizer_update": outer_id * 16,
                        "benchmark": bench,
                        "accuracy": float(row[prefix + "reward/mean"]),
                        "mean_length": float(row[prefix + "generation/response_length/mean"]),
                        "length_rate": float(row[prefix + "generation/finish_reason/length_fraction"]),
                        "repetition_rate": float(row[prefix + "generation/finish_reason/repetition_fraction"]),
                        "stop_rate": float(row[prefix + "generation/finish_reason/stop_fraction"]),
                    }
                )
        updates = [r for r in rows if "step/update" in r and "optim/actor/update_applied" in r]
        groups = {}
        for row in updates:
            groups.setdefault(int(row["step/outer"]), []).append(row)
        for outer_id, group in sorted(groups.items()):
            if len(group) != 16:
                continue
            summary = next(
                (r for r in reversed(rows) if r.get("step/outer") == outer_id and "rollout/response_length/mean" in r),
                None,
            )
            if summary is None:
                continue
            mean = lambda key, group=group: statistics.mean(float(r[key]) for r in group)
            outer.append(
                {
                    "run": run,
                    "outer": outer_id,
                    "optimizer_update": outer_id * 16,
                    "actor_loss": mean("optim/actor/loss"),
                    "objective": mean("distill/mdpo/objective"),
                    "teacher_kl": mean("distill/teacher_kl"),
                    "grad_norm": mean("optim/actor/grad_norm_preclip"),
                    "norm_above_one": statistics.mean(float(r["optim/actor/grad_norm_preclip"]) > 1 for r in group),
                    "train_length": float(summary["rollout/response_length/mean"]),
                }
            )
    return valid, outer


def refresh():
    valid_old, outer_old = old_data()
    valid_new, outer_new = wandb_data()
    valid, outer = valid_old + valid_new, outer_old + outer_new
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "validation.csv", valid)
    write_csv(OUT / "outers.csv", outer)
    sources = []
    for path in (OLD / "validation.csv", OLD / "outers.csv", OLD / "source-audit.json"):
        sources.append(source(path, path.read_bytes()))
    for run_id in ("sjdbuh0p", "sh3mrf3r"):
        path = SNAPSHOT / run_id / "history.jsonl"
        sources.append(source(path, path.read_bytes()))
    (OUT / "provenance.json").write_text(
        json.dumps(
            {
                "snapshot_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "sources": sources,
                "axis": "optimizer updates; 16 updates per outer",
                "validation": "complete avg@16 nodes; equal-weight Macro-5",
                "runs": {run: info["label"] for run, info in RUNS.items()},
                "limits": (
                    "Card count is metadata, not a comparison axis. One run per condition; "
                    "changing on-policy prefixes. Raw actor loss is run-specific."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    (OUT / "source-audit.json").write_text(
        json.dumps(
            {
                "E82": {
                    "source": str(OLD),
                    "last_validation_update": max(int(r["optimizer_update"]) for r in valid_old if r["run"] == "E82"),
                },
                "E83": {
                    "source": str(OLD),
                    "last_validation_update": max(int(r["optimizer_update"]) for r in valid_old if r["run"] == "E83"),
                },
                "E85": {
                    "source": "runs/wandb-e85-e86-20261109/sjdbuh0p/history.jsonl",
                    "last_validation_update": max(int(r["optimizer_update"]) for r in valid_new if r["run"] == "E85"),
                },
                "E86": {
                    "source": "runs/wandb-e85-e86-20261109/sh3mrf3r/history.jsonl",
                    "last_validation_update": max(int(r["optimizer_update"]) for r in valid_new if r["run"] == "E86"),
                },
            },
            indent=2,
        )
        + "\n"
    )


def plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator, NullLocator

    fv = style()
    theme = fv.apply_rc("paper")
    palette = fv.series_colors("paper", 9)
    valid, outer = read_csv(OUT / "validation.csv"), read_csv(OUT / "outers.csv")
    for rows in (valid, outer):
        for row in rows:
            for key in row:
                if key not in ("run", "benchmark"):
                    row[key] = number(row[key])
    colors = {run: palette[info["color"]] for run, info in RUNS.items()}
    handles = [
        Line2D([], [], color=colors[run], marker=info["marker"], label=info["label"]) for run, info in RUNS.items()
    ]
    max_update = max(r["optimizer_update"] for r in outer)

    def frame(ax, title, ylabel):
        ax.set(title=title, xlabel="Optimizer updates", ylabel=ylabel, xlim=(-max_update * 0.02, max_update * 1.04))
        fv.apply_frame(ax, theme.frame)
        fv.apply_grid(ax, theme.grid)
        ax.margins(y=0.1)

    def finish(fig, stem, caption):
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.055), ncol=2, frameon=False)
        fig.text(0.5, 0.015, caption, ha="center", color=fv.NEUTRALS["baseline"], fontsize="small")
        fig.tight_layout(rect=(0, 0.13, 1, 1))
        save(fig, OUT, stem, fv, independent=True)

    def auto_ylim(ax, floor=0.0, pad=0.08):
        values = [value for line in ax.lines for value in line.get_ydata() if math.isfinite(value)]
        if not values:
            return
        lower, upper = min(values), max(values)
        span = max(upper - lower, abs(upper) * 0.05, 1.0)
        ax.set_ylim(max(floor, lower - pad * span), upper + pad * span)

    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.8))
    for ax, bench in zip(axes.flat, ("Macro-5", *BENCHMARKS), strict=True):
        for run, info in RUNS.items():
            points = [r for r in valid if r["run"] == run]
            if bench == "Macro-5":
                grouped = {}
                for r in points:
                    grouped.setdefault(r["optimizer_update"], []).append(r["accuracy"])
                points = [{"optimizer_update": x, "accuracy": statistics.mean(y)} for x, y in sorted(grouped.items())]
            else:
                points = [r for r in points if r["benchmark"] == bench]
            ax.plot(
                [r["optimizer_update"] for r in points],
                [100 * r["accuracy"] for r in points],
                color=colors[run],
                marker=info["marker"],
            )
        frame(ax, bench, "Validation accuracy (%)")
        auto_ylim(ax)
    finish(
        fig,
        "validation-accuracy",
        "Complete avg@16 nodes; Macro-5 weights five benchmarks equally; card count is metadata",
    )

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.3))
    for ax, key, title, ylabel in zip(
        axes.flat,
        ("objective", "teacher_kl", "actor_loss", "grad_norm"),
        ("MDPO objective", "Teacher KL", "Raw actor loss", "Preclip gradient norm"),
        ("Token mean", "Token mean", "Run scale", "Run scale"),
        strict=True,
    ):
        for run, info in RUNS.items():
            points = [r for r in outer if r["run"] == run]
            ax.plot(
                [r["optimizer_update"] for r in points],
                [r[key] for r in points],
                color=colors[run],
                marker=info["marker"],
            )
        frame(ax, title, ylabel)
        if key in ("actor_loss", "grad_norm"):
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
            ax.yaxis.set_minor_locator(NullLocator())
    finish(
        fig,
        "optimization",
        "Objective and teacher KL are token means; raw actor loss is shown by scale and is not a cross-run ranking",
    )

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.4))
    for ax, key, title, ylabel in zip(
        axes,
        ("train_length", "length_rate", "repetition_rate"),
        ("Training response length", "AIME length stops", "AIME repetition stops"),
        ("Tokens", "Responses (%)", "Responses (%)"),
        strict=True,
    ):
        for run, info in RUNS.items():
            train = [r for r in outer if r["run"] == run]
            if key == "train_length":
                x, y = [r["optimizer_update"] for r in train], [r[key] for r in train]
            else:
                rows = [r for r in valid if r["run"] == run and r["benchmark"] in ("AIME24", "AIME25", "AIME26")]
                grouped = {}
                for r in rows:
                    grouped.setdefault(r["optimizer_update"], []).append(r[key])
                x, y = (
                    zip(
                        *[(u, 100 * statistics.mean(v)) for u, v in sorted(grouped.items())],
                        strict=True,
                    )
                    if grouped
                    else ([], [])
                )
            ax.plot(x, y, color=colors[run], marker=info["marker"])
        frame(ax, title, ylabel)
    finish(
        fig,
        "length-stopping",
        "Training length and AIME validation stop fractions; complete observed nodes only",
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
