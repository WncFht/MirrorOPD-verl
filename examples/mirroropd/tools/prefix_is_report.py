# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Preserve prefix-IS observations in docs and publish inner-aligned Trackio curves."""

import argparse
import hashlib
import json
from pathlib import Path


def collect(source, destination):
    """Copy completed atomic artifacts; preserve exact probabilities as evidence."""
    import shutil

    destination.mkdir(parents=True, exist_ok=True)
    for pattern in ("outer-*/trajectories.json", "outer-*/inner-*.json", "outer-*/complete.json", "banks-*.json"):
        for path in source.glob(pattern):
            target = destination / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(target.read_bytes()).digest():
                    raise RuntimeError(f"Conflicting source identity: {target}")
            else:
                temporary = target.with_suffix(target.suffix + ".tmp")
                shutil.copy2(path, temporary)
                temporary.replace(target)
    rows = []
    for path in sorted(destination.glob("outer-*/inner-*.json")):
        data = json.loads(path.read_text())
        rows.append((path, data))
    return rows


def publish(rows):
    """Use the same local SQLite storage as the existing dashboard."""
    from trackio.sqlite_storage import SQLiteStorage

    for path, row in rows:
        metrics = {"inner": row["inner"], "updates": row["update"]}
        root = path.parents[1]
        if row["inner"] == 0:
            pattern = "banks-initial-*.json" if row["batch"] == 1 else f"banks-b{row['batch'] - 1}-i16-*.json"
        else:
            pattern = f"banks-b{row['batch']}-i{row['inner']:02d}-*.json"
        for bank in root.glob(pattern):
            values = json.loads(bank.read_text())["metrics"]
            for key, value in values.items():
                metrics[key.replace("eval/rkl/", "heldout/").replace("/question_mean", "")] = value
        for denominator in ("actual_behavior", "same_scorer_anchor"):
            for horizon, values in row[denominator].items():
                for key in ("n", "ess_fraction", "log_mean_weight", "max_normalized_weight"):
                    metrics[f"{denominator}/h{horizon}/{key}"] = values[key]
                for cap in (2, 5, 10):
                    for key, value in values[f"clip{cap}"].items():
                        metrics[f"{denominator}/h{horizon}/clip{cap}/{key}"] = value
        SQLiteStorage.bulk_log(
            project="prefix-is",
            run=f"outer{row['batch']} · inner aligned",
            run_id=f"prefix-is-platform480-outer{row['batch']}",
            metrics_list=[metrics],
            steps=[row["inner"]],
            log_ids=[hashlib.sha256((str(path) + json.dumps(metrics, sort_keys=True)).encode()).hexdigest()],
            config={
                "origin": "inner-bank update480, model + AdamW restored",
                "batch": 64,
                "reuse": 16,
                "lr": 1e-6,
                "axis": "inner update",
                "training_is_weighted": False,
            },
        )


def render(rows, output):
    """One line per outer; axes share inner coordinate, not statistical independence."""
    import sys

    repo = next(p for p in output.resolve().parents if (p / "pyproject.toml").exists())
    sys.path.insert(0, str(repo.parent / "dotfiles/.agents/skills/figure-viz/assets/mpl"))
    import fvstyle as fv
    import matplotlib.pyplot as plt

    fv.apply_rc("paper")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), layout="constrained")
    colors = fv.series_colors("paper", 3)
    for ax, horizon in zip(axes, (128, 1024, 4096), strict=True):
        for batch, color in enumerate(colors, 1):
            sample = [r for _, r in rows if r["batch"] == batch and str(horizon) in r["actual_behavior"]]
            for denominator, style in (("actual_behavior", "-"), ("same_scorer_anchor", "--")):
                if sample:
                    ax.plot(
                        [r["inner"] for r in sample],
                        [r[denominator][str(horizon)]["ess_fraction"] for r in sample],
                        style,
                        color=color,
                        marker="o",
                        markersize=3,
                        label=f"Outer{batch}" if denominator == "actual_behavior" else None,
                    )
        ax.set(
            xlabel="Inner update",
            ylabel="Empirical ESS / surviving trajectories",
            title=f"Prefix {horizon} tokens",
            ylim=(0, 1.04),
            xlim=(-0.4, 16.4),
        )
        ax.grid(alpha=0.15)
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Solid: current / rollout behavior; dashed: current / same scorer at inner0", fontsize=10)
    fig.canvas.draw()
    audit = fv.audit(fig, ignore=["legend-coverage", "legend-overlaps-data", "print-fit"])
    # Legend has only the outer color; line styles are defined in the figure title.
    (output / "prefix-ess-audit.json").write_text(json.dumps(audit, default=str, indent=2) + "\n")
    for ext in ("png", "pdf"):
        fig.savefig(output / f"prefix-ess.{ext}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), layout="constrained", sharey=True)
    for ax, bank in zip(axes, ("public_initial", "public_teacher"), strict=True):
        for batch, color in enumerate(colors, 1):
            initial = "banks-initial-*.json" if batch == 1 else f"banks-b{batch - 1}-i16-*.json"
            files = [(0, p) for p in output.glob(initial)]
            files += [(int(p.name.split("-i")[1][:2]), p) for p in output.glob(f"banks-b{batch}-i*-*.json")]
            values = sorted(
                (inner, json.loads(p.read_text())["metrics"][f"eval/rkl/{bank}/math/question_mean"])
                for inner, p in files
            )
            if len(values) > 1:
                baseline = values[0][1]
                ax.plot(
                    [i for i, _ in values],
                    [v - baseline for _, v in values],
                    color=color,
                    marker="o",
                    markersize=3,
                    label=f"Outer{batch}",
                )
        ax.set(xlabel="Inner update", ylabel="Math RKL change from inner0", title=bank, xlim=(-0.4, 16.4))
        ax.axhline(0, color=colors[0], alpha=0.25, linewidth=0.7)
        ax.grid(alpha=0.15)
    if axes[0].lines and axes[0].get_legend_handles_labels()[0]:
        fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=3)
    fig.canvas.draw()
    bank_audit = fv.audit(fig, ignore=["print-fit"])
    (output / "prefix-bank-audit.json").write_text(json.dumps(bank_audit, default=str, indent=2) + "\n")
    for ext in ("png", "pdf"):
        fig.savefig(output / f"prefix-bank.{ext}", dpi=180)
    plt.close(fig)
    table = []
    for _, r in rows:
        for h, v in r["actual_behavior"].items():
            table.append(
                {
                    "batch": r["batch"],
                    "inner": r["inner"],
                    "update": r["update"],
                    "horizon": int(h),
                    "n": v["n"],
                    "actual_ess_fraction": v["ess_fraction"],
                    "anchor_ess_fraction": r["same_scorer_anchor"][h]["ess_fraction"],
                }
            )
    (output / "prefix-ess-summary.json").write_text(json.dumps(table, indent=2) + "\n")
    minimum = min((r["actual_ess_fraction"] for r in table), default=None)
    print("readouts", len(rows), "audit", audit, "min ESS", minimum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    rows = collect(args.source, args.output)
    if rows and args.publish:
        publish(rows)
    if rows and args.render:
        render(rows, args.output)
    print("prefix readouts", len(rows))


if __name__ == "__main__":
    main()
