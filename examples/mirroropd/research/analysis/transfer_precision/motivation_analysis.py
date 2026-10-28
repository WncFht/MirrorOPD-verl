# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Analyze the hypotheses behind E49/E50/E51 using local W&B histories.

Ledger: each MDPO line is one run's mean fixed-batch target KL at reuse index
1..16, normalized by the mean first observation in outer205..224. Updates are
observed before execution: index16 follows 15 optimizer attempts. Decomposition
rows separate teacher improvement, anchor cost, and net objective improvement.
Polaris panels show conditional divergence on changing student rollout states;
these are not fixed held-out risks. Same run has the same color in every panel.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

from examples.mirroropd.research.analysis.transfer_precision.analyze import RUNS, write_csv

METRICS = (
    "distill/target_kl",
    "distill/target_log_z",
    "distill/mdpo/objective",
    "distill/teacher_kl",
    "distill/anchor_kl",
    "distill/full_kl",
    "distill/full_forward_kl",
    "distill/js",
    "distill/tv",
    "distill/student_entropy",
    "distill/teacher_entropy",
)
TOL = 1e-10


def collect(logs, output):
    cohorts, trajectories, decomposition, polaris, final_eval = [], [], [], [], []
    details = {}
    for label in ("E20", "E45", "E48", "E49", "E50", "E51"):
        source = next((logs / "derived/MirrorOPD-verl").glob("*" + RUNS[label]))
        history = [json.loads(line) for line in (source / "history.jsonl").read_text().splitlines()]
        groups, outer = {}, {}
        for row in history:
            if "optim/actor/loss" in row:
                groups.setdefault(row["step/outer"], []).append(row)
            if "rollout/response_length/mean" in row:
                outer[row["step/outer"]] = row
        assert all(len(rows) == 16 for rows in groups.values())
        primary = "distill/target_kl" if label in ("E45", "E51") else "distill/full_kl"
        if label == "E49":
            primary = "distill/full_forward_kl"
        windows = [(1, max(groups)), (1, 32), (65, 96), (193, 224), (205, 224)]
        for lo, hi in windows:
            selected = {k: rows for k, rows in groups.items() if lo <= k <= hi}
            for metric in METRICS:
                if metric not in next(iter(selected.values()))[0]:
                    continue
                first = statistics.mean(rows[0][metric] for rows in selected.values())
                last = statistics.mean(rows[-1][metric] for rows in selected.values())
                cohorts.append(
                    {
                        "run": label,
                        "start": lo,
                        "end": hi,
                        "batches": len(selected),
                        "metric": metric,
                        "first": first,
                        "second": statistics.mean(rows[1][metric] for rows in selected.values()),
                        "last": last,
                        "fractional_reduction_of_means": 1 - last / first if first else None,
                        "end_worse_count": sum(rows[-1][metric] > rows[0][metric] + TOL for rows in selected.values()),
                        "first_step_rise_count": sum(
                            rows[1][metric] > rows[0][metric] + TOL for rows in selected.values()
                        ),
                        "adjacent_rises": sum(
                            b[metric] > a[metric] + TOL
                            for rows in selected.values()
                            for a, b in zip(rows, rows[1:], strict=False)
                        ),
                        "pairs": 15 * len(selected),
                    }
                )
            for index in range(16):
                vals = {
                    key: statistics.mean(rows[index][key] for rows in selected.values())
                    for key in METRICS
                    if key in next(iter(selected.values()))[0]
                }
                trajectories.append({"run": label, "start": lo, "end": hi, "reuse": index + 1, **vals})
            if label in ("E45", "E51"):
                teacher_gain = statistics.mean(
                    rows[0]["distill/teacher_kl"] - rows[-1]["distill/teacher_kl"] for rows in selected.values()
                )
                anchor_cost = statistics.mean(
                    (7 / 3) * (rows[-1]["distill/anchor_kl"] - rows[0]["distill/anchor_kl"])
                    for rows in selected.values()
                )
                decomposition.append(
                    {
                        "run": label,
                        "start": lo,
                        "end": hi,
                        "teacher_gain": teacher_gain,
                        "anchor_cost": anchor_cost,
                        "net_gain": teacher_gain - anchor_cost,
                        "cost_to_gain": anchor_cost / teacher_gain,
                    }
                )
        exceptions = {
            k: [r[primary] for r in rows] for k, rows in groups.items() if rows[-1][primary] > rows[0][primary] + TOL
        }
        details[label] = {"primary_metric": primary, "worse_batches": exceptions}
        if label in ("E45", "E51"):
            details[label]["max_log_z_drift"] = max(
                max(r["distill/target_log_z"] for r in rows) - min(r["distill/target_log_z"] for r in rows)
                for rows in groups.values()
            )
            details[label]["max_decomposition_error"] = max(
                abs(r["distill/mdpo/objective"] - r["distill/teacher_kl"] - (7 / 3) * r["distill/anchor_kl"])
                for rows in groups.values()
                for r in rows
            )
        if label in ("E48", "E49"):
            for lo in range(1, 257, 32):
                selected = {k: rows for k, rows in groups.items() if lo <= k < lo + 32}
                record = {"run": label, "start": lo, "end": lo + 31}
                for metric in METRICS:
                    if metric in next(iter(selected.values()))[0]:
                        for phase, index in (("first", 0), ("last", -1)):
                            record[metric + "/" + phase] = statistics.mean(
                                rows[index][metric] for rows in selected.values()
                            )
                record["cumulative_rkl_per_response"] = statistics.mean(
                    rows[0]["distill/full_kl"]
                    * outer[k]["rollout/response_tokens"]
                    / outer[k]["rollout/response_count"]
                    for k, rows in selected.items()
                )
                record["length_stop_fraction"] = sum(
                    outer[k]["rollout/finish_reason/length_count"] for k in selected
                ) / sum(outer[k]["rollout/response_count"] for k in selected)
                polaris.append(record)
            last = next(r for r in reversed(history) if "eval/aime25/acc/mean" in r)
            for dataset in ("aime25", "amc23", "math500"):
                prefix = "eval/" + dataset + "/"
                keys = (
                    "acc/mean",
                    "acc/pass@16",
                    "acc/majority@16",
                    "acc/all_correct@16",
                    "acc/prediction_missing_count",
                    "generation/response_length/mean",
                    "generation/finish_reason/length_count",
                )
                final_eval.append({"run": label, "dataset": dataset, **{key: last[prefix + key] for key in keys}})
    for name, rows in (
        ("inner-cohorts", cohorts),
        ("inner-trajectories", trajectories),
        ("mdpo-decomposition", decomposition),
        ("polaris-windows", polaris),
        ("polaris-evaluation", final_eval),
    ):
        write_csv(output / (name + ".csv"), rows)
    (output / "mechanism-checks.json").write_text(json.dumps(details, indent=2) + "\n")
    return trajectories, polaris


def plot(trajectories, polaris, destination, skill):
    sys.path.insert(0, str(skill / "assets/mpl"))
    import fvstyle as fv
    import matplotlib.pyplot as plt

    fv.apply_rc("paper")
    palette = fv.series_colors("paper", 5)
    colors = dict(zip(("E48", "E49", "E50", "E51", "E45"), palette, strict=True))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, lo, hi, title in zip(axes, (1, 205), (32, 224), ("Early: outer 1-32", "Late: outer 205-224"), strict=True):
        for label in ("E45", "E51"):
            rows = [r for r in trajectories if r["run"] == label and r["start"] == lo and r["end"] == hi]
            baseline = rows[0]["distill/target_kl"]
            ax.plot(
                [r["reuse"] for r in rows],
                [r["distill/target_kl"] / baseline for r in rows],
                "o-",
                color=colors[label],
                ms=3,
                label=label,
            )
        ax.axhline(1, linestyle="--", color=fv.NEUTRALS["baseline"], linewidth=0.8)
        ax.set(
            title=title,
            xlabel="Pre-update observation index",
            ylabel="Target KL / first observation",
            xticks=[1, 4, 8, 12, 16],
            ylim=(0.2, 2.4),
        )
        ax.grid(alpha=0.15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.subplots_adjust(top=0.8, bottom=0.23, wspace=0.3)
    fig.text(
        0.5,
        0.04,
        "E45: historical BF16; E51: repaired FP16 path. Index 16 follows 15 attempted updates.",
        ha="center",
        fontsize=9,
    )
    fig.canvas.draw()
    audit = {"mdpo-inner": [str(x) for x in fv.audit(fig, print_slot=10)]}
    for ext in ("png", "pdf", "svg"):
        fig.savefig(destination / ("mdpo-inner." + ext), dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for label in ("E48", "E49"):
        rows = [r for r in polaris if r["run"] == label]
        for ax, key, ylabel in zip(
            axes,
            ("distill/full_kl/first", "cumulative_rkl_per_response"),
            ("First-observation native RKL", "Cumulative conditional RKL / response"),
            strict=True,
        ):
            ax.plot(
                [(r["start"] + r["end"]) / 2 for r in rows],
                [r[key] for r in rows],
                "o-",
                color=colors[label],
                label=label,
                ms=3,
            )
            ax.set(xlabel="Outer step (32-batch windows)", ylabel=ylabel, xticks=[32, 96, 160, 224])
            ax.grid(alpha=0.15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.subplots_adjust(top=0.82, bottom=0.23, wspace=0.4)
    fig.text(
        0.5,
        0.04,
        "Online rollout states differ; cumulative values are computed per batch before averaging.",
        ha="center",
        fontsize=9,
    )
    fig.canvas.draw()
    # The two panels intentionally use token-mean and sequence-sum units.
    audit["polaris-transfer"] = [str(x) for x in fv.audit(fig, print_slot=10, ignore=["facet-ylim-mismatch"])]
    for ext in ("png", "pdf", "svg"):
        fig.savefig(destination / ("polaris-transfer." + ext), dpi=180)
    plt.close(fig)
    (destination / "mechanism-figure-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--figure-skill", type=Path, required=True)
    args = parser.parse_args()
    args.evidence.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    trajectories, polaris = collect(args.logs, args.evidence)
    plot(trajectories, polaris, args.figures, args.figure_skill)


if __name__ == "__main__":
    main()
