# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Audit and plot E50–E53 diagnostics from local records.

Semantic ledger: each color denotes one run throughout. Training curves are
nonoverlapping 16-outer means, not independent seeds. Evaluation points retain
their actual checkpoints. Positions are deciles of each observed response,
not absolute token positions or guaranteed answer spans. Position gains use
sum/count aggregation, never an unweighted mean of per-token-bin means.
Geometric ledger: equally sized panels; common outer axis 1..256, E53 stops at
103. Reuse coordinates 1..16 are PRE-update observations (15 intervening
attempts). Probability axes are percent; log axes contain positive values.
Plots consume exported aggregation rows. No cloud calls or training occur.
"""

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from examples.mirroropd.research.core.io import write_csv
from examples.mirroropd.research.studies.production_precision.precision_comparison import DATASETS, RUNS

WINDOWS = (("early", 1, 32), ("middle", 65, 96), ("late", 225, 256))


def collect(logs, out):
    out.mkdir(parents=True, exist_ok=True)
    tables = {
        key: [] for key in ("training", "inner", "evaluation", "sampling", "positions", "trajectories", "inventory")
    }
    audit = {}
    for label, rid in RUNS.items():
        (source,) = (logs / "derived/MirrorOPD-verl").glob("*" + rid)
        meta = json.loads((source / "run.json").read_text())
        raw = Path(meta["cache_key"]["source"])
        stat = raw.stat()
        if (stat.st_size, stat.st_mtime_ns) != (meta["cache_key"]["bytes"], meta["cache_key"]["mtime_ns"]):
            raise ValueError(f"Stale decoded log: {label}")
        if meta["status"] not in ("finished", "snapshot", "partial_snapshot"):
            raise ValueError(f"Unusable snapshot: {label}")
        history = [json.loads(line) for line in (source / "history.jsonl").read_text().splitlines()]
        keys = Counter(k for row in history for k in row)
        tables["inventory"].extend({"run": label, "key": k, "record_count": n} for k, n in sorted(keys.items()))
        groups = defaultdict(list)
        primary = "distill/target_kl" if label in ("E51", "E53") else "distill/full_kl"
        teacher = "distill/teacher_kl" if label in ("E51", "E53") else "distill/full_kl"
        position_prefix = "distill/mdpo/objective_position" if label in ("E51", "E53") else "distill/full_kl_position"
        audit[label] = {
            "source_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
            "keys": len(keys),
            "position_reconstruction_max_error": 0.0,
            "position_count_changes": 0,
            "alias_max_error": 0.0,
            "max_target_identity_error": 0.0,
            "empty_mask_max": 0.0,
            "staleness_max": 0.0,
            "empty_mask_records": 0,
            "staleness_records": 0,
        }
        for row in history:
            outer = row.get("step/outer")
            if "optim/actor/loss" in row:
                groups[outer].append(row)
            if "rollout/response_tokens" in row:
                n = row["rollout/response_count"]
                tables["training"].append(
                    {
                        "run": label,
                        "outer": outer,
                        "tokens": row["rollout/response_tokens"],
                        "response_count": n,
                        "length": row["rollout/response_length/mean"],
                        "length_stop": row["rollout/finish_reason/length_count"] / n,
                        "actor_seconds": row["perf/timing/update_actor_seconds"],
                        "outer_seconds": row["perf/timing/outer_step_seconds"],
                        "actor_peak_gib": row["perf/actor/allocated_peak_gib"],
                        "rollout_score": row.get("rollout/score/mean"),
                        "actor_ms_per_response_token": 1000
                        * row["perf/timing/update_actor_seconds"]
                        / row["rollout/response_tokens"],
                    }
                )
                audit[label]["empty_mask_max"] = max(
                    audit[label]["empty_mask_max"], row.get("rollout/empty_loss_mask_fraction", 0)
                )
                audit[label]["staleness_max"] = max(
                    audit[label]["staleness_max"], row.get("rollout/staleness/trajectory_staleness/max", 0)
                )
                audit[label]["empty_mask_records"] += "rollout/empty_loss_mask_fraction" in row
                audit[label]["staleness_records"] += "rollout/staleness/trajectory_staleness/max" in row
            for dataset in DATASETS:
                p = f"eval/{dataset}/"
                if p + "acc/mean" not in row:
                    continue
                n = row[p + "response_count"]
                values = {
                    "run": label,
                    "outer": outer,
                    "dataset": dataset,
                    "response_count": n,
                    "prompts": row[p + "prompt_count"],
                    "mean": row[p + "acc/mean"],
                    "ci_low": row[p + "acc/mean_ci_low"],
                    "ci_high": row[p + "acc/mean_ci_high"],
                    "pass16": row[p + "acc/pass@16"],
                    "majority16": row[p + "acc/majority@16"],
                    "all16": row[p + "acc/all_correct@16"],
                    "within_prompt_std": row[p + "acc/within_prompt_std"],
                    "length": row[p + "generation/response_length/mean"],
                    "length_stop": row[p + "generation/finish_reason/length_count"] / n,
                    "missing_prediction": row[p + "acc/prediction_missing_count"] / n,
                    "missing_score": row[p + "acc/missing_count"],
                    "failed": row[p + "failed_count"],
                    "nonfinite": row[p + "nonfinite_score_count"],
                    "materialized": row[p + "materialized_response_count"],
                    "format": row[p + "format/mean"],
                    "prediction_conflict": row[p + "acc/prediction_score_conflict_fraction"],
                }
                if values["materialized"] != n or n != 16 * values["prompts"]:
                    raise ValueError(f"Evaluation population mismatch: {label}/{outer}/{dataset}")
                values["correct_responses"] = round(values["mean"] * n)
                values["covered_prompts"] = round(values["pass16"] * values["prompts"])
                values["correct_per_covered_prompt"] = (
                    values["correct_responses"] / values["covered_prompts"] if values["covered_prompts"] else None
                )
                values["completion_union_upper_bound_pp"] = 100 * (values["length_stop"] + values["missing_prediction"])
                tables["evaluation"].append(values)
                audit[label]["alias_max_error"] = max(
                    audit[label]["alias_max_error"],
                    *(abs(row[p + kind + "/mean"] - values["mean"]) for kind in ("score", "reward")),
                )
                for k in range(1, 17):
                    tables["sampling"].append(
                        {
                            "run": label,
                            "outer": outer,
                            "dataset": dataset,
                            "k": k,
                            "pass": row[p + f"acc/pass@{k}"],
                            "majority": row[p + f"acc/majority@{k}"],
                            "all_correct": row[p + f"acc/all_correct@{k}"],
                        }
                    )
        for outer, rows in groups.items():
            if len(rows) != 16:
                raise ValueError(f"Incomplete reuse group: {label}/{outer}")
            for index, row in enumerate(rows):
                item = {
                    "run": label,
                    "outer": outer,
                    "reuse": index + 1,
                    "teacher": row[teacher],
                    "primary": row[primary],
                    "primary_ratio": row[primary] / rows[0][primary],
                    "grad_norm": row.get("optim/actor/grad_norm_preclip", math.nan),
                    "skipped": row["optim/actor/update_skipped"],
                }
                if label in ("E51", "E53"):
                    item.update(
                        teacher_gain=rows[0][teacher] - row[teacher],
                        anchor_cost=(7 / 3) * (row["distill/anchor_kl"] - rows[0]["distill/anchor_kl"]),
                    )
                    item["net_gain"] = item["teacher_gain"] - item["anchor_cost"]
                    residual = abs(
                        row[primary] - 0.3 * row[teacher] - 0.7 * row["distill/anchor_kl"] - row["distill/target_log_z"]
                    )
                    audit[label]["max_target_identity_error"] = max(audit[label]["max_target_identity_error"], residual)
                tables["inner"].append(item)
                sums = [row[f"{position_prefix}/p{b:02d}_{b + 10:02d}/sum"] for b in range(0, 100, 10)]
                counts = [row[f"{position_prefix}/p{b:02d}_{b + 10:02d}/token_count"] for b in range(0, 100, 10)]
                expected = row.get("distill/mdpo/objective", row[teacher])
                audit[label]["position_reconstruction_max_error"] = max(
                    audit[label]["position_reconstruction_max_error"], abs(math.fsum(sums) / sum(counts) - expected)
                )
                for b in range(0, 100, 10):
                    key = f"{position_prefix}/p{b:02d}_{b + 10:02d}/token_count"
                    audit[label]["position_count_changes"] += row[key] != rows[0][key]
        for window, lo, hi in WINDOWS:
            selected = [rows for outer, rows in groups.items() if lo <= outer <= hi]
            if len(selected) != hi - lo + 1:
                continue
            first = statistics.mean(rows[0][primary] for rows in selected)
            for index in range(16):
                tables["trajectories"].append(
                    {
                        "run": label,
                        "window": window,
                        "reuse": index + 1,
                        "normalized_primary": statistics.mean(rows[index][primary] for rows in selected) / first,
                    }
                )
            for b in range(0, 100, 10):
                prefix = f"{position_prefix}/p{b:02d}_{b + 10:02d}/"
                n = sum(rows[0][prefix + "token_count"] for rows in selected)
                # Python 3.12 changed built-in float sum; fsum keeps frozen
                # aggregates consistent with the Python 3.10 decoding host.
                initial = math.fsum(rows[0][prefix + "sum"] for rows in selected) / n
                tables["positions"].append(
                    {
                        "run": label,
                        "window": window,
                        "bin_start": b,
                        "count": n,
                        "first": initial,
                        "first_step_gain": initial - math.fsum(rows[1][prefix + "sum"] for rows in selected) / n,
                        "net_gain": initial - math.fsum(rows[-1][prefix + "sum"] for rows in selected) / n,
                    }
                )
        audit[label]["skipped_events"] = [
            {
                "outer": r["outer"],
                "reuse": r["reuse"],
                "grad_norm": r["grad_norm"] if math.isfinite(r["grad_norm"]) else None,
            }
            for r in tables["inner"]
            if r["run"] == label and r["skipped"]
        ]
        update_rows = [row for rows in groups.values() for row in rows]
        audit[label]["loss_flag_records"] = sum("optim/actor/loss_nonfinite" in row for row in update_rows)
        audit[label]["nonfinite_loss_count"] = sum(row.get("optim/actor/loss_nonfinite", 0) for row in update_rows)
        audit[label]["nonfinite_gradient_count"] = sum(
            bool(
                row.get("optim/actor/grad_norm_preclip_nonfinite", 0)
                or row.get("optim/actor/grad_norm_preclip/nonfinite", 0)
            )
            for row in update_rows
        )
    for name, rows in tables.items():
        write_csv(out / (name + ".csv"), rows)
    (out / "metric-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return tables, audit


def plot(tables, destination, skill, source_tables):
    sys.path.insert(0, str(skill / "assets/mpl"))
    from examples.mirroropd.research.core.plotting import load_style

    fv = load_style()
    import matplotlib.pyplot as plt

    destination.mkdir(parents=True, exist_ok=True)
    fv.apply_rc("paper")
    colors = dict(zip(RUNS, fv.series_colors("paper", 4), strict=True))
    audits = {}
    plotted = {}

    def panel(rows, cols, height=3.8):
        fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, height * rows), squeeze=False)
        fig.subplots_adjust(
            left=0.09,
            right=0.98,
            bottom=0.22 if rows == 1 else 0.12,
            top=0.8 if rows == 1 else 0.9,
            wspace=0.36,
            hspace=0.5,
        )
        return fig, axes

    def finish(fig, name, caption):
        present = {line.get_label() for axis in fig.axes for line in axis.lines}
        handles = [plt.Line2D([], [], color=colors[r], label=r) for r in RUNS if r in present]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1), ncol=4, frameon=False)
        fig.text(0.5, 0.015, caption, ha="center", fontsize=8, color=fv.NEUTRALS["baseline"])
        fig.canvas.draw()
        # Different benchmarks have distinct difficulty/ranges, explicitly
        # stated in the captions; do not force 30% and 95% onto a flat shared axis.
        scoped = (
            [("facet-ylim-mismatch", axis) for axis in fig.axes] if name in ("accuracy", "sampling-coverage") else []
        )
        issues = fv.audit(fig, scoped_ignore=scoped, print_slot=fig.get_figwidth())
        audits[name] = issues
        plotted[name] = [
            {
                "title": axis.get_title(),
                "xlabel": axis.get_xlabel(),
                "ylabel": axis.get_ylabel(),
                "xlim": list(axis.get_xlim()),
                "ylim": list(axis.get_ylim()),
                "series": [
                    {
                        "label": line.get_label(),
                        "x": list(map(float, line.get_xdata())),
                        "y": list(map(float, line.get_ydata())),
                    }
                    for line in axis.lines
                    if not (line.get_gid() or "").startswith("fv-")
                ],
                "points": [collection.get_offsets().tolist() for collection in axis.collections],
            }
            for axis in fig.axes
        ]
        fv.save(fig, name, outdir=destination, formats=["png", "pdf"])

    def curve(ax, rows, x, y, label, scale=1, style="-", marker=None):
        ax.plot(
            [r[x] for r in rows],
            [scale * r[y] for r in rows],
            color=colors[label],
            linestyle=style,
            marker=marker,
            markersize=3,
            linewidth=1.5,
            label=label if style == "-" else None,
        )

    def smooth(rows, metrics):
        groups = defaultdict(list)
        for r in rows:
            groups[(r["outer"] - 1) // 16].append(r)
        return [
            {
                "outer": statistics.mean(r["outer"] for r in g),
                **{m: statistics.mean(r[m] for r in g if math.isfinite(r[m])) for m in metrics},
            }
            for g in groups.values()
        ]

    fig, ax = panel(2, 2)
    for label in RUNS:
        rows = [r for r in tables["inner"] if r["run"] == label]
        for axis, reuse in ((ax[0, 0], 1), (ax[0, 1], 16)):
            curve(axis, smooth([r for r in rows if r["reuse"] == reuse], ["teacher"]), "outer", "teacher", label)
        per_batch = []
        for outer in sorted({r["outer"] for r in rows}):
            g = [r for r in rows if r["outer"] == outer]
            finite = [r["grad_norm"] for r in g if math.isfinite(r["grad_norm"])]
            per_batch.append(
                {"outer": outer, "grad": statistics.median(finite), "gain": 100 * (1 - g[-1]["primary_ratio"])}
            )
        curve(ax[1, 0], smooth(per_batch, ["grad"]), "outer", "grad", label)
        curve(ax[1, 1], smooth(per_batch, ["gain"]), "outer", "gain", label)
    for axis, title, ylabel in zip(
        ax.flat,
        [
            "New-batch teacher RKL",
            "Last observed teacher RKL",
            "Median finite pre-clip gradient",
            "Within-batch primary reduction",
        ],
        ["RKL (nats/token)", "RKL (nats/token)", "Gradient norm", "Reduction (%)"],
        strict=True,
    ):
        axis.set(title=title, xlabel="Outer step", ylabel=ylabel)
    for axis in (ax[0, 0], ax[0, 1], ax[1, 0]):
        axis.set_yscale("log")
    lower = min(axis.get_ylim()[0] for axis in ax[0])
    upper = max(axis.get_ylim()[1] for axis in ax[0])
    for axis in ax[0]:
        axis.set_ylim(lower, upper)
    finish(
        fig,
        "training-dynamics",
        "16-outer means; primary = teacher RKL (OPD) or target KL (MDPO); pre-update observations",
    )

    fig, ax = panel(1, 2)
    for label in ("E50", "E51", "E52"):
        rows = [r for r in tables["inner"] if r["run"] == label and r["reuse"] == 1 and 148 <= r["outer"] <= 164]
        curve(ax[0, 0], rows, "outer", "teacher", label, marker="o")
        rows = [r for r in tables["inner"] if r["run"] == label and r["outer"] == 156]
        curve(ax[0, 1], rows, "reuse", "teacher", label, marker="o")
        skipped = [r for r in rows if r["skipped"]]
        fv.mark_neutral(
            ax[0, 1].scatter(
                [r["reuse"] for r in skipped],
                [r["teacher"] for r in skipped],
                marker="x",
                color=colors[label],
                s=70,
                zorder=5,
            )
        )
    ax[0, 0].set(title="New-batch RKL around outer 156", xlabel="Outer step", ylabel="Teacher RKL (nats/token)")
    ax[0, 1].set(
        title="Inner trajectory at outer 156", xlabel="Pre-update reuse observation", ylabel="Teacher RKL (nats/token)"
    )
    ax[0, 1].set_xticks([1, 4, 8, 12, 16])
    for axis in ax.flat:
        axis.set_yscale("log")
        axis.set_ylim(2e-4, 3e-2)
    finish(
        fig,
        "numerical-event",
        "Observed rare event; crosses mark skipped attempts after that observation; no causal replay",
    )

    fig, ax = panel(2, 2)
    for label in ("E51", "E53"):
        for axis, window in ((ax[0, 0], "early"), (ax[0, 1], "middle")):
            rows = [r for r in tables["trajectories"] if r["run"] == label and r["window"] == window]
            curve(axis, rows, "reuse", "normalized_primary", label)
        rows = [r for r in tables["inner"] if r["run"] == label and r["reuse"] == 2]
        curve(ax[1, 0], smooth(rows, ["primary_ratio"]), "outer", "primary_ratio", label)
        rows = [
            {**r, "cost_fraction": r["anchor_cost"] / r["teacher_gain"]}
            for r in tables["inner"]
            if r["run"] == label and r["reuse"] == 16
        ]
        curve(ax[1, 1], smooth(rows, ["cost_fraction"]), "outer", "cost_fraction", label)
    for axis, title in ((ax[0, 0], "Outer 1–32"), (ax[0, 1], "Outer 65–96")):
        axis.set(title=title, xlabel="Pre-update reuse observation", ylabel="Target KL / initial mean")
        axis.set_ylim(0.2, 1.15)
        axis.set_xticks([1, 4, 8, 12, 16])
    ax[1, 0].set(title="First-step target change", xlabel="Outer step", ylabel="Second / first target KL")
    ax[1, 1].set(title="Anchor cost versus teacher gain", xlabel="Outer step", ylabel="Weighted anchor cost / gain")
    for axis in ax.flat:
        fv.mark_neutral(axis.axhline(1, color=fv.NEUTRALS["baseline"], lw=0.8, ls=":"))
    finish(
        fig,
        "mdpo-mechanism",
        "Top: ratios of cohort means. Bottom: 16-outer means of batch ratios; E53 ends at outer 103",
    )

    fig, ax = panel(1, 2)
    for label in ("E51", "E53"):
        rows = [r for r in tables["positions"] if r["run"] == label and r["window"] == "middle"]
        for axis, metric in ((ax[0, 0], "first_step_gain"), (ax[0, 1], "net_gain")):
            curve(axis, rows, "bin_start", metric, label, scale=1e5, marker="o")
    for axis, title in zip(ax.flat, ["First-step objective gain", "First-to-last objective gain"], strict=True):
        axis.set(title=title, xlabel="Response decile start (%)", ylabel="Objective decrease (× 1e−5)")
        axis.set_ylim(-6, 38)
        fv.mark_neutral(axis.axhline(0, color=fv.NEUTRALS["baseline"], lw=0.8))
    finish(
        fig,
        "position-gains",
        "Outer 65–96; token-weighted MDPO objective gain; relative observed response position, not answer position",
    )

    for name, metrics in (
        ("accuracy", ["mean"]),
        ("sampling-coverage", ["pass16", "majority16"]),
        ("validation-behavior", ["length_stop", "missing_prediction"]),
    ):
        fig, ax = panel(len(metrics), 3)
        titles = {
            "mean": "Mean accuracy (%)",
            "pass16": "Pass@16 (%)",
            "majority16": "Majority@16 (%)",
            "length_stop": "Length-limited responses (%)",
            "missing_prediction": "Unparsed predictions (%)",
        }
        for i, metric in enumerate(metrics):
            for j, dataset in enumerate(DATASETS):
                for label in RUNS:
                    rows = [r for r in tables["evaluation"] if r["run"] == label and r["dataset"] == dataset]
                    curve(ax[i, j], rows, "outer", metric, label, scale=100, marker="o")
                ax[i, j].set(title=dataset.upper(), xlabel="Validation outer", ylabel=titles[metric])
                if name == "validation-behavior":
                    ax[i, j].set_ylim(0, 4)
        finish(
            fig,
            name,
            "16 responses/prompt; single seed; benchmark-specific y scales for accuracy/coverage; "
            "E53 stops at outer 96",
        )

    fig, ax = panel(1, 2)
    for label in ("E50", "E51", "E52"):
        rows = [r for r in tables["sampling"] if r["run"] == label and r["outer"] == 256 and r["dataset"] == "aime25"]
        for axis, metric in ((ax[0, 0], "pass"), (ax[0, 1], "majority")):
            curve(axis, rows, "k", metric, label, scale=100, marker="o")
    for axis, title in zip(ax.flat, ["AIME25: at least one correct", "AIME25: majority vote"], strict=True):
        axis.set(title=title, xlabel="Responses k (subsets of 16)", ylabel="Accuracy (%)", ylim=(25, 65))
        axis.set_xticks([1, 4, 8, 12, 16])
    finish(
        fig,
        "sampling-budget",
        "Outer 256; k curves reuse the same 16 samples/prompt; intermediate majority uses Monte Carlo subsets",
    )

    fig, ax = panel(2, 2)
    for label in RUNS:
        rows = smooth(
            [r for r in tables["training"] if r["run"] == label],
            ["length", "length_stop", "actor_ms_per_response_token", "actor_peak_gib"],
        )
        for axis, metric, scale in zip(
            ax.flat,
            ["length", "length_stop", "actor_ms_per_response_token", "actor_peak_gib"],
            [1, 100, 1, 1],
            strict=True,
        ):
            curve(axis, rows, "outer", metric, label, scale=scale)
    for axis, title, ylabel in zip(
        ax.flat,
        ["Training response length", "Training length limit", "Actor compute per new token", "Actor allocated peak"],
        ["Tokens", "Responses (%)", "ms / new response token", "GiB"],
        strict=True,
    ):
        axis.set(title=title, xlabel="Outer step", ylabel=ylabel)
    finish(fig, "generation-cost", "16-outer means; A100: E50/E52, A800: E51/E53; CPU count also differs for E51/E53")
    (destination / "figure-audit.json").write_text(json.dumps(audits, indent=2) + "\n")
    (destination / "plotted-data.json").write_text(json.dumps(plotted, indent=2) + "\n")
    (destination / "figure-design.json").write_text(
        json.dumps(
            {
                "print_slot": "native figure width; no forced downscaling to a single paper column",
                "scoped_exceptions": {
                    "accuracy": "facet-ylim-mismatch: distinct benchmarks",
                    "sampling-coverage": "facet-ylim-mismatch: distinct benchmarks",
                },
                "source_tables": "wandb-logs/MirrorOPD-verl/analysis/e50-e53-20261029/diagnostics",
                "source_table_sha256": {
                    name: hashlib.sha256((source_tables / (name + ".csv")).read_bytes()).hexdigest() for name in tables
                },
                "ledger": __doc__,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(audits, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--figures", type=Path)
    parser.add_argument("--figure-skill", type=Path)
    args = parser.parse_args()
    tables, audit = collect(args.logs_repo.resolve(), args.out.resolve())
    print(json.dumps(audit, indent=2))
    if args.figures:
        if not args.figure_skill:
            parser.error("--figures requires --figure-skill")
        plot(tables, args.figures, args.figure_skill, args.out.resolve())


if __name__ == "__main__":
    main()
