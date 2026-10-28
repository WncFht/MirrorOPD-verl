# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Reproduce E49–E51 snapshot analysis from locally decoded W&B records.

Semantic ledger: one line is one run/metric, never a merged population. Evaluation
points are mean accuracy across 16 responses per prompt; axes use outer steps.
KL curves average consecutive groups of 16 outer steps, separately for the first
and sixteenth pre-update observations on each batch. These are moving rollout
states, not a held-out distribution or post-update measurements. Geometry:
three equal panels per row; each benchmark has its own accuracy range. All
aggregation is exported before plotting; figures consume these frozen rows.
"""

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

RUNS = {
    "E20": "lwgb5eex",
    "E45": "625vw4ob",
    "E48": "grcuotmi",
    "E49": "zyru8wdf",
    "E50": "rsi5krqd",
    "E51": "6vv5o9b8",
}
DATASETS = ("aime25", "amc23", "math500")


def mean(values):
    values = [x for x in values if x is not None and math.isfinite(x)]
    return statistics.mean(values) if values else None


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def flatten(value, prefix=""):
    out = {}
    for key, item in value.items():
        name = prefix + key
        if isinstance(item, dict):
            out.update(flatten(item, name + "."))
        else:
            out[name] = item
    return out


def analyze(repo, output):
    summary, evaluations, batches, configs, windows = {}, [], [], {}, []
    for label, rid in RUNS.items():
        source = next((repo / "derived/MirrorOPD-verl").glob("*" + rid))
        run = json.loads((source / "run.json").read_text())
        config = json.loads((source / "config.json").read_text())
        configs[label] = flatten(config)
        history = [json.loads(line) for line in (source / "history.jsonl").read_text().splitlines()]
        updates = [row for row in history if "optim/actor/loss" in row]
        groups = {}
        for row in updates:
            groups.setdefault(row["step/outer"], []).append(row)
        skipped = [row for row in updates if row.get("optim/actor/update_skipped", 0)]
        summary[label] = {
            "id": rid,
            "name": run["run"]["display_name"],
            "status": run["status"],
            "exit_code": run["exit_code"],
            "history_rows": len(history),
            "source": str(Path(run["cache_key"]["source"]).relative_to(repo)),
            "sha256": hashlib.sha256(Path(run["cache_key"]["source"]).read_bytes()).hexdigest(),
            "axes": {
                key: max(row.get(key, 0) for row in history) for key in ("step/outer", "step/update", "step/optimizer")
            },
            "runtime_hours": max(row.get("_runtime", 0) for row in history) / 3600,
            "skipped_updates": [
                {key: val for key, val in row.items() if key.startswith(("step/", "optim/"))} for row in skipped
            ],
            "nonfinite_loss_count": sum(row.get("optim/actor/loss_nonfinite", 0) for row in updates),
            "per_outer_update_counts": sorted(set(len(rows) for rows in groups.values())),
        }
        for row in history:
            if "eval/aime25/acc/mean" in row:
                for dataset in DATASETS:
                    prefix = "eval/" + dataset + "/"
                    evaluations.append(
                        {
                            "run": label,
                            "outer": row["step/outer"],
                            "dataset": dataset,
                            **{
                                key: row.get(prefix + key)
                                for key in (
                                    "acc/mean",
                                    "acc/mean_ci_low",
                                    "acc/mean_ci_high",
                                    "acc/pass@16",
                                    "acc/majority@16",
                                    "prompt_count",
                                    "samples_per_prompt_min",
                                    "samples_per_prompt_max",
                                    "failed_count",
                                    "generation/response_length/mean",
                                    "generation/finish_reason/length_count",
                                )
                            },
                        }
                    )
        for outer, rows in groups.items():
            item = {"run": label, "outer": outer}
            for key in (
                "distill/full_kl",
                "distill/full_forward_kl",
                "distill/teacher_kl",
                "distill/anchor_kl",
                "distill/student_entropy",
                "distill/teacher_entropy",
                "distill/tv",
                "distill/js",
                "distill/teacher_outside_student_mass",
            ):
                if key in rows[0]:
                    item[key + "/first"] = rows[0][key]
                    item[key + "/last"] = rows[-1].get(key)
            batches.append(item)
        own_batches = [row for row in batches if row["run"] == label]
        for title, begin, end in (
            ("early32", 1, 32),
            ("common_late32", 193, 224),
            ("late32", max(groups) - 31, max(groups)),
        ):
            selected = [row for row in own_batches if begin <= row["outer"] <= end]
            keys = set().union(*(row.keys() for row in selected)) - {"run", "outer"}
            stats = {key: mean([row.get(key) for row in selected]) for key in sorted(keys)}
            outer_rows = [row for row in history if "rollout/score/mean" in row and begin <= row["step/outer"] <= end]
            for key in (
                "rollout/score/mean",
                "rollout/response_length/mean",
                "perf/timing/outer_step_seconds",
                "perf/timing/update_actor_seconds",
                "perf/timing/teacher_forward_seconds",
            ):
                stats[key] = mean([row.get(key) for row in outer_rows])
            windows.append({"run": label, "window": title, "start": begin, "end": end, **stats})
    diffs = {}
    for left, right in (("E48", "E49"), ("E50", "E51"), ("E20", "E50"), ("E45", "E51")):
        a, b = configs[left], configs[right]
        diffs[left + "_vs_" + right] = {
            key: [a.get(key), b.get(key)] for key in sorted(a.keys() | b.keys()) if a.get(key) != b.get(key)
        }
    result = {"runs": summary, "config_diffs": diffs}
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    write_csv(output / "evaluations.csv", evaluations)
    write_csv(output / "batches.csv", batches)
    write_csv(output / "windows.csv", windows)
    return evaluations, batches


def make_figures(evaluations, batches, output, skill):
    sys.path.insert(0, str(skill / "assets/mpl"))
    import fvstyle as fv
    import matplotlib.pyplot as plt

    fv.apply_rc("paper")
    palette = fv.series_colors("paper", 4)
    colors = dict(zip(("E48", "E49", "E50", "E51"), palette, strict=True))
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.8))
    titles = {"aime25": "AIME25 (30 prompts)", "amc23": "AMC23 (40 prompts)", "math500": "MATH-500 (500 prompts)"}
    for i, pair in enumerate((("E48", "E49"), ("E50", "E51"))):
        for j, dataset in enumerate(DATASETS):
            ax = axes[i, j]
            for label in pair:
                rows = [row for row in evaluations if row["run"] == label and row["dataset"] == dataset]
                ax.plot(
                    [row["outer"] for row in rows],
                    [100 * row["acc/mean"] for row in rows],
                    "o-",
                    ms=3,
                    color=colors[label],
                    label=label,
                )
            ax.set(
                title=titles[dataset], xlabel="Outer step", ylabel="Mean accuracy (%)", xticks=[0, 64, 128, 192, 256]
            )
            ax.grid(alpha=0.15)
    handles = [plt.Line2D([], [], color=colors[label], marker="o", label=label) for label in colors]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0))
    fig.subplots_adjust(top=0.88, bottom=0.13, hspace=0.65, wspace=0.37)
    fig.text(
        0.5,
        0.025,
        "16 samples/prompt; E51 ends at evaluation 224. One run per setting; no seed-level uncertainty.",
        ha="center",
        fontsize=9,
    )
    fig.canvas.draw()
    # Panels intentionally use different benchmark ranges; deliver at 12-inch width.
    audit = {"evaluation": [str(x) for x in fv.audit(fig, ignore=["facet-ylim-mismatch"], print_slot=12)]}
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output / ("evaluation." + ext), dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    plot_rows = []
    for ax, label, metric in zip(
        axes, ("E49", "E50", "E51"), ("distill/full_forward_kl", "distill/full_kl", "distill/teacher_kl"), strict=True
    ):
        rows = [row for row in batches if row["run"] == label]
        for phase, style in (("first", "--"), ("last", "-")):
            agg = []
            for start in range(1, int(max(row["outer"] for row in rows)) + 1, 16):
                chunk = [row for row in rows if start <= row["outer"] < start + 16]
                agg.append(
                    {
                        "run": label,
                        "metric": metric,
                        "phase": phase,
                        "outer": mean([row["outer"] for row in chunk]),
                        "value": mean([row.get(metric + "/" + phase) for row in chunk]),
                    }
                )
            plot_rows.extend(agg)
            ax.plot(
                [row["outer"] for row in agg],
                [row["value"] for row in agg],
                style,
                color=colors[label],
                label="_nolegend_",
            )
        ax.set(
            title=label + (" / Forward KL" if label == "E49" else " / Teacher reverse KL"),
            xlabel="Outer step",
            ylabel="Token-mean KL",
            yscale="log",
            xticks=[0, 64, 128, 192, 256],
        )
        ax.grid(alpha=0.15)
    handles = [
        plt.Line2D([], [], color="0.3", linestyle=style, label=label)
        for label, style in (("First pre-update observation", "--"), ("16th pre-update observation", "-"))
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2)
    fig.subplots_adjust(top=0.78, bottom=0.24, wspace=0.42)
    fig.text(
        0.5,
        0.045,
        "Each point averages up to 16 outer steps; different teachers/objectives across panels. Logarithmic y axes.",
        ha="center",
        fontsize=9,
    )
    fig.canvas.draw()
    # Panels identify run/color in their titles; the shared legend explains line style.
    audit["kl"] = [str(x) for x in fv.audit(fig, ignore=["facet-ylim-mismatch", "legend-coverage"], print_slot=12)]
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output / ("kl." + ext), dpi=180)
    plt.close(fig)
    write_csv(output / "kl-plot-data.csv", plot_rows)
    (output / "figure-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
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
    evaluations, batches = analyze(args.logs.resolve(), args.evidence)
    make_figures(evaluations, batches, args.figures, args.figure_skill)


if __name__ == "__main__":
    main()
