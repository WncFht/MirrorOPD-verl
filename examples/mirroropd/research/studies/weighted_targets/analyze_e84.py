# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Preserve the E84 console export and compare its observed results with E82.

Semantic ledger: E82 is immediate MDPO; E84 adds strict-suffix mean cost.
Accuracy is benchmark mean, with MATH n16/n4 and all other benchmarks n16.
Each curve point is an observed complete validation, never an interpolation.
Geometry ledger: x is explicit outer * 16 optimizer updates. Training console
values are outer summaries, not the full history of 1600 individual updates.
Only the final E84 pprint contains stop fractions; earlier ones stay missing.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
import shutil
import statistics
from pathlib import Path

from examples.mirroropd.research.studies.weighted_targets.plot_e81_validation import ROOT, SLUGS, source, style
from examples.mirroropd.research.studies.weighted_targets.plot_validation_comparison import save

BASE = ROOT / "examples/mirroropd/docs/experiments/learning-signal"
DATA = BASE / "data/e84-20261110"
OUT = BASE / "figures/e82-e84-20261110"
RAW = ROOT / "runs/e84-results-20261110/train-console.txt"
REFERENCE = BASE / "figures/validation-comparison-20261106/validation.csv"
BENCHMARKS = tuple(SLUGS)


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def extract():
    raw = RAW.read_bytes()
    lines = [re.sub(r"^\(TaskRunnerV1 pid=\d+\) ?", "", line) for line in raw.decode().splitlines()]
    start = next(i for i, line in enumerate(lines) if line.startswith("{'actor_rollout_ref':"))
    for end in range(start + 1, len(lines)):
        if not lines[end].rstrip().endswith("}"):
            continue
        try:
            config = ast.literal_eval("\n".join(lines[start : end + 1]))
            break
        except (SyntaxError, ValueError):
            continue
    else:
        raise ValueError("No complete resolved config")
    start = next(i for i, line in enumerate(lines) if line.startswith('("Final validation metrics:'))
    end = next(i for i in range(start, len(lines)) if lines[i].endswith('}")'))
    # Ray/progress output interrupts pprint; retain its quoted string fragments.
    expression = "\n".join(line for line in lines[start : end + 1] if line.lstrip().startswith(('"', "'", '("')))
    final = ast.literal_eval(ast.literal_eval(expression).removeprefix("Final validation metrics: "))
    console = []
    for line_no, line in enumerate(lines, 1):
        if not line.startswith("step:"):
            continue
        values = {k: float(v) for k, v in re.findall(r"([\w/]+):([-+\deE.]+)(?= - |$)", line)}
        if values:
            console.append({"source_line": line_no, "metrics": values})
    train = [row for row in console if "step/optimizer" in row["metrics"]]
    valid = [row for row in console if "eval/math500/acc/mean" in row["metrics"]]
    assert [int(row["metrics"]["step/outer"]) for row in train] == list(range(1, 101))
    assert [int(row["metrics"]["step"]) for row in valid] == [20, 40, 60, 80, 100]
    assert config["distillation"]["distillation_loss"]["full_vocab"]["future"] == {"gamma": 1.0, "mode": "mean"}
    assert "E84" in config["trainer"]["experiment_name"]
    validation = []
    for entry in valid:
        metrics = entry["metrics"]
        outer = int(metrics["step"])
        for benchmark, slug in SLUGS.items():
            prefix = f"eval/{slug}/"
            if outer == 100:
                for key, value in metrics.items():
                    if key.startswith(prefix) and key in final:
                        assert math.isclose(value, final[key], abs_tol=1e-10)
            responses = int(metrics[prefix + "response_count"])
            expected = 2000 if benchmark == "MATH-500" else 640 if benchmark == "AMC23" else 480
            assert responses == expected and metrics[prefix + "failed_count"] == 0
            accuracy = metrics[prefix + "acc/mean"]
            assert math.isclose(accuracy * responses, round(accuracy * responses), abs_tol=1e-8)
            row = dict(
                run="E84",
                benchmark=benchmark,
                outer=outer,
                optimizer_update=outer * 16,
                response_presentations=outer * 4096,
                fresh_responses=outer * 256,
                responses=responses,
                accuracy=accuracy,
                mean_length=metrics[prefix + "generation/response_length/mean"],
            )
            for field, suffix in (
                ("length_rate", "length_fraction"),
                ("repetition_rate", "repetition_fraction"),
                ("stop_rate", "stop_fraction"),
            ):
                row[field] = final[prefix + "generation/finish_reason/" + suffix] if outer == 100 else ""
            validation.append(row)
    training = []
    for entry in train:
        m = entry["metrics"]
        outer = int(m["step/outer"])
        assert m["step/optimizer"] == m["step/update"] == 16 * outer
        training.append(
            dict(
                run="E84",
                outer=outer,
                optimizer_update=16 * outer,
                actor_loss=m["optim/actor/loss"],
                grad_norm=m["optim/actor/grad_norm_preclip"],
                train_length=m["rollout/response_length/mean"],
                reward=m["rollout/score/mean"],
                rollout_seconds=m["perf/timing/rollout_batch_seconds"],
                actor_seconds=m["perf/timing/update_actor_seconds"],
                outer_seconds=m["perf/timing/outer_step_seconds"],
                validation_seconds=m.get("perf/timing/validation_seconds", ""),
            )
        )
    endpoints = [row for row in validation if row["outer"] == 100]
    aime = [row for row in endpoints if row["benchmark"].startswith("AIME")]
    details = []
    for row in endpoints:
        prefix = f"eval/{SLUGS[row['benchmark']]}/"
        n = 4 if row["benchmark"] == "MATH-500" else 16
        details.append(
            {
                **row,
                "samples_per_prompt": n,
                "correct_responses": round(row["accuracy"] * row["responses"]),
                "accuracy_ci_low": final[prefix + "acc/mean_ci_low"],
                "accuracy_ci_high": final[prefix + "acc/mean_ci_high"],
                "length_p50": final[prefix + "generation/response_length/p50"],
                "length_p95": final[prefix + "generation/response_length/p95"],
                "best_at_n": final[prefix + f"score/best@{n}"],
                "majority_at_n": final[prefix + f"acc/majority@{n}"],
                "prediction_missing_count": final[prefix + "acc/prediction_missing_count"],
            }
        )
    DATA.mkdir(parents=True, exist_ok=True)
    dump(DATA / "resolved-config.json", config)
    dump(DATA / "final-validation-metrics.json", final)
    with (DATA / "console-metrics.jsonl").open("w") as stream:
        for row in console:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    write_csv(DATA / "validation.csv", validation)
    write_csv(DATA / "training-outers.csv", training)
    write_csv(DATA / "final-benchmarks.csv", details)
    audit = dict(
        run="E84",
        wandb_run="x73jzh0z",
        training_outers=len(train),
        last_optimizer_counter=1600,
        complete_validation_outers=[20, 40, 60, 80, 100],
        validation_responses=20400,
        final_metric_count=len(final),
        macro5=statistics.mean(row["accuracy"] for row in endpoints),
        aime_mean=statistics.mean(row["accuracy"] for row in aime),
        aime_length=statistics.mean(row["mean_length"] for row in aime),
        aime_length_rate=statistics.mean(row["length_rate"] for row in aime),
        aime_repetition_rate=statistics.mean(row["repetition_rate"] for row in aime),
        aime_stop_rate=statistics.mean(row["stop_rate"] for row in aime),
        training_seconds=sum(row["outer_seconds"] for row in training),
        validation_seconds=sum(float(row["validation_seconds"]) for row in training if row["validation_seconds"] != ""),
        final_progress="100/100 [31:54:22]",
        wandb_teardown="BrokenPipeError after final validation and progress completion",
        evidence_scope=(
            "Full received console preserved; no per-update history, "
            "validation responses or checkpoint tensors in attachment."
        ),
        sources=[source(RAW, raw), source(Path(__file__), Path(__file__).read_bytes())],
    )
    dump(DATA / "summary.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fv = style()
    theme = fv.apply_rc("paper")
    palette = fv.series_colors("paper", 9)
    colors = {"E82": palette[4], "E84": palette[1]}
    labels = {"E82": "E82 immediate | MATH n16", "E84": "E84 future mean | MATH n4"}
    rows = []
    for path, run in ((REFERENCE, "E82"), (DATA / "validation.csv", "E84")):
        for row in csv.DictReader(path.open()):
            if row["run"] == run:
                rows.append(
                    {k: v if k in ("run", "benchmark") else float(v) if v != "" else None for k, v in row.items()}
                )
    OUT.mkdir(parents=True, exist_ok=True)

    def finish(fig, stem, caption):
        handles = [
            Line2D([], [], color=colors[run], marker="D" if run == "E82" else "o", label=labels[run]) for run in colors
        ]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.055), ncol=2, frameon=False)
        fig.text(0.5, 0.017, caption, ha="center", fontsize="small", color=fv.NEUTRALS["baseline"])
        fig.tight_layout(rect=(0, 0.14, 1, 1))
        save(fig, OUT, stem, fv, independent=True)

    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.6))
    for ax, benchmark in zip(axes.flat, ("Macro-5", *BENCHMARKS), strict=True):
        for run in colors:
            points = []
            for outer in sorted({r["outer"] for r in rows if r["run"] == run}):
                group = [
                    r
                    for r in rows
                    if r["run"] == run
                    and r["outer"] == outer
                    and (benchmark == "Macro-5" or r["benchmark"] == benchmark)
                ]
                assert len(group) == (5 if benchmark == "Macro-5" else 1)
                points.append((outer * 16, 100 * statistics.mean(r["accuracy"] for r in group)))
            ax.plot(*zip(*points, strict=True), color=colors[run], marker="D" if run == "E82" else "o")
        ax.set(title=benchmark, xlabel="Optimizer updates", ylabel="Mean accuracy (%)", xlim=(0, 1680))
        fv.apply_frame(ax, theme.frame)
        fv.apply_grid(ax, theme.grid)
        ax.margins(y=0.15)
    finish(fig, "validation-accuracy", "All AMC/AIME evaluations use n16; Macro-5 gives each benchmark equal weight")

    final = {
        run: [r for r in rows if r["run"] == run and r["outer"] == 100 and r["benchmark"].startswith("AIME")]
        for run in colors
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8))
    for run in colors:
        axes[0].plot(
            range(3),
            [r["mean_length"] for r in sorted(final[run], key=lambda r: r["benchmark"])],
            marker="D" if run == "E82" else "o",
            color=colors[run],
        )
        for j, metric in enumerate(("length_rate", "repetition_rate", "stop_rate")):
            offset = -0.16 if run == "E82" else 0.16
            axes[1].bar(j + offset, 100 * statistics.mean(r[metric] for r in final[run]), width=0.28, color=colors[run])
    axes[0].set(
        title="AIME response length at update 1600",
        xticks=range(3),
        xticklabels=["2024", "2025", "2026"],
        xlabel="AIME year",
        ylabel="Mean response tokens",
        ylim=(0, 15000),
    )
    axes[1].set(
        title="AIME stopping at update 1600",
        xticks=range(3),
        xticklabels=["Length limit", "Repetition", "EOS / stop"],
        xlabel="Engine finish category",
        ylabel="Responses (%)",
        ylim=(0, 100),
    )
    for ax in axes:
        fv.apply_frame(ax, theme.frame)
        fv.apply_grid(ax, theme.grid)
    finish(fig, "length-stopping", "Both runs: 30 prompts per AIME year, 16 responses each; response cap 31,744")

    # The published points remain figure readouts under their original protocol.
    comparison = [
        r for r in csv.DictReader(REFERENCE.open()) if r["run"] in ("E57", "E82", "TOP-D alpha=0.1", "TOP-D alpha=0.2")
    ]
    comparison += list(csv.DictReader((DATA / "validation.csv").open()))
    series = {
        "E57": (palette[0], "o", "-", "E57 | immediate b64"),
        "E82": (colors["E82"], "D", "-", "E82 | immediate b256"),
        "E84": (colors["E84"], "s", "-", "E84 | future mean b256"),
        "TOP-D alpha=0.1": (palette[2], "", "--", "TOP-D alpha 0.1 | paper"),
        "TOP-D alpha=0.2": (palette[2], "", ":", "TOP-D alpha 0.2 | paper"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.7), sharey=True)
    for ax, bench in zip(axes, ("AIME24", "AIME25", "AIME26"), strict=True):
        for run, (color, marker, ls, label) in series.items():
            points = sorted(
                (float(r["optimizer_update"]), 100 * float(r["accuracy"]))
                for r in comparison
                if r["run"] == run and r["benchmark"] == bench and float(r["optimizer_update"]) <= 1600
            )
            ax.plot(*zip(*points, strict=True), color=color, marker=marker, linestyle=ls)
        ax.set(title=bench, xlabel="Optimizer updates", xlim=(0, 1680), ylim=(0, 18))
        fv.apply_frame(ax, theme.frame)
        fv.apply_grid(ax, theme.grid)
    axes[0].set_ylabel("Mean accuracy (%)")
    handles = [
        Line2D([], [], color=color, marker=marker, linestyle=ls, label=label)
        for color, marker, ls, label in series.values()
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.055), ncol=3, frameon=False)
    fig.text(
        0.5,
        0.017,
        "Local avg@16 vs paper avg@32; protocols differ; all points observed, no interpolation",
        ha="center",
        fontsize="small",
    )
    fig.tight_layout(rect=(0, 0.25, 1, 1))
    save(fig, OUT, "aime-paper-updates", fv)

    training = [
        {k: float(v) if k != "run" and v != "" else v for k, v in r.items()}
        for r in csv.DictReader((DATA / "training-outers.csv").open())
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0))
    for ax, key, title, ylabel in zip(
        axes.flat,
        ("train_length", "reward", "actor_loss", "grad_norm"),
        ("Training response length", "Training rollout score", "E84 training surrogate", "Preclip gradient norm"),
        ("Tokens", "Mean score", "Signed outer summary", "Norm"),
        strict=True,
    ):
        ax.plot([r["optimizer_update"] for r in training], [r[key] for r in training], color=colors["E84"])
        ax.set(title=title, xlabel="Optimizer updates", ylabel=ylabel)
        fv.apply_frame(ax, theme.frame)
        fv.apply_grid(ax, theme.grid)
    fig.text(
        0.5,
        0.025,
        "E84 console outer summaries; signed surrogate includes the future score term",
        ha="center",
        fontsize="small",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, OUT, "training", fv, independent=True)
    dump(
        OUT / "provenance.json",
        dict(
            sources=[
                source(path, path.read_bytes())
                for path in (REFERENCE, DATA / "validation.csv", DATA / "summary.json", Path(__file__))
            ],
            axis="optimizer updates",
            e84_earlier_stop_fractions="not present in console; only endpoint plotted",
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Preserve original attachment before extraction")
    args = parser.parse_args()
    if args.input:
        RAW.parent.mkdir(parents=True, exist_ok=True)
        if (
            RAW.exists()
            and hashlib.sha256(RAW.read_bytes()).digest() != hashlib.sha256(args.input.read_bytes()).digest()
        ):
            raise ValueError("Refusing to overwrite different frozen input")
        if not RAW.exists():
            shutil.copyfile(args.input, RAW)
    extract()
    plot()


if __name__ == "__main__":
    main()
