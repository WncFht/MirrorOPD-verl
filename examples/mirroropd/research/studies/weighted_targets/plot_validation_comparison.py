# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Compare frozen local validations and published TOP-D accuracy curves.

Semantic ledger: local markers are complete benchmark avg@16 evaluations;
paper markers are Figure 5 vector vertices (avg@32), not recovered run logs.
Color identifies a run across all figures. E55/E56/E57 did not enable repetition
stopping: their repetition_rate is missing, never zero. EOS/stop is an engine
finish category, not evidence of a coherent answer. Macro-5 and AIME means give
each benchmark equal weight. TOP-D has no length/finish data in these sources.
Geometry ledger: local runs use 16 updates/outer and batch64 (E55--57) or
batch256 (E81--83). TOP-D alpha0.1/0.2 use 16 updates and 4096 presentations/outer.
Accuracy panels use labelled independent benchmark y ranges; paper panels share
0--25 percent. Full and early views contain only observed points within each
window: no interpolated boundary observations, smoothing or extrapolation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from examples.mirroropd.research.studies.weighted_targets.metrics import trusted_metrics
from examples.mirroropd.research.studies.weighted_targets.plot_e81_validation import (
    BENCHMARKS,
    ROOT,
    SLUGS,
    source,
    style,
)

BASE = ROOT / "examples/mirroropd/docs/experiments/learning-signal"
DEFAULT_OUT = BASE / "figures/validation-comparison-20261106"
RUNS = ROOT / "runs/topd-large-batch-20261105/formal"
LOCAL = ("E55", "E56", "E57", "E81", "E82", "E83")
REFERENCE = ("Base reference", "Qwen3-30B teacher")
ACCURACY_RUNS = (*LOCAL, *REFERENCE)
PAPER = ("TOP-D alpha=0.1", "TOP-D alpha=0.2")
FIELDS = (
    "run",
    "benchmark",
    "outer",
    "optimizer_update",
    "response_presentations",
    "fresh_responses",
    "responses",
    "accuracy",
    "mean_length",
    "length_rate",
    "repetition_rate",
    "stop_rate",
)


def clock(run, outer):
    batch = 256 if run in ("E81", "E82", "E83") or run in PAPER else 64
    return dict(
        run=run,
        outer=outer,
        optimizer_update=outer * 16,
        response_presentations=outer * 16 * batch,
        fresh_responses=outer * (4096 if run in PAPER else batch),
    )


def reference_rows():
    """Independent x=0 readouts; they are never treated as training events."""
    base = {
        "MATH-500": 0.185875,
        "AMC23": 0.1140625,
        "AIME24": 0.025,
        "AIME25": 0.004166666666666667,
        "AIME26": 0.008333333333333333,
    }
    teacher = {
        "MATH-500": 0.9685,
        "AMC23": 0.9578,
        "AIME24": 0.7479,
        "AIME25": 0.6333,
        "AIME26": 0.7542,
    }
    rows = []
    for run, values in ((REFERENCE[0], base), (REFERENCE[1], teacher)):
        for benchmark, accuracy in values.items():
            rows.append(
                {
                    **clock(run, 0),
                    "benchmark": benchmark,
                    "responses": 0,
                    "accuracy": accuracy,
                    "mean_length": "",
                    "length_rate": "",
                    "repetition_rate": "",
                    "stop_rate": "",
                }
            )
    return rows


def reference_value(rows, run, benchmark):
    names = (
        set(BENCHMARKS)
        if benchmark == "Macro-5"
        else {"AIME24", "AIME25", "AIME26"}
        if benchmark == "AIME mean"
        else {benchmark}
    )
    values = [r["accuracy"] for r in rows if r["run"] == run and r["benchmark"] in names]
    if benchmark in ("Macro-5", "AIME mean"):
        return statistics.mean(values)
    if len(values) != 1:
        raise ValueError(f"Expected one {run} reference for {benchmark}")
    return values[0]


def external_source(path):
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def extract_live(run, sources):
    root = RUNS / run if run != "E83" else ROOT / "runs/topd-learning-signal-20261106/formal/E83"
    path = root / "metrics.jsonl"
    original = path.read_bytes()
    raw, exclusions = trusted_metrics(path, original)
    sources.append(source(path, original))
    if exclusions is not None:
        sources.append(source(path.with_name("metrics-exclusions.json"), exclusions))
    metrics = [json.loads(line)["data"] for line in raw.splitlines()]
    updates = [r["step/optimizer"] for r in metrics if "optim/actor/update_applied" in r]
    if updates != list(range(1, len(updates) + 1)):
        raise ValueError(f"Ambiguous optimizer history in {run}; audit duplicate runs before refreshing")
    complete = {int(r["step/outer"]): r for r in metrics if "eval/math500/acc/mean" in r}
    rows = []
    for outer, logged in sorted(complete.items()):
        path = root / f"validation/records/{outer}.jsonl"
        raw = path.read_bytes()
        sources.append(source(path, raw))
        groups = defaultdict(list)
        for line in raw.splitlines():
            r = json.loads(line)
            assert r["status"] == "ok" and int(r["step"]) == outer
            assert isinstance(r["acc"], bool | int | float) and math.isfinite(r["acc"])
            groups[r["data_source"]].append(r)
        assert set(groups) == set(BENCHMARKS)
        for bench, count in BENCHMARKS.items():
            records = groups[bench]
            identities = defaultdict(list)
            for r in records:
                identities[r["prompt_id"]].append(int(r["sample_index"]))
            assert len(records) == count * 16
            assert set(identities) == {f"{bench}:{i}" for i in range(count)}
            assert all(sorted(indices) == list(range(16)) for indices in identities.values())
            finish = Counter(r["finish_reason"] for r in records)
            assert set(finish) <= {"stop", "length", "repetition"}
            acc = statistics.mean(float(r["acc"]) for r in records)
            length = statistics.mean(r["response_length"] for r in records)
            prefix = f"eval/{SLUGS[bench]}/"
            assert math.isclose(acc, logged[prefix + "acc/mean"], abs_tol=1e-12)
            assert math.isclose(length, logged[prefix + "generation/response_length/mean"], abs_tol=1e-9)
            assert logged[prefix + "failed_count"] == 0
            for reason in ("stop", "length", "repetition"):
                assert finish[reason] == logged[prefix + f"generation/finish_reason/{reason}_count"]
            rows.append(
                {
                    **clock(run, outer),
                    "benchmark": bench,
                    "responses": len(records),
                    "accuracy": acc,
                    "mean_length": length,
                    **{f"{reason}_rate": finish[reason] / len(records) for reason in ("length", "repetition", "stop")},
                }
            )
    updates = [r for r in metrics if "optim/actor/update_applied" in r]
    assert [r["step/update"] for r in updates] == list(range(1, len(updates) + 1))
    audit = {
        "complete_outers": sorted(complete),
        "checked_responses": len(complete) * 10080,
        "optimizer_updates": len(updates),
    }
    for key in ("update_applied", "update_skipped", "loss_nonfinite", "grad_norm_preclip_nonfinite"):
        audit[key] = sum(r["optim/actor/" + key] for r in updates)
    runtime = [
        {
            "run": run,
            "outer": outer,
            "optimizer_update": record["step/update"],
            "train_seconds": record["perf/timing/outer_step_seconds"],
            "validation_seconds": record["perf/timing/validation_seconds"],
            "actor_seconds": record["perf/timing/update_actor_seconds"],
            "rollout_seconds": record["perf/timing/rollout_pipeline_seconds"],
            "train_response_length": record["rollout/response_length/mean"],
        }
        for outer, record in sorted(complete.items())
    ]
    return rows, audit, runtime


def refresh(out):
    sources = []
    for path in (Path(__file__).resolve(), Path(__file__).with_name("plot_e81_validation.py")):
        sources.append(source(path, path.read_bytes()))
    path = BASE / "data/e55-e57-20261103/summary.json"
    raw = path.read_bytes()
    sources.append(source(path, raw))
    rows = []
    for run, summary in json.loads(raw).items():
        assert run in LOCAL and summary["source_status"] == "finished" and summary["exit_code"] == 0
        for outer, validation in summary["validation"].items():
            for bench, count in BENCHMARKS.items():
                r = validation["datasets"][SLUGS[bench]]
                assert r["responses"] == count * 16
                assert r["cap_count"] + r["stop_count"] == r["responses"]
                rows.append(
                    {
                        **clock(run, int(outer)),
                        "benchmark": bench,
                        "responses": r["responses"],
                        "accuracy": r["accuracy"],
                        "mean_length": r["length"],
                        "length_rate": r["cap_rate"],
                        "repetition_rate": "",
                        "stop_rate": r["stop_count"] / r["responses"],
                    }
                )
    audit = {}
    runtime = []
    for run in ("E81", "E82", "E83"):
        live, audit[run], timing = extract_live(run, sources)
        rows.extend(live)
        if run != "E81":
            runtime.extend(timing)
    path = BASE / "data/topd-e57-all-curves-20261104/curves.csv"
    raw = path.read_bytes()
    sources.append(source(path, raw))
    paper_rows = list(csv.DictReader(raw.decode().splitlines()))
    for r in paper_rows:
        if r["series"] == "E57":
            match = next(
                v
                for v in rows
                if v["run"] == "E57" and v["benchmark"] == r["benchmark"] and v["outer"] == int(r["outer"])
            )
            assert math.isclose(match["accuracy"] * 100, float(r["accuracy_percent"]), abs_tol=1e-10)
        elif r["series"] in PAPER:
            c = clock(r["series"], int(r["outer"]))
            assert c["optimizer_update"] == int(r["optimizer_updates"])
            assert c["response_presentations"] == int(r["response_presentations"])
            rows.append({**c, "benchmark": r["benchmark"], "accuracy": float(r["accuracy_percent"]) / 100})
    rows.extend(reference_rows())
    for name in ("provenance.json",):
        path = BASE / "data/topd-e57-all-curves-20261104" / name
        sources.append(source(path, path.read_bytes()))
    for run in PAPER:
        for bench in ("AIME24", "AIME25", "AIME26"):
            assert [r["outer"] for r in rows if r["run"] == run and r["benchmark"] == bench] == list(range(0, 411, 10))
    with (out / "validation.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with (out / "runtime.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "run",
                "outer",
                "optimizer_update",
                "train_seconds",
                "validation_seconds",
                "actor_seconds",
                "rollout_seconds",
                "train_response_length",
            ),
        )
        writer.writeheader()
        writer.writerows(runtime)
    (out / "source-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    provenance = {
        "snapshot_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": sources
        + [
            source(
                BASE / "data/topd-e57-step0-20261104/baseline.json",
                (BASE / "data/topd-e57-step0-20261104/baseline.json").read_bytes(),
            ),
            external_source(ROOT.parent / "wandb-logs/derived/MirrorOPD-verl/run-20261026_223012-f0e44ec7/config.json"),
            source(
                ROOT / "examples/mirroropd/docs/experiments/teacher-transfer/appendix/qwen30b-reference.md",
                (
                    ROOT / "examples/mirroropd/docs/experiments/teacher-transfer/appendix/qwen30b-reference.md"
                ).read_bytes(),
            ),
        ],
        "complete_validations": {k: v["complete_outers"] for k, v in audit.items()},
        "local_metric": "benchmark avg@16; macro5/AIME mean are unweighted benchmark means",
        "paper_metric": "Figure5 vector vertices avg@32; published smoothing, if any, is retained",
        "missing": "Repetition stopping disabled in E55--57: blank, not zero. Paper length/finish metrics unavailable.",
        "clocks": "16 updates/outer; local batch64(E55--57)/256(E81--83); TOP-D4096 presentations/outer",
        "limitations": [
            "Paper/local prompts, decoding, cap, scoring and update algorithms differ; no causal ranking",
            "E55--57 use historical numerical execution and no repetition stopping; E81--83 enable detector",
            "E81/82 share a nominal seed/prefix setting, not independent replications",
            "Only complete live validations with matching metrics are admitted; "
            "independent Base/teacher references are explicit x=0 points",
            "No interpolation for equal-budget tables; plotted lines only connect real observations",
            "Default invocation reads frozen CSV; --refresh explicitly refreshes the live snapshot",
        ],
    }
    (out / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")


def load_rows(out):
    with (out / "validation.csv").open() as stream:
        return [
            {k: v if k in ("run", "benchmark") or v == "" else float(v) for k, v in r.items()}
            for r in csv.DictReader(stream)
        ]


def series(rows, run, bench, metric="accuracy"):
    groups = defaultdict(list)
    expected = 5 if bench == "Macro-5" else 3 if bench == "AIME mean" else 1
    for r in rows:
        if r["run"] != run:
            continue
        if (
            bench == "Macro-5"
            or r["benchmark"] == bench
            or (bench == "AIME mean" and r["benchmark"].startswith("AIME"))
        ):
            groups[r["outer"]].append(r)
    points = []
    for outer, values in sorted(groups.items()):
        if len(values) == expected and all(r.get(metric, "") != "" for r in values):
            points.append({**clock(run, int(outer)), metric: statistics.mean(r[metric] for r in values)})
    return points


def save(fig, out, stem, fv, independent=False):
    import matplotlib.pyplot as plt

    fig.canvas.draw()
    scoped = [("facet-ylim-mismatch", ax) for ax in fig.axes] if independent else []
    problems = fv.audit(fig, scoped_ignore=scoped, print_slot=fig.get_figwidth())
    (out / f"{stem}-audit.json").write_text(json.dumps(problems, ensure_ascii=False, indent=2) + "\n")
    if any(fv.problem_severity(p) == "error" for p in problems):
        raise RuntimeError(problems)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{ext}", dpi=180)
    plt.close(fig)


def plot(rows, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fv = style()
    theme = fv.apply_rc("paper")
    # Reuse the same run colors throughout the figure set.
    palette = fv.series_colors("paper", 9)
    colors = {
        "E55": fv.NEUTRALS["baseline"],
        "E56": fv.NEUTRALS["baseline"],
        "E57": palette[0],
        "E81": palette[1],
        "E82": palette[4],
        "E83": palette[4],
        PAPER[0]: palette[2],
        PAPER[1]: palette[2],
        REFERENCE[0]: fv.NEUTRALS["baseline"],
        REFERENCE[1]: palette[5],
    }
    markers = dict(zip(LOCAL, ("v", "^", "o", "s", "D", "P"), strict=True)) | {
        PAPER[0]: "",
        PAPER[1]: "",
        REFERENCE[0]: "D",
        REFERENCE[1]: "*",
    }

    def draw(ax, run, bench, key="optimizer_update", metric="accuracy", factor=100, limit=None):
        points = series(rows, run, bench, metric)
        if run in REFERENCE and metric == "accuracy":
            ax.axhline(
                factor * reference_value(rows, run, bench),
                label=run,
                color=colors[run],
                marker=markers[run],
                lw=1.4,
                linestyle="--" if run == REFERENCE[1] else "-.",
            )
            return
        if limit is not None:
            points = [p for p in points if p[key] <= limit]
        if not points:
            return
        ax.plot(
            [p[key] for p in points],
            [factor * p[metric] for p in points],
            label=run,
            color=colors[run],
            marker=markers[run],
            ms=3.4,
            lw=1.4,
            linestyle="--" if run == PAPER[0] else ":" if run == PAPER[1] else "-",
        )

    def frame(ax):
        fv.apply_frame(ax, theme.frame)
        fv.apply_grid(ax, theme.grid)

    def finish(fig, runs, notes, columns=5):
        handles = [
            Line2D(
                [],
                [],
                label=r,
                color=colors[r],
                marker=markers[r],
                ms=4,
                linestyle=(
                    "--"
                    if r in (PAPER[0], REFERENCE[1])
                    else ":"
                    if r == PAPER[1]
                    else "-."
                    if r == REFERENCE[0]
                    else "-"
                ),
            )
            for r in runs
        ]
        fig.legend(handles=handles, ncol=columns, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.055))
        fig.text(0.5, 0.012, notes, ha="center", fontsize=10.5, color=fv.NEUTRALS["baseline"])
        fig.tight_layout(rect=(0, 0.15, 1, 1))

    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.5))
    for ax, bench in zip(axes.flat, (*BENCHMARKS, "Macro-5"), strict=True):
        for run in ACCURACY_RUNS:
            draw(ax, run, bench)
        ax.set(
            title=bench,
            xlabel="Optimizer updates",
            ylabel="Accuracy (%)",
            xlim=(0, 1640),
            ylim=(0, 80 if bench.startswith("AIME") else 100),
        )
        frame(ax)
    finish(
        fig,
        ACCURACY_RUNS,
        "Training curves are avg@16; Base x=0 combines E39 and TOP-D base/local; teacher is independent avg@16",
        columns=4,
    )
    save(fig, out, "local-accuracy-optimizer-updates", fv, independent=True)

    for bench, stem in (
        ("Macro-5", "behavior-macro5-optimizer-updates"),
        ("AIME mean", "behavior-aime-optimizer-updates"),
    ):
        fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.5))
        for ax, (metric, label, factor, upper) in zip(
            axes.flat,
            (
                ("mean_length", "Mean response length (tokens)", 1, 33000),
                ("length_rate", "Length-limit stops (%)", 100, 100),
                ("repetition_rate", "Repetition stops (%)", 100, 100),
                ("stop_rate", "EOS / stop (%)", 100, 100),
            ),
            strict=True,
        ):
            for run in LOCAL:
                draw(ax, run, bench, metric=metric, factor=factor)
            ax.set(xlabel="Optimizer updates", ylabel=label, xlim=(0, 1640), ylim=(0, upper))
            if metric == "repetition_rate":
                ax.set_title("Detector off: E55 / E56 / E57 = N/A", fontsize=11)
            frame(ax)
        fig.suptitle(f"{bench}: equal benchmark means", y=0.99)
        finish(
            fig,
            LOCAL,
            "Validation cap = 31,744 tokens; missing repetition data is not zero; TOP-D behavior unavailable",
        )
        save(fig, out, stem, fv)

    runs = ("E57", "E81", "E82", "E83", *PAPER, REFERENCE[0])
    for key, label, stem, full, early, divisor in (
        ("optimizer_update", "Optimizer updates", "aime-optimizer-updates", 6720, 1600, 1),
        (
            "response_presentations",
            "Training response presentations (thousands)",
            "aime-response-presentations",
            1720320,
            102400,
            1000,
        ),
    ):
        fig, axes = plt.subplots(2, 3, figsize=(11.6, 6.9))
        for row, limit in enumerate((full, early)):
            for col, bench in enumerate(("AIME24", "AIME25", "AIME26")):
                ax = axes[row, col]
                for run in runs:
                    if run in REFERENCE:
                        ax.axhline(
                            reference_value(rows, run, bench) * 100,
                            label=run,
                            color=colors[run],
                            lw=1.4,
                            linestyle="--" if run == REFERENCE[1] else "-.",
                        )
                        continue
                    points = [p for p in series(rows, run, bench) if p[key] <= limit]
                    if points:
                        ax.plot(
                            [p[key] / divisor for p in points],
                            [p["accuracy"] * 100 for p in points],
                            label=run,
                            color=colors[run],
                            marker=markers[run],
                            ms=3.4,
                            lw=1.4,
                            linestyle="--" if run == PAPER[0] else ":" if run == PAPER[1] else "-",
                        )
                ax.set(
                    title=f"{bench} | {'full' if row == 0 else 'early'}",
                    xlabel=label,
                    ylabel="Accuracy (%)",
                    xlim=(-limit * 0.015 / divisor, limit * 1.025 / divisor),
                    ylim=(0, 25),
                )
                ax.set_xticks(
                    [0, 1600, 3200, 4800, 6400]
                    if key == "optimizer_update" and row == 0
                    else [0, 400, 800, 1200, 1600]
                    if key == "optimizer_update"
                    else [0, 400, 800, 1200, 1600]
                    if row == 0
                    else [0, 25, 50, 75, 100]
                )
                frame(ax)
        finish(
            fig,
            runs,
            "Local avg@16 / paper avg@32 vector readouts; protocols differ; no invented local step0\n"
            "Teacher omitted above 25%; see full-range accuracy panels",
            columns=5,
        )
        save(fig, out, stem, fv)

    with (out / "runtime.csv").open() as stream:
        runtime = list(csv.DictReader(stream))
    if runtime:
        fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), sharey=True)
        for ax, run in zip(axes, ("E82", "E83"), strict=True):
            points = [r for r in runtime if r["run"] == run]
            xs = [int(r["outer"]) for r in points]
            train = [float(r["train_seconds"]) / 60 for r in points]
            valid = [float(r["validation_seconds"]) / 60 for r in points]
            width = 1.7
            ax.bar(xs, train, width=width, label="Training", color=palette[4])
            ax.bar(xs, valid, width=width, bottom=train, label="Validation", color=fv.NEUTRALS["baseline"])
            ax.set(title=run, xlabel="Completed outer", xticks=xs)
            frame(ax)
        axes[0].set_ylabel("Minutes per outer and validation event")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.035))
        fig.tight_layout(rect=(0, 0.12, 1, 1))
        save(fig, out, "training-validation-runtime", fv)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.refresh:
        refresh(args.out)
    plot(load_rows(args.out), args.out)


if __name__ == "__main__":
    main()
