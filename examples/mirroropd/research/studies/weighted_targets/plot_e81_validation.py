# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Replot frozen E81/E57 validation data; --refresh audits live source records.

Semantic ledger: a marker represents benchmark mean accuracy (16 responses per
question), not pass@16. Macro-5 weights benchmarks equally. Each panel is one
benchmark or the macro; color denotes run consistently. Behavior panels show
E81 macro-5 and the equal mean of three AIME benchmarks. Length-limit stops and
repetition stops remain separate measured categories.
Geometry ledger: outer * 16 optimizer updates; response presentations are
outer * 16 * batch (E57=64, E81=256). No interpolation creates observations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
HERE = ROOT / "examples/mirroropd/docs/experiments/learning-signal/figures/e81-valid-20261105"
RUN = ROOT / "runs/topd-large-batch-20261105/formal/E81"
E57 = ROOT / "examples/mirroropd/docs/experiments/learning-signal/data/e55-e57-20261103/summary.json"
BENCHMARKS = {"MATH-500": 500, "AMC23": 40, "AIME24": 30, "AIME25": 30, "AIME26": 30}
SLUGS = {"MATH-500": "math500", "AMC23": "amc23", "AIME24": "aime24", "AIME25": "aime25", "AIME26": "aime26"}
FIELDS = [
    "run",
    "outer",
    "optimizer_update",
    "response_presentations",
    "benchmark",
    "responses",
    "accuracy",
    "mean_length",
    "length_rate",
    "repetition_rate",
    "stop_rate",
    "format_rate",
]


def style():
    for parent in ROOT.parents:
        scripts = parent / "dotfiles/.agents/skills/figure-viz/scripts"
        if (scripts / "fvlocate.py").is_file():
            sys.path.insert(0, str(scripts))
            import fvlocate

            fvlocate.ensure()
            import fvstyle

            return fvstyle
    raise RuntimeError("Set up figure-viz style assets before rendering")


def source(path, raw):
    return {"path": str(path.relative_to(ROOT)), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def clock(run, outer):
    return {
        "run": run,
        "outer": outer,
        "optimizer_update": outer * 16,
        "response_presentations": outer * 16 * (256 if run == "E81" else 64),
    }


def refresh():
    metrics_raw = (RUN / "metrics.jsonl").read_bytes()
    metrics = [json.loads(line)["data"] for line in metrics_raw.splitlines()]
    generator = Path(__file__).resolve()
    sources = [source(generator, generator.read_bytes()), source(RUN / "metrics.jsonl", metrics_raw)]
    rows = []
    for path in sorted((RUN / "validation/records").glob("*.jsonl"), key=lambda p: int(p.stem)):
        raw = path.read_bytes()
        records = [json.loads(line) for line in raw.splitlines()]
        outer = int(path.stem)
        assert len(records) == 10080
        sources.append(source(path, raw))
        groups = defaultdict(list)
        for r in records:
            assert r["status"] == "ok" and int(r["step"]) == outer
            assert isinstance(r["acc"], bool | int | float) and math.isfinite(r["acc"])
            groups[r["data_source"]].append(r)
        assert set(groups) == set(BENCHMARKS)
        logged = [r for r in metrics if r.get("step/outer") == outer and "eval/math500/acc/mean" in r]
        assert len(logged) == 1
        for bench, expected in BENCHMARKS.items():
            rs = groups[bench]
            prompts = defaultdict(list)
            for r in rs:
                prompts[r["prompt_id"]].append(int(r["sample_index"]))
            assert len(rs) == expected * 16 and len(prompts) == expected
            assert all(sorted(indices) == list(range(16)) for indices in prompts.values())
            assert set(prompts) == {f"{bench}:{i}" for i in range(expected)}
            finish = Counter(r["finish_reason"] for r in rs)
            assert set(finish) <= {"stop", "length", "repetition"}
            accuracy = statistics.mean(float(r["acc"]) for r in rs)
            length = statistics.mean(r["response_length"] for r in rs)
            prefix = f"eval/{SLUGS[bench]}/"
            assert math.isclose(accuracy, logged[0][prefix + "acc/mean"], abs_tol=1e-12)
            assert math.isclose(length, logged[0][prefix + "generation/response_length/mean"], abs_tol=1e-9)
            assert logged[0][prefix + "failed_count"] == 0
            for reason in ["stop", "length", "repetition"]:
                assert finish[reason] == logged[0][prefix + f"generation/finish_reason/{reason}_count"]
            rows.append(
                {
                    **clock("E81", outer),
                    "benchmark": bench,
                    "responses": len(rs),
                    "accuracy": accuracy,
                    "mean_length": length,
                    "length_rate": finish["length"] / len(rs),
                    "repetition_rate": finish["repetition"] / len(rs),
                    "stop_rate": finish["stop"] / len(rs),
                    "format_rate": statistics.mean(r["format_score"] for r in rs),
                }
            )
    e57_raw = E57.read_bytes()
    sources.append(source(E57, e57_raw))
    for outer, v in json.loads(e57_raw)["E57"]["validation"].items():
        for bench in BENCHMARKS:
            r = v["datasets"][SLUGS[bench]]
            rows.append(
                {
                    **clock("E57", int(outer)),
                    "benchmark": bench,
                    "responses": r["responses"],
                    "accuracy": r["accuracy"],
                    "mean_length": r["length"],
                    "length_rate": r["cap_rate"],
                    "repetition_rate": "",
                    "stop_rate": r["stop_count"] / r["responses"],
                    "format_rate": "",
                }
            )
    with (HERE / "e81-valid.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    updates = [r for r in metrics if "optim/actor/update_applied" in r]
    assert [r["step/update"] for r in updates] == list(range(1, len(updates) + 1))
    fits = defaultdict(list)
    for r in updates:
        fits[r["step/outer"]].append(r)
    audit = {
        "optimizer_updates": len(updates),
        "applied": sum(r["optim/actor/update_applied"] for r in updates),
        "skipped": sum(r["optim/actor/update_skipped"] for r in updates),
        "loss_nonfinite": sum(r["optim/actor/loss_nonfinite"] for r in updates),
        "gradient_nonfinite": sum(r["optim/actor/grad_norm_preclip_nonfinite"] for r in updates),
        "blocks": [
            {
                "outer": outer,
                "updates": len(rs),
                "objective_first": rs[0]["distill/mdpo/objective"],
                "objective_last": rs[-1]["distill/mdpo/objective"],
                "teacher_kl_first": rs[0]["distill/teacher_kl"],
                "teacher_kl_last": rs[-1]["distill/teacher_kl"],
            }
            for outer, rs in fits.items()
        ],
        "validation_checks": (
            "All benchmark accuracies, mean token lengths, finish counts and failures independently match "
            "metrics.jsonl; source/prompt/sample identities are complete."
        ),
    }
    (HERE / "source-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    provenance = {
        "snapshot_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": sources,
        "available_e81_outer": sorted({r["outer"] for r in rows if r["run"] == "E81"}),
        "metric": (
            "mean acc per prompt over samples 0..15, then benchmark mean; "
            "equal sample counts make this equal to response mean within each benchmark"
        ),
        "macro5": "unweighted mean of five benchmarks; not response-weighted (MATH-500 has 500 of 630 prompts)",
        "identity": (
            "Group records by data_source and prompt_id; never infer benchmark from JSONL row order, "
            "which follows completion order"
        ),
        "clocks": "outer*16 updates; outer*16*batch response presentations (E81 batch256, E57 batch64)",
        "limitations": [
            "E81 outer25 endpoint absent at this snapshot unless listed in available_e81_outer",
            "E57 had no repetition stopping and used a different execution path; comparisons are descriptive",
            "No run-native step0; no invented baseline or extrapolated endpoints",
            "Single training seed; no statistical significance claim",
            "CSV freezes the plotted observations; default invocation replots it, --refresh rereads live records",
        ],
    }
    (HERE / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")


def load_rows():
    with (HERE / "e81-valid.csv").open() as stream:
        return [
            {k: v if k in ["run", "benchmark"] or v == "" else float(v) for k, v in row.items()}
            for row in csv.DictReader(stream)
        ]


def series(rows, run, bench, metric="accuracy"):
    selected = [
        r
        for r in rows
        if r["run"] == run
        and (
            r["benchmark"] == bench
            if bench not in ["Macro-5", "AIME mean"]
            else (bench == "Macro-5" or r["benchmark"].startswith("AIME"))
        )
    ]
    grouped = defaultdict(list)
    for r in selected:
        grouped[r["outer"]].append(r)
    return [{**clock(run, int(o)), metric: statistics.mean(r[metric] for r in rs)} for o, rs in sorted(grouped.items())]


def save(fig, stem, fv, independent_accuracy_axes=False):
    import matplotlib.pyplot as plt

    fig.canvas.draw()
    # Independent benchmark panels need distinct, explicitly labelled y ranges
    # so the 3--10% AIME curves remain readable alongside 50--75% MATH curves.
    scoped = [("facet-ylim-mismatch", ax) for ax in fig.axes] if independent_accuracy_axes else []
    problems = fv.audit(fig, scoped_ignore=scoped)
    (HERE / f"{stem}-audit.json").write_text(json.dumps(problems, ensure_ascii=False, indent=2) + "\n")
    errors = [p for p in problems if fv.RENDER_RULES.get(p.split()[0], "error") == "error"]
    if errors:
        raise RuntimeError(errors)
    for ext in ["png", "pdf"]:
        fig.savefig(HERE / f"{stem}.{ext}", dpi=180)
    plt.close(fig)


def plot(rows):
    import matplotlib.pyplot as plt

    fv = style()
    theme = fv.apply_rc("paper")
    colors = fv.series_colors("paper", 2)
    for key, label, stem in [
        ("outer", "Training outer", "e81-valid-accuracy"),
        ("optimizer_update", "Optimizer updates", "e81-valid-optimizer-update"),
        ("response_presentations", "Training response presentations", "e81-valid-response-presentations"),
    ]:
        fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.8))
        for ax, bench in zip(axes.flat, [*BENCHMARKS, "Macro-5"], strict=True):
            for run, color, marker in zip(["E57", "E81"], colors, ["o", "s"], strict=True):
                data = series(rows, run, bench)
                ax.plot(
                    [r[key] for r in data],
                    [100 * r["accuracy"] for r in data],
                    label=run,
                    color=color,
                    marker=marker,
                    ms=3.2,
                    lw=1.5,
                )
            ax.set_title(bench)
            upper = 15 if bench.startswith("AIME") else {"MATH-500": 80, "AMC23": 60, "Macro-5": 40}[bench]
            ax.set_ylim(0, upper)
            ax.set_xlim(left=0)
            ax.set_xlabel(label)
            ax.set_ylabel("Accuracy (%)")
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 4), useMathText=True)
            fv.apply_frame(ax, theme.frame)
            fv.apply_grid(ax, theme.grid)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.035))
        fig.text(
            0.5,
            0.008,
            "Panel y-ranges differ; avg@16; E81 repetition stopping on, E57 off; no step0",
            ha="center",
            fontsize=10.5,
            color=fv.NEUTRALS["baseline"],
        )
        fig.tight_layout(rect=(0, 0.11, 1, 1))
        save(fig, stem, fv, independent_accuracy_axes=True)
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.6))
    for ax, (metric, label, factor, limit) in zip(
        axes.flat,
        [
            ("mean_length", "Mean response length (tokens)", 1, 33000),
            ("length_rate", "Length-limit stops (%)", 100, 100),
            ("repetition_rate", "Repetition stops (%)", 100, 100),
            ("stop_rate", "Natural stops (%)", 100, 100),
        ],
        strict=True,
    ):
        for bench, color, marker in zip(["Macro-5", "AIME mean"], colors, ["o", "s"], strict=True):
            data = series(rows, "E81", bench, metric)
            ax.plot(
                [r["outer"] for r in data],
                [factor * r[metric] for r in data],
                color=color,
                marker=marker,
                ms=4,
                lw=1.6,
                label=bench,
            )
        ax.set_xlabel("E81 training outer")
        ax.set_ylabel(label)
        ax.set_ylim(0, limit)
        ax.set_xticks(sorted({int(r["outer"]) for r in rows if r["run"] == "E81"}))
        fv.apply_frame(ax, theme.frame)
        fv.apply_grid(ax, theme.grid)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.035))
    fig.text(
        0.5,
        0.008,
        "Equal benchmark means; length and repetition stops are disjoint; cap = 31,744 tokens",
        ha="center",
        fontsize=10.5,
        color=fv.NEUTRALS["baseline"],
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    save(fig, "e81-valid-behavior", fv)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Refresh CSV and source audit from live records")
    args = parser.parse_args()
    if args.refresh:
        refresh()
    plot(load_rows())


if __name__ == "__main__":
    main()
