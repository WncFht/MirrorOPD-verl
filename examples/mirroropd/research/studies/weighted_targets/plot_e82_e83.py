# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Audit and plot E82/E83 alone; default reads frozen CSV, --refresh reads logs.

Semantic ledger: color and marker identify the run across all panels. Actual
actor loss uses each run's reduction; MDPO objective and teacher KL use logged
token means on each run's own changing prefixes, not a common fixed bank.
Faint optimization curves are individual updates; bold curves are arithmetic
means of exactly 16 updates in one outer, located at the update midpoint.
Grad-norm exceedance is the share of preclip norms > 1, not a PPO clip metric.
Validation markers are complete audited avg@16 nodes; macro weights benchmarks
equally. Training reward is a changing DAPO batch readout, not validation acc.
Geometry ledger: x = optimizer updates, 16/outer, 256 responses/update in both
runs; no interpolation, artificial step0, extrapolation or confidence bands.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from examples.mirroropd.research.studies.weighted_targets.metrics import trusted_metrics
from examples.mirroropd.research.studies.weighted_targets.plot_e81_validation import ROOT, source, style
from examples.mirroropd.research.studies.weighted_targets.plot_validation_comparison import extract_live, save

OUT = ROOT / "examples/mirroropd/docs/experiments/learning-signal/figures/e82-e83-comparison-20261107"
RUNS = {
    "E82": ROOT / "runs/topd-large-batch-20261105/formal/E82",
    "E83": ROOT / "runs/topd-learning-signal-20261106/formal/E83",
}
UPDATE_KEYS = {
    "actor_loss": "optim/actor/loss",
    "token_objective": "distill/mdpo/objective",
    "teacher_kl": "distill/teacher_kl",
    "grad_norm": "optim/actor/grad_norm_preclip",
}


def write_csv(path, rows):
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def refresh(out):
    sources = []
    for name in ("plot_e82_e83.py", "plot_validation_comparison.py", "metrics.py", "plot_e81_validation.py"):
        path = Path(__file__).with_name(name)
        sources.append(source(path, path.read_bytes()))
    for name in ("output.py", "losses.py"):
        path = ROOT / "runs/topd-large-batch-20261105/source/verl/trainer/distillation" / name
        sources.append(source(path, path.read_bytes()))
    valid, updates, outers, audits = [], [], [], {}
    for run, root in RUNS.items():
        rows, audit, _ = extract_live(run, sources)
        valid.extend(rows)
        path = root / "metrics.jsonl"
        original = path.read_bytes()
        raw, _ = trusted_metrics(path, original)
        sources.append(source(path, original))
        data = [json.loads(line)["data"] for line in raw.splitlines()]
        uu = [r for r in data if "optim/actor/update_applied" in r]
        # Bound all training series to the update count of the audited valid read.
        uu = [r for r in uu if r["step/optimizer"] <= audit["optimizer_updates"]]
        assert [r["step/optimizer"] for r in uu] == list(range(1, audit["optimizer_updates"] + 1))
        assert audit["update_applied"] == audit["optimizer_updates"]
        assert all(audit[k] == 0 for k in ("update_skipped", "loss_nonfinite", "grad_norm_preclip_nonfinite"))
        for r in uu:
            point = {"run": run, "outer": int(r["step/outer"]), "optimizer_update": int(r["step/optimizer"])}
            point.update({key: float(r[value]) for key, value in UPDATE_KEYS.items()})
            assert all(math.isfinite(point[key]) for key in UPDATE_KEYS)
            updates.append(point)
        groups = defaultdict(list)
        for r in updates:
            if r["run"] == run:
                groups[r["outer"]].append(r)
        summaries = {int(r["step/outer"]): r for r in data if "rollout/reward/mean" in r}
        for outer, group in sorted(groups.items()):
            if len(group) != 16 or outer not in summaries:
                continue
            record = summaries[outer]
            assert [r["optimizer_update"] for r in group] == list(range(outer * 16 - 15, outer * 16 + 1))
            point = {
                "run": run,
                "outer": outer,
                "optimizer_update": outer * 16,
                "update_midpoint": statistics.mean(r["optimizer_update"] for r in group),
                "train_length": record["rollout/response_length/mean"],
                "train_reward": record["rollout/reward/mean"],
                "norm_above_one": statistics.mean(r["grad_norm"] > 1 for r in group),
            }
            for key in UPDATE_KEYS:
                point[key] = statistics.mean(r[key] for r in group)
                point[key + "_first"] = group[0][key]
                point[key + "_last"] = group[-1][key]
            outers.append(point)
        audit["complete_training_outers"] = sum(r["run"] == run for r in outers)
        audits[run] = audit
    for name, rows in (("validation", valid), ("updates", updates), ("outers", outers)):
        write_csv(out / f"{name}.csv", rows)
    (out / "source-audit.json").write_text(json.dumps(audits, indent=2) + "\n")
    provenance = {
        "snapshot_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": sources,
        "settings": {"E82": "future=none, token-mean", "E83": "future=none, seq-mean-token-mean"},
        "loss": "Actor loss has run-specific aggregation; token_objective and teacher_kl are logged token means.",
        "summary": "Bold optimization lines are means of 16 updates per outer at update midpoint; faint lines are raw.",
        "limits": "Different on-policy prefixes, one run per condition; no fixed-bank or cross-seed causal ranking.",
    }
    (out / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")


def validation_series(rows, run, bench, key):
    groups = defaultdict(list)
    names = {"AIME24", "AIME25", "AIME26"} if bench == "AIME mean" else None
    for row in rows:
        selected = bench == "Macro-5" or row["benchmark"] == bench or names and row["benchmark"] in names
        if row["run"] == run and selected:
            groups[int(row["optimizer_update"])].append(float(row[key]))
    expected = 5 if bench == "Macro-5" else 3 if names else 1
    assert all(len(values) == expected for values in groups.values())
    return [(update, statistics.mean(values)) for update, values in sorted(groups.items())]


def plot(out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fv = style()
    theme = fv.apply_rc("paper")
    palette = fv.series_colors("paper", 9)
    colors, markers = {"E82": palette[0], "E83": palette[1]}, {"E82": "D", "E83": "P"}
    valid, updates, outers = [read_csv(out / f"{name}.csv") for name in ("validation", "updates", "outers")]
    max_update = max(int(r["optimizer_update"]) for r in updates)
    handles = [
        Line2D([], [], color=colors[run], marker=markers[run], label=label)
        for run, label in (("E82", "E82 | token-mean"), ("E83", "E83 | seq-mean-token-mean"))
    ]

    def canvas():
        return plt.subplots(2, 3, figsize=(12.6, 7.8))

    def frame(ax, title, ylabel):
        ax.set(title=title, xlabel="Optimizer updates", ylabel=ylabel, xlim=(0, max_update * 1.035))
        fv.apply_frame(ax, theme.frame)
        fv.apply_grid(ax, theme.grid)

    def finish(fig, stem, caption):
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.062), ncol=2, frameon=False)
        fig.text(0.5, 0.018, caption, ha="center", color=fv.NEUTRALS["baseline"], fontsize="small")
        fig.tight_layout(rect=(0, 0.14, 1, 1))
        save(fig, out, stem, fv, independent=True)

    fig, axes = canvas()
    for ax, bench in zip(axes.flat, ("Macro-5", "MATH-500", "AMC23", "AIME24", "AIME25", "AIME26"), strict=True):
        for run in RUNS:
            points = validation_series(valid, run, bench, "accuracy")
            ax.plot([p[0] for p in points], [100 * p[1] for p in points], color=colors[run], marker=markers[run])
        frame(ax, bench, "Validation accuracy (%)")
        top = max(max(line.get_ydata()) for line in ax.lines)
        ax.set_ylim(0, top * 1.12)
    finish(fig, "accuracy", "Complete avg@16 only; Macro-5 weights benchmarks equally; no local step0 or extrapolation")

    fig, axes = canvas()
    specs = (
        ("actor_loss", "Actor loss | run-specific reduction", "Loss"),
        ("token_objective", "MDPO objective | token mean", "Objective"),
        ("teacher_kl", "Teacher KL | token mean", "KL"),
        ("grad_norm", "Preclip gradient norm", "Norm"),
        ("norm_above_one", "Updates with preclip norm > 1", "Share per outer (%)"),
        ("train_reward", "Training rollout correctness", "DAPO batch accuracy (%)"),
    )
    for ax, (key, title, ylabel) in zip(axes.flat, specs, strict=True):
        for run in RUNS:
            rr = [r for r in outers if r["run"] == run]
            factor = 100 if key in ("norm_above_one", "train_reward") else 1
            if key in UPDATE_KEYS:
                raw = [r for r in updates if r["run"] == run]
                ax.plot(
                    [int(r["optimizer_update"]) for r in raw],
                    [float(r[key]) for r in raw],
                    color=colors[run],
                    alpha=0.22,
                    lw=0.6,
                )
            xkey = "update_midpoint" if key in UPDATE_KEYS or key == "norm_above_one" else "optimizer_update"
            ax.plot([float(r[xkey]) for r in rr], [factor * float(r[key]) for r in rr], color=colors[run], lw=1.6)
        frame(ax, title, ylabel)
        if key == "grad_norm":
            ax.axhline(1, color=fv.NEUTRALS["baseline"], linestyle=":", lw=0.9)
        ax.set_ylim(bottom=0, top=105 if key == "norm_above_one" else None)
    finish(
        fig,
        "optimization",
        "Faint: raw updates; bold: 16-update outer means. Loss reductions and sampled prefixes differ.\n"
        "Norm > 1 is threshold exceedance; dotted norm line = 1. Training correctness is not held-out validation.",
    )

    fig, axes = canvas()
    specs = (
        ("train", "Training response length", "Mean tokens", None),
        ("Macro-5", "Validation length | Macro-5", "Mean tokens", "mean_length"),
        ("AIME mean", "Validation length | AIME mean", "Mean tokens", "mean_length"),
        ("AIME mean", "AIME length-limit stops", "Responses (%)", "length_rate"),
        ("AIME mean", "AIME repetition stops", "Responses (%)", "repetition_rate"),
        ("AIME mean", "AIME EOS / stop", "Responses (%)", "stop_rate"),
    )
    for ax, (bench, title, ylabel, key) in zip(axes.flat, specs, strict=True):
        for run in RUNS:
            if bench == "train":
                points = [(int(r["optimizer_update"]), float(r["train_length"])) for r in outers if r["run"] == run]
            else:
                points = validation_series(valid, run, bench, key)
            factor = 100 if key and key.endswith("rate") else 1
            ax.plot(
                [p[0] for p in points],
                [factor * p[1] for p in points],
                color=colors[run],
                marker=markers[run],
                ms=3.4,
            )
        frame(ax, title, ylabel)
        ax.set_ylim(bottom=0, top=100 if key and key.endswith("rate") else None)
    finish(
        fig,
        "length-stopping",
        "Train cap 16,384; valid cap 31,744. Benchmark means are equally weighted.\n"
        "EOS / stop is an engine finish category, not a guarantee of coherent reasoning.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.refresh:
        refresh(args.out)
    plot(args.out)


if __name__ == "__main__":
    main()
