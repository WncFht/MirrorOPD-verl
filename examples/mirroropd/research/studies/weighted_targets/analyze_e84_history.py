# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Audit E84's complete local W&B history against the frozen E82 baseline.

Semantic ledger: each update is one pre-update observation, grouped into 100
fixed-trajectory reuse blocks. E82/E84 use the same adapted teacher_kl metric;
native teacher_rkl is shown separately. Future coefficients are unscaled
strict-suffix mean costs, not correctness rewards. Scalar surrogate ratios do
not measure gradient ratios. Position bins are relative response deciles.
Geometry ledger: trajectory plots use outer-block arithmetic means (16 updates),
except position profiles which pool logged sums/counts across updates/tokens.
Reuse changes compare observations before updates 1 and 16 (15 intervening
updates). No smoothing, extrapolation, or missing-value imputation is used.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import statistics as st
from pathlib import Path

from examples.mirroropd.research.studies.weighted_targets.plot_e81_validation import ROOT, style
from examples.mirroropd.research.studies.weighted_targets.plot_validation_comparison import save

BASE = ROOT / "examples/mirroropd/docs/experiments/learning-signal"
LOGS = ROOT.parent / "wandb-logs"
SOURCE = LOGS / "derived/MirrorOPD-verl/offline-run-20261109_022054-x73jzh0z"
SNAPSHOT = ROOT / "runs/wandb-e84-20261110"
OUT = BASE / "figures/e84-future-kl-20261110"
DATA = BASE / "data/e84-wandb-20261110"
REFERENCE = BASE / "figures/e82-e83-comparison-20261107/updates.csv"
FIELDS = {
    "objective": "distill/mdpo/objective",
    "target_kl": "distill/target_kl",
    "teacher_kl": "distill/teacher_kl",
    "native_teacher_rkl": "distill/teacher_rkl",
    "anchor_kl": "distill/anchor_kl",
    "coefficient": "distill/future/coefficient",
    "coefficient_std": "distill/future/coefficient_std",
    "coefficient_p95": "distill/future/coefficient_p95",
    "coefficient_max": "distill/future/coefficient_max",
    "suffix_mean_cost": "distill/future/suffix_mean_cost",
    "effective_mass": "distill/future/effective_mass",
    "score_surrogate": "distill/future/score_surrogate",
    "local_surrogate": "distill/local_surrogate",
    "training_surrogate": "distill/training_surrogate",
    "grad_norm": "optim/actor/grad_norm_preclip",
}


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def extract():
    for folder in (SNAPSHOT, DATA, OUT):
        folder.mkdir(parents=True, exist_ok=True)
    for name in ("history.jsonl", "config.json", "summary.json", "run.json"):
        shutil.copyfile(SOURCE / name, SNAPSHOT / name)
    raw = [json.loads(line) for line in (SNAPSHOT / "history.jsonl").open()]
    updates = [r for r in raw if "distill/future/coefficient" in r]
    assert len(raw) == 1700 and len(updates) == 1600
    assert [r["step/optimizer"] for r in updates] == list(range(1, 1601))
    assert all(r["optim/actor/update_applied"] == 1 for r in updates)
    assert all(r["optim/actor/update_skipped"] == r["optim/actor/loss_nonfinite"] == 0 for r in updates)
    assert all(r["optim/actor/grad_norm_preclip_nonfinite"] == 0 for r in updates)
    assert all(
        math.isclose(
            r["distill/training_surrogate"],
            r["distill/local_surrogate"] + r["distill/future/score_surrogate"],
            abs_tol=1e-7,
        )
        and math.isclose(r["distill/mdpo/objective"], r["distill/target_kl"] * 10, abs_tol=1e-7)
        for r in updates
    )
    rows = [
        dict(
            run="E84",
            outer=r["step/outer"],
            optimizer_update=r["step/optimizer"],
            reuse=r["rollout/data_reuse/update_index"],
            **{k: r[v] for k, v in FIELDS.items()},
        )
        for r in updates
    ]
    baseline = [r for r in csv.DictReader(REFERENCE.open()) if r["run"] == "E82"]
    assert len(baseline) == 1600
    baseline = [
        dict(
            run="E82",
            outer=int(r["outer"]),
            optimizer_update=int(r["optimizer_update"]),
            reuse=(int(r["optimizer_update"]) - 1) % 16 + 1,
            objective=float(r["token_objective"]),
            teacher_kl=float(r["teacher_kl"]),
            grad_norm=float(r["grad_norm"]),
        )
        for r in baseline
    ]
    outers, changes = [], []
    for run_rows in (baseline, rows):
        for outer in range(1, 101):
            block = [r for r in run_rows if r["outer"] == outer]
            assert len(block) == 16 and [r["reuse"] for r in block] == list(range(1, 17))
            metrics = [k for k in block[0] if k not in ("run", "outer", "optimizer_update", "reuse")]
            result = dict(run=block[0]["run"], outer=outer, optimizer_update=16 * outer)
            result.update({k: st.mean(r[k] for r in block) for k in metrics})
            result["clipped_fraction"] = st.mean(r["grad_norm"] > 1 for r in block)
            outers.append(result)
            for metric in metrics:
                first, last = block[0][metric], block[-1][metric]
                changes.append(
                    dict(
                        run=result["run"],
                        outer=outer,
                        metric=metric,
                        first=first,
                        last=last,
                        delta=last - first,
                        relative_percent=100 * (last / first - 1) if first else None,
                    )
                )
    position = []
    for label, lo, hi in (("Outer 1-10", 1, 10), ("Outer 91-100", 91, 100)):
        selected = [r for r in updates if lo <= r["step/outer"] <= hi]
        for metric in ("future/coefficient", "teacher_rkl", "future/effective_mass"):
            for decile in range(10):
                prefix = f"distill/{metric}_position/p{decile * 10:02d}_{(decile + 1) * 10:02d}/"
                total = sum(r[prefix + "sum"] for r in selected)
                count = sum(r[prefix + "token_count"] for r in selected)
                position.append(dict(window=label, metric=metric, position=decile * 10 + 5, value=total / count))
    stats = {}
    for run in ("E82", "E84"):
        selected = [r for r in outers if r["run"] == run]
        stats[run] = {
            k: st.mean(r[k] for r in selected[-10:])
            for k in ("objective", "teacher_kl", "grad_norm", "clipped_fraction")
        }
        stats[run]["block_changes"] = {}
        for metric in ("objective", "teacher_kl", "coefficient", "coefficient_p95", "score_surrogate"):
            points = [r for r in changes if r["run"] == run and r["metric"] == metric]
            if points:
                stats[run]["block_changes"][metric] = dict(
                    decreased=sum(p["delta"] < 0 for p in points),
                    mean_delta=st.mean(p["delta"] for p in points),
                    median_relative_percent=st.median(p["relative_percent"] for p in points),
                )
    stats["E84"]["first_update"] = rows[0]
    stats["E84"]["last_update"] = rows[-1]
    stats["E84"]["negative_surrogate_updates"] = sum(r["training_surrogate"] < 0 for r in rows)
    stats["E84"]["negative_surrogate_outers"] = sorted({r["outer"] for r in rows if r["training_surrogate"] < 0})
    stats["E84"]["max_decomposition_error"] = max(
        abs(r["training_surrogate"] - r["local_surrogate"] - r["score_surrogate"]) for r in rows
    )
    stats["E84"]["final_scalar_cancellation_ratio"] = abs(rows[-1]["score_surrogate"]) / rows[-1]["local_surrogate"]
    # Sparse validation events must agree with the separately preserved console.
    valid = [r for r in raw if "eval/math500/acc/mean" in r]
    assert len(valid) == 5
    for r in csv.DictReader((BASE / "data/e84-20261110/validation.csv").open()):
        event = next(v for v in valid if v["step/outer"] == int(r["outer"]))
        slug = {"MATH-500": "math500", "AMC23": "amc23", "AIME24": "aime24", "AIME25": "aime25", "AIME26": "aime26"}[
            r["benchmark"]
        ]
        assert math.isclose(event[f"eval/{slug}/acc/mean"], float(r["accuracy"]), abs_tol=1e-12)
    write_csv(DATA / "updates.csv", rows)
    # Shared schema includes blank E82-only unavailable future diagnostics.
    columns = list(dict.fromkeys(k for row in outers for k in row))
    write_csv(DATA / "outers.csv", [{k: r.get(k, "") for k in columns} for r in outers])
    write_csv(DATA / "reuse-changes.csv", changes)
    write_csv(DATA / "position-profiles.csv", position)
    dump(DATA / "summary.json", stats)
    sources = [SNAPSHOT / "history.jsonl", SNAPSHOT / "run.json", REFERENCE, Path(__file__)]
    dump(
        DATA / "provenance.json",
        dict(
            log_repo_commit="3633b7c3a554e7e6e4ebdfa5c85ed9d61b881fd1",
            source=str(SOURCE),
            history_rows=len(raw),
            optimizer_updates=len(rows),
            outer_rows=100,
            validation_events=5,
            sources=[
                dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size)
                for p in sources
            ],
        ),
    )
    return rows, baseline, outers, changes, position


def plot(rows, baseline, outers, changes, position):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fv = style()
    theme = fv.apply_rc("paper")
    palette = fv.series_colors("paper", 9)
    colors = {"E82": palette[4], "E84": palette[1]}

    def finish(fig, stem, caption, handles):
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.045), ncol=len(handles), frameon=False)
        fig.text(0.5, 0.012, caption, ha="center", fontsize="small", color=fv.NEUTRALS["baseline"])
        fig.tight_layout(rect=(0, 0.13, 1, 1))
        save(fig, OUT, stem, fv, independent=True)

    def frame(ax, title, ylabel, xlabel="Outer (16 updates each)"):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        fv.apply_frame(ax, theme.frame)
        fv.apply_grid(ax, theme.grid)

    selected = [r for r in outers if r["run"] == "E84"]
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.6))
    specs = [
        (
            "(a) Future coefficient",
            "Unscaled cost",
            [("coefficient", "Mean", palette[1]), ("coefficient_p95", "p95", palette[0])],
        ),
        ("(b) Coefficient tail", "Unscaled cost", [("coefficient_max", "Maximum", palette[3])]),
        (
            "(c) Surrogate decomposition",
            "Scaled scalar",
            [
                ("local_surrogate", "Local", palette[0]),
                ("score_surrogate", "Future score", palette[3]),
                ("training_surrogate", "Total", palette[1]),
            ],
        ),
        ("(d) Coefficient spread", "Standard deviation", [("coefficient_std", "Std", palette[1])]),
        ("(e) Strict-suffix horizon", "Mean valid suffix tokens", [("effective_mass", "Suffix tokens", palette[1])]),
        ("(f) Scalar cancellation", "Absolute future / local", []),
    ]
    for ax, (title, ylabel, metrics) in zip(axes.flat, specs, strict=True):
        for key, label, color in metrics:
            ax.plot([r["outer"] for r in selected], [r[key] for r in selected], color=color, label=label)
        if not metrics:
            ax.plot(
                [r["outer"] for r in selected],
                [abs(r["score_surrogate"]) / r["local_surrogate"] for r in selected],
                color=palette[1],
            )
        frame(ax, title, ylabel)
        if len(metrics) > 1:
            ax.legend(frameon=False)
    # Panel-local legends explain diagnostic roles; no duplicated figure legend.
    fig.text(
        0.5,
        0.018,
        "E84: each point averages 16 logged pre-update values; scalar ratios are not gradient ratios",
        ha="center",
        fontsize="small",
        color=fv.NEUTRALS["baseline"],
    )
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    save(fig, OUT, "future-diagnostics", fv, independent=True)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.3))
    for ax, key, title, ylabel in zip(
        axes.flat,
        ("teacher_kl", "objective", "grad_norm", "clipped_fraction"),
        ("(a) Adapted teacher KL", "(b) Local MDPO objective", "(c) Preclip gradient norm", "(d) Norm above 1"),
        ("KL / token", "Target KL / eta", "Gradient norm", "Fraction of updates"),
        strict=True,
    ):
        for run in colors:
            group = [r for r in outers if r["run"] == run]
            ax.plot([r["outer"] for r in group], [r[key] for r in group], color=colors[run])
        frame(ax, title, ylabel)
    finish(
        fig,
        "kl-comparison",
        "Block means on each run's own trajectories; same teacher-KL metric, changing visited prefixes",
        [Line2D([], [], color=c, label=r) for r, c in colors.items()],
    )

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    for ax, metric, title in zip(
        axes,
        ("objective", "teacher_kl", "coefficient"),
        ("(a) Local MDPO objective", "(b) Adapted teacher KL", "(c) E84 future coefficient"),
        strict=True,
    ):
        for run in colors:
            group = [r for r in changes if r["run"] == run and r["metric"] == metric]
            if group:
                ax.plot([r["outer"] for r in group], [r["relative_percent"] for r in group], color=colors[run])
        frame(ax, title, "First-to-last change (%)")
    finish(
        fig,
        "reuse-changes",
        "Observations before updates 1 and 16: 15 intervening updates on identical trajectories",
        [Line2D([], [], color=c, label=r) for r, c in colors.items()],
    )

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    for ax, metric, title, ylabel in zip(
        axes,
        ("future/coefficient", "teacher_rkl", "future/effective_mass"),
        ("(a) E84 future coefficient", "(b) E84 native teacher RKL", "(c) E84 strict-suffix horizon"),
        ("Unscaled cost", "KL / token", "Valid suffix tokens"),
        strict=True,
    ):
        for label, color in (("Outer 1-10", palette[0]), ("Outer 91-100", palette[1])):
            group = [r for r in position if r["metric"] == metric and r["window"] == label]
            ax.plot([r["position"] for r in group], [r["value"] for r in group], color=color, marker="o")
        frame(ax, title, ylabel, "Relative response position (%)")
    finish(
        fig,
        "position-profiles",
        "Relative-length deciles; pooled token sums/counts across each window (E84 only)",
        [
            Line2D([], [], color=c, marker="o", label=r)
            for r, c in (("Outer 1-10", palette[0]), ("Outer 91-100", palette[1]))
        ],
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for reuse in (1, 8, 16):
        group = [r for r in rows if r["reuse"] == reuse]
        color = palette[{1: 0, 8: 1, 16: 3}[reuse]]
        axes[0].plot([r["outer"] for r in group], [r["anchor_kl"] for r in group], color=color)
        axes[1].plot([r["outer"] for r in group], [r["target_kl"] for r in group], color=color)
    frame(axes[0], "(a) E84 anchor KL", "KL / token")
    frame(axes[1], "(b) E84 target KL", "KL / token")
    finish(
        fig,
        "anchor-target-kl",
        "Pre-update diagnostics; anchor resets at each outer, target is the fixed proximal mixture",
        [Line2D([], [], color=palette[i], label=f"Reuse {r}") for r, i in ((1, 0), (8, 1), (16, 3))],
    )


if __name__ == "__main__":
    plot(*extract())
