# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Join existing L risks, batch Gram and frozen-request probes without model runs.

Semantic ledger: teacher colors fixed; solid/dashed denote own/receiver signal;
prefix intervals resample paired trajectories, never individual token positions.
Geometry ledger: joint x is L optimizer updates; prefix x is split/domain;
ratios compare stage 240 with stage 0, holding selected coordinates fixed.
B8 PAC and full B16 training risk are separate observations, not paired samples.
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .report_stages import DEFAULT_STYLE, frame
from .results import write_csv


def load(path, sources):
    sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


def gram_stats(data, indices):
    g = np.asarray(data["geometry"]["gram"])[np.ix_(indices, indices)]
    own = float(np.diag(g).mean())
    cross = float((g.sum() - np.trace(g)) / (len(g) * (len(g) - 1))) if len(g) > 1 else None
    return dict(self_energy=own, cross_energy=cross, cross_coherence=cross / own if cross is not None else None)


def prefix_ratios(initial, final, target, indices):
    def values(data, field):
        f = data["fields"][f"{target}/{field}"]
        p = np.asarray(f["pressure"])[indices].mean()
        a = np.asarray(f["gradient_norm_sq"])[indices].mean() / p
        return p, a

    p0, a0 = values(initial, "live")
    p, a = values(final, "live")
    _, af = values(final, "frozen_B0")
    return np.asarray([p / p0, a / a0, af / a0])


def prefix_analysis(root, sources):
    rows, pooled, sensitivity = [], [], []
    for branch in ["justrl", "polaris"]:
        directory = root / "point-probe" / f"r1-{branch}"
        initial = load(directory / "s000.json", sources)
        final = load(directory / "s240.json", sources)
        if initial["prefixes"] != final["prefixes"]:
            raise ValueError("Prefix coordinates changed")
        strata = {}
        for i, p in enumerate(initial["prefixes"]):
            strata.setdefault(f"{p['split']}/{p['domain']}", {}).setdefault(p["row_id"], []).append(i)
        for name, trajectories in sorted(strata.items()):
            units = list(trajectories.values())
            indices = sum(units, [])
            estimate = prefix_ratios(initial, final, branch, indices)
            leave_one = np.asarray(
                [
                    prefix_ratios(initial, final, branch, sum([u for k, u in enumerate(units) if k != drop], []))
                    for drop in range(len(units))
                ]
            )
            f = final["fields"][f"{branch}/live"]
            contributions = [float(np.asarray(f["pressure"])[u].sum()) for u in units]
            sensitivity.append(
                dict(
                    branch=branch,
                    stratum=name,
                    largest_live_pressure_share=max(contributions) / sum(contributions),
                    frozen_A_leave_one_min=float(leave_one[:, 2].min()),
                    frozen_A_leave_one_max=float(leave_one[:, 2].max()),
                )
            )
            rng = np.random.default_rng(20261109)
            draws = np.asarray(
                [
                    prefix_ratios(initial, final, branch, sum([units[i] for i in choice], []))
                    for choice in rng.integers(0, len(units), size=(2000, len(units)))
                ]
            )
            for k, metric in enumerate(["live_pressure_ratio", "live_A_ratio", "frozen_A_ratio"]):
                lo, hi = np.quantile(draws[:, k], [0.025, 0.975])
                rows.append(
                    dict(
                        branch=branch,
                        stage=240,
                        stratum=name,
                        trajectories=len(units),
                        prefixes=len(indices),
                        metric=metric,
                        estimate=float(estimate[k]),
                        q025=float(lo),
                        q975=float(hi),
                    )
                )
        for stage in [0, 16, 80, 160, 240, 320]:
            path = directory / f"s{stage:03d}.json"
            if not path.exists():
                continue
            data = load(path, sources)
            for field in ["live", "frozen_B0"]:
                s = data["fields"][f"{branch}/{field}"]["summary"]
                pooled.append(dict(branch=branch, stage=stage, field=field, **s))
    return rows, pooled, sensitivity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--style", type=Path, default=DEFAULT_STYLE)
    args = parser.parse_args()
    sys.path.insert(0, str(args.style))
    import fvstyle as fv

    fv.apply_rc("paper")
    root = args.manifest.parent
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    sources, audits = {}, {}
    colors = dict(justrl=fv.HUES["vteal"], polaris=fv.HUES["vorange"])

    def save(fig, name, caption):
        fig.text(0.5, 0.025, caption, ha="center", color=fv.NEUTRALS["baseline"], fontsize=8)
        fig.canvas.draw()
        audits[name] = [str(p) for p in fv.audit(fig)]
        for suffix in ["png", "pdf"]:
            fig.savefig(out / f"{name}.{suffix}", dpi=220)
        plt.close(fig)

    manifest = load(args.manifest, sources)
    risks, geometry, actual = [], [], []
    for target in colors:
        path = args.history / f"{target}-risk-matrix.csv"
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open() as stream:
            matrix = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
        for i in range(20):
            label = chr(ord("A") + i)
            before, after = matrix[i], matrix[i + 1]
            held_before = np.mean([before[f"M{k:02d}"] for k in range(1, 13)])
            held_after = np.mean([after[f"M{k:02d}"] for k in range(1, 13)])
            risks.append(
                dict(
                    target=target,
                    stage=i * 16,
                    incoming_group=label,
                    own_before=before[label],
                    own_after=after[label],
                    incoming_remaining=before[label] / matrix[0][label],
                    local_relative_gain=1 - after[label] / before[label],
                    held_loss=held_before,
                    held_relative_gain=1 - held_after / held_before,
                )
            )
        probes = load(args.history / target / "update-probes.json", sources)
        for p in probes:
            actual.append(
                dict(
                    target=target,
                    stage=(p["training_batch"] - 1) * 16,
                    predicted_held_gain=-p["held_dot_actual_update"],
                    actual_held_gain=-p["finite_held_change"],
                    adam_alignment=p["descent_alignment"],
                )
            )
    for task in manifest["tasks"]:
        if task["model"] != "r1" or "-target-" in task["id"]:
            continue
        data = load(root / "results" / task["id"] / "result.json", sources)
        if not data.get("finished") or data["signature"]["task"] != task:
            raise ValueError(f"Invalid stage: {task['id']}")
        groups = data["groups"]
        held = [i for i, g in enumerate(groups) if g["split"] == "heldout" and g["domain"] == "math"]
        # Restrict sources to batches not trained yet at this checkpoint.
        future = [
            i
            for i, g in enumerate(groups)
            if g["split"] == "train" and ord(g["label"]) - ord("A") >= task["stage"] // 16
        ]
        G = np.asarray(data["geometry"]["gram"])
        row = dict(target=task["target"], stage=task["stage"], **gram_stats(data, held))
        row["future_batches"] = len(future)
        row["future_to_held_inner"] = float(G[np.ix_(future, held)].mean()) if future else None
        row["future_to_held_relative"] = (
            row["future_to_held_inner"] / data["summaries"]["heldout/math"]["loss_mean"] if future else None
        )
        geometry.append(row)
    prefix, pooled, sensitivity = prefix_analysis(root, sources)
    for name, rows in [
        ("joint-risk", risks),
        ("joint-geometry", geometry),
        ("joint-adam", actual),
        ("prefix-stratified", prefix),
        ("prefix-pooled", pooled),
        ("prefix-sensitivity", sensitivity),
    ]:
        write_csv(out / f"{name}.csv", rows)

    fig, axes = plt.subplots(3, 2, figsize=(7, 8.3))
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.12, top=0.89, hspace=0.65, wspace=0.5)
    for target, color in colors.items():
        rr = [r for r in risks if r["target"] == target]
        gg = sorted([r for r in geometry if r["target"] == target], key=lambda r: r["stage"])
        x = [r["stage"] for r in rr]
        axes[0, 0].plot(x, [r["own_before"] for r in rr], "o-", color=color, ms=2, label=f"{target}: before")
        axes[0, 0].plot(x, [r["own_after"] for r in rr], "--", color=color, label=f"{target}: after")
        axes[0, 1].plot(x, [r["incoming_remaining"] for r in rr], "o-", color=color, ms=2, label=target)
        axes[1, 0].plot(x, [r["local_relative_gain"] for r in rr], "-", color=color)
        axes[1, 0].plot(x, [r["held_relative_gain"] for r in rr], "--", color=color)
        axes[1, 1].plot([r["stage"] for r in gg], [r["cross_coherence"] for r in gg], "o-", color=color, ms=3)
        axes[2, 0].plot([r["stage"] for r in gg], [r["self_energy"] for r in gg], "o-", color=color, ms=3)
        axes[2, 0].plot([r["stage"] for r in gg], [r["cross_energy"] for r in gg], "s--", color=color, ms=3)
        gg = [r for r in gg if r["future_to_held_inner"] is not None]
        axes[2, 1].plot([r["stage"] for r in gg], [r["future_to_held_relative"] for r in gg], "o-", color=color, ms=3)
    labels = [
        "Current batch RKL: before / after",
        "Incoming RKL / its initial RKL",
        "Relative gain: current / heldout",
        "Heldout distinct-batch coherence",
        "Heldout self / cross energy",
        "Future-to-held inner / held loss",
    ]
    for ax, label in zip(axes.flat, labels, strict=True):
        frame(ax, "Updates before training block", label)
        ax.set_xticks([0, 160, 320])
    for ax in [axes[0, 0], axes[2, 0]]:
        ax.set_yscale("log")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=2, fontsize=8)
    axes[0, 1].legend(fontsize=8)
    save(
        fig,
        "r1_joint_transfer",
        "Top/left-middle: original B16 risk. Bottom/other middle: B8 diagnostic subset.\n"
        "Solid/dashed: before/after (top left), current/heldout gain (middle left), self/cross (bottom left).",
    )

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.4))
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.3, top=0.92, wspace=0.55)
    for target, color in colors.items():
        rr = [r for r in actual if r["target"] == target]
        x = [r["stage"] for r in rr]
        axes[0].plot(x, [r["adam_alignment"] for r in rr], "o-", color=color, ms=2, label=target)
        axes[1].plot(x, [r["predicted_held_gain"] for r in rr], "-", color=color)
        axes[1].plot(x, [r["actual_held_gain"] for r in rr], "--", color=color)
    axes[0].legend(fontsize=8)
    frame(axes[0], "Updates before training block", "Adam descent alignment")
    frame(axes[1], "Updates before training block", "Held64 one-step decrease")
    axes[1].set_yscale("symlog", linthresh=1e-7)
    save(
        fig,
        "r1_joint_adam",
        "Existing first-step B16 probes on fixed Held64.\n"
        "Right: solid = local linear prediction, dashed = measured decrease; symlog preserves negative values.",
    )

    fig, axes = plt.subplots(2, 2, figsize=(7, 5.8), sharey="row")
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.18, top=0.9, hspace=0.65, wspace=0.5)
    strata = sorted({r["stratum"] for r in prefix})
    for col, branch in enumerate(colors):
        for row, metric in enumerate(["live_pressure_ratio", "frozen_A_ratio"]):
            ss = [
                next(r for r in prefix if r["branch"] == branch and r["stratum"] == s and r["metric"] == metric)
                for s in strata
            ]
            y = np.asarray([r["estimate"] for r in ss])
            low, high = np.asarray([[r["q025"], r["q975"]] for r in ss]).T
            fv.mark_neutral(axes[row, col].vlines(range(len(ss)), low, high, color=colors[branch]))
            fv.mark_neutral(axes[row, col].scatter(range(len(ss)), y, color=colors[branch], s=15))
            axes[row, col].axhline(1, color=fv.NEUTRALS["baseline"], ls=":", lw=0.8)
            axes[row, col].set_xticks(
                range(len(ss)),
                [s.replace("heldout/", "H/").replace("train/", "T/") for s in strata],
                rotation=25,
                ha="right",
            )
            axes[row, col].set_yscale("log")
            frame(
                axes[row, col],
                "Fixed prefix stratum",
                "Live P: stage240 / initial" if row == 0 else "Frozen A: stage240 / initial",
            )
        axes[0, col].set_title(branch)
    save(
        fig,
        "r1_prefix_stratified",
        "Same initial request B0 for frozen A; 4 trajectories per stratum, 3 sites each.\n"
        "Intervals: 2,000 paired trajectory bootstrap draws; sparse-bank uncertainty, not training-seed variation.",
    )
    (out / "joint-analysis-audit.json").write_text(
        json.dumps(
            dict(
                sources=sources,
                figure_audits=audits,
                limitations=[
                    "Prefix Gram not saved: subgroup C and cross-trajectory shared energy cannot be recovered.",
                    "Pooled prefix C includes within-trajectory site pairs and multiple domains.",
                    "Future source block is empty at update320; no extrapolated point.",
                ],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
