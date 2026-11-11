# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Calibration-selected finite updates and paired untouched-test improvements.

Semantic ledger: direction/native objective are distinct; every test gain is
on the unchanged KL, not a component objective. Source replicates are three
fixed previously unseen B8 groups, not independent training seeds. Geometric
ledger: equal displacement grid on every direction; gain divided by each
receiver panel's baseline KL; bands resample source and receiver groups.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .report_groups import basic_interval, bootstrap_weights
from .report_stages import AUDITS, DEFAULT_STYLE, digest, frame, save
from .results import write_csv
from .runtime import write_json


def analyze_updates(data, geometry):
    groups, gram = geometry["groups"], np.asarray(geometry["gram"])
    lookup = {(g["group"], g["split"], g["field"]): i for i, g in enumerate(groups)}
    baseline = {r["row_id"]: r for r in data["baseline"]}
    cal = set(data["signature"]["calibration"])
    branch = data["signature"]["task"]["target"]
    rows, grouped = [], []
    for trial in data["trials"]:
        source, field = trial["source"], trial["field"]
        il = lookup[(source, "train", "live")]
        ip = lookup.get((source, "train", "parallel"))
        weights = {"native": [(il, 1)], "parallel": [(ip, 1)], "perpendicular": [(il, 1), (ip, -1)]}[field]
        # A live/frozen trajectory Gram cannot predict a parallel intervention.
        available = all(i is not None for i, _ in weights)
        predicted_norm = (
            np.sqrt(sum(a * b * gram[i, j] for i, a in weights for j, b in weights)) if available else float("nan")
        )
        if available:
            np.testing.assert_allclose(predicted_norm, trial["gradient_norm"], rtol=2e-5, atol=1e-10)
        for role in ("train", "calibration", "test"):
            values = [
                r
                for r in trial["records"]
                if (role == "train" and r["split"] == "train")
                or (
                    r["split"] == "heldout"
                    and ((role == "calibration" and r["group"] in cal) or (role == "test" and r["group"] not in cal))
                )
            ]
            if not values:
                continue
            receiver = sorted({r["group"] for r in values})
            before, gain, prediction = [], [], []
            for label in receiver:
                selected = [r for r in values if r["group"] == label]
                old = np.mean([baseline[r["row_id"]]["loss"] for r in selected])
                difference = old - np.mean([r["loss"] for r in selected])
                index = lookup[(label, "train" if role == "train" else "heldout", "live")]
                expected = (
                    sum(a * gram[index, i] for i, a in weights) * trial["displacement"] / predicted_norm
                    if available
                    else float("nan")
                )
                before.append(old)
                gain.append(difference)
                prediction.append(expected)
                grouped.append(
                    dict(
                        branch=branch,
                        source=source,
                        field=field,
                        displacement=trial["displacement"],
                        role=role,
                        group=label,
                        baseline=old,
                        gain=difference,
                        predicted_gain=expected,
                    )
                )
            rows.append(
                dict(
                    branch=branch,
                    source=source,
                    field=field,
                    displacement=trial["displacement"],
                    role=role,
                    baseline=float(np.mean(before)),
                    gain=float(np.mean(gain)),
                    relative_gain=float(np.mean(gain) / np.mean(before)),
                    predicted_gain=float(np.mean(prediction)),
                    groups=len(receiver),
                )
            )
    selected = []
    for source in sorted({r["source"] for r in rows}):
        for field in ("native", "parallel", "perpendicular"):
            candidates = [
                r for r in rows if r["source"] == source and r["field"] == field and r["role"] == "calibration"
            ]
            best = max(candidates, key=lambda r: r["gain"])
            if best["gain"] > 0:
                test = next(
                    r
                    for r in rows
                    if r["source"] == source
                    and r["field"] == field
                    and r["role"] == "test"
                    and r["displacement"] == best["displacement"]
                )
                selected.append(dict(**test, calibration_gain=best["gain"]))
            else:
                selected.append(
                    dict(
                        branch=branch,
                        source=source,
                        field=field,
                        displacement=0.0,
                        role="test",
                        baseline=next(
                            r["baseline"]
                            for r in rows
                            if r["role"] == "test" and r["source"] == source and r["field"] == field
                        ),
                        gain=0.0,
                        relative_gain=0.0,
                        predicted_gain=0.0,
                        calibration_gain=0.0,
                        groups=len(
                            {r["group"] for r in data["baseline"] if r["split"] == "heldout" and r["group"] not in cal}
                        ),
                    )
                )
    return rows, grouped, selected


def aggregate(grouped, replicates=4000):
    rows = []
    conditions = sorted({(r["branch"], r["field"], r["displacement"]) for r in grouped if r["role"] == "test"})
    for branch, field, delta in conditions:
        selected = [
            r
            for r in grouped
            if r["branch"] == branch and r["field"] == field and r["displacement"] == delta and r["role"] == "test"
        ]
        sources = sorted({r["source"] for r in selected})
        receivers = sorted({r["group"] for r in selected})
        lookup = {(r["source"], r["group"]): r for r in selected}
        gain = np.array([[lookup[(s, h)]["gain"] for h in receivers] for s in sources])
        predicted = np.array([[lookup[(s, h)]["predicted_gain"] for h in receivers] for s in sources])
        loss = np.array([lookup[(sources[0], h)]["baseline"] for h in receivers])
        rng = np.random.default_rng(20261109)
        ws = bootstrap_weights(len(sources), replicates, rng)
        wh = bootstrap_weights(len(receivers), replicates, rng)
        point = gain.mean() / loss.mean()
        sample = np.einsum("bi,ij,bj->b", ws, gain, wh) / (wh @ loss)
        low, high = basic_interval(point, sample)
        rows.append(
            dict(
                branch=branch,
                field=field,
                displacement=delta,
                value=point,
                low=low,
                high=high,
                predicted=predicted.mean() / loss.mean(),
                source_groups=len(sources),
                receiver_groups=len(receivers),
            )
        )
    return rows


def paired_direction_contrasts(grouped):
    """Compare directions using identical source and receiver resamples."""
    native = {
        (r["branch"], r["source"], r["displacement"], r["group"]): r
        for r in grouped
        if r["role"] == "test" and r["field"] == "native"
    }
    differences = []
    for row in grouped:
        if row["role"] != "test" or row["field"] == "native":
            continue
        reference = native[(row["branch"], row["source"], row["displacement"], row["group"])]
        differences.append(
            dict(
                row,
                field=row["field"] + "-minus-native",
                gain=row["gain"] - reference["gain"],
                predicted_gain=row["predicted_gain"] - reference["predicted_gain"],
            )
        )
    return aggregate(differences)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    args = parser.parse_args()
    out = args.root / "analysis"
    out.mkdir(exist_ok=True)
    rows, grouped, selected, hashes = [], [], [], {}
    for path in sorted(args.root.glob("*.updates.json")):
        data = json.loads(path.read_text())
        task = data["signature"]["task"]
        geometry = args.geometry / f"{task['target']}-s{task['stage']:03d}.json"
        a, b, c = analyze_updates(data, json.loads(geometry.read_text()))
        rows.extend(a)
        grouped.extend(b)
        selected.extend(c)
        hashes[str(path)], hashes[str(geometry)] = digest(path), digest(geometry)
    summary = aggregate(grouped)
    for name, values in (
        ("updates", rows),
        ("update-groups", grouped),
        ("selected-test", selected),
        ("test-summary", summary),
        ("direction-contrasts", paired_direction_contrasts(grouped)),
    ):
        write_csv(out / (name + ".csv"), values)
    write_json(out / "sources.json", hashes)
    sys.path.insert(0, str(DEFAULT_STYLE))
    import fvstyle as fv

    fv.apply_rc("paper")
    for name, limit in (("equal-displacement-test", 0.03), ("small-displacement-test", 0.01)):
        fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.2), sharey=True)
        fig.subplots_adjust(left=0.10, right=0.98, bottom=0.26, top=0.86, wspace=0.36)
        for ax, branch in zip(axes, ("justrl", "polaris"), strict=True):
            for field, color in zip(("native", "parallel", "perpendicular"), fv.series_colors("paper", 3), strict=True):
                values = sorted(
                    [
                        r
                        for r in summary
                        if r["branch"] == branch and r["field"] == field and r["displacement"] <= limit
                    ],
                    key=lambda r: r["displacement"],
                )
                x = [r["displacement"] for r in values]
                ax.plot(x, [100 * r["value"] for r in values], "o-", label=field, color=color)
                ax.fill_between(
                    x, [100 * r["low"] for r in values], [100 * r["high"] for r in values], color=color, alpha=0.16
                )
            ax.axhline(0, color=fv.NEUTRALS["baseline"], linewidth=0.6)
            frame(ax, branch + " displacement norm", "Untouched test KL reduction (%)")
        fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False)
        save(
            fig,
            out / name,
            "Three unseen source batches; six untouched B8 receiver groups; original RKL.",
            fv,
            caption_fontsize=8,
        )
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.28, top=0.85)
    fields = ("native", "parallel", "perpendicular")
    for i, (branch, color) in enumerate(zip(("justrl", "polaris"), fv.series_colors("paper", 2), strict=True)):
        points = [
            next(r for r in summary if r["branch"] == branch and r["field"] == f and r["displacement"] == 0.003)
            for f in fields
        ]
        x = np.arange(3) + (i - 0.5) * 0.16
        ax.plot(x, [100 * r["value"] for r in points], "o", color=color, label=branch)
        ax.vlines(x, [100 * r["low"] for r in points], [100 * r["high"] for r in points], color=color)
    ax.set_xticks(np.arange(3), fields)
    ax.axhline(0, color=fv.NEUTRALS["baseline"], linewidth=0.6)
    frame(ax, "Update direction (displacement 0.003)", "Test KL reduction (%)")
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False)
    save(
        fig,
        out / "fixed-displacement-test",
        "Three source batches; 48 untouched test questions; paired B8 intervals.",
        fv,
        caption_fontsize=8,
    )
    write_json(out / "figure-audits.json", AUDITS)


if __name__ == "__main__":
    main()
