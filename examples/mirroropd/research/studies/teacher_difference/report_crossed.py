# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Four-cell teacher/student interventions and restored-history AdamW.

Each test point retains original B8 receiver groups. Adam scale multiplies the
saved step displacement; it is not a change to the optimizer learning rate.
All uncertainty concerns the fixed-bank source/receiver sampling, not run seeds.
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
from .report_updates import aggregate, analyze_updates, paired_direction_contrasts
from .results import write_csv
from .runtime import write_json


def adam_groups(data, branch):
    baseline = {r["row_id"]: r for r in data["baseline"]}
    calibration = set(data["signature"]["calibration"])
    output = []
    for trial in data["trials"]:
        for group in sorted({r["group"] for r in trial["records"]}):
            rows = [r for r in trial["records"] if r["group"] == group]
            split = {r["split"] for r in rows}
            if len(split) != 1:
                raise ValueError("Mixed split within receiver group")
            role = "train" if split == {"train"} else "calibration" if group in calibration else "test"
            before = np.mean([baseline[r["row_id"]]["loss"] for r in rows])
            output.append(
                dict(
                    branch=branch,
                    source=trial["source"],
                    field="restored-adam",
                    displacement=trial["scale"],
                    role=role,
                    group=group,
                    baseline=float(before),
                    gain=float(before - np.mean([r["loss"] for r in rows])),
                    predicted_gain=trial["predictions"][group],
                )
            )
    return output


def calibrated_test(grouped):
    """Select with calibration only, including an explicit zero-update option."""
    selected = []
    keys = sorted({(r["branch"], r["source"], r["field"]) for r in grouped})
    for branch, source, field in keys:
        rows = [r for r in grouped if (r["branch"], r["source"], r["field"]) == (branch, source, field)]
        candidates = sorted({r["displacement"] for r in rows})
        scores = {
            delta: np.mean([r["gain"] for r in rows if r["role"] == "calibration" and r["displacement"] == delta])
            for delta in candidates
        }
        best = max(scores, key=scores.get)
        if scores[best] <= 0:
            selected.extend(
                dict(r, displacement=0.0, gain=0.0, predicted_gain=0.0)
                for r in rows
                if r["role"] == "test" and r["displacement"] == candidates[0]
            )
        else:
            selected.extend(r for r in rows if r["role"] == "test" and r["displacement"] == best)
    # Aggregate expects one common condition key, while selections vary by source.
    choices = [dict(r, selected_displacement=r["displacement"], displacement=-1.0) for r in selected]
    return selected, aggregate(choices)


def paired_cell_contrasts(grouped, *, replicates=4000):
    """Differences of relative gains, with each condition's own native risk."""
    output = []
    pairs = (
        ("teacher-J-minus-P-on-J", "justrl-on-justrl", "polaris-on-justrl"),
        ("teacher-J-minus-P-on-P", "justrl-on-polaris", "polaris-on-polaris"),
        ("student-J-minus-P-for-J", "justrl-on-justrl", "justrl-on-polaris"),
        ("student-J-minus-P-for-P", "polaris-on-justrl", "polaris-on-polaris"),
    )
    conditions = sorted({(r["field"], r["displacement"]) for r in grouped if r["field"] in ("native", "restored-adam")})
    for field, delta in conditions:
        cells = {}
        for name in {name for _, a, b in pairs for name in (a, b)}:
            rows = [
                r
                for r in grouped
                if r["branch"] == name and r["field"] == field and r["displacement"] == delta and r["role"] == "test"
            ]
            cells[name] = {(r["source"], r["group"]): r for r in rows}
        keys = set(cells["justrl-on-justrl"])
        if any(set(rows) != keys for rows in cells.values()):
            raise ValueError("Paired cell contrasts require identical source/test groups")
        sources, receivers = sorted({s for s, _ in keys}), sorted({h for _, h in keys})
        rng = np.random.default_rng(20261109)
        ws, wh = bootstrap_weights(len(sources), replicates, rng), bootstrap_weights(len(receivers), replicates, rng)
        values = {}
        for name, rows in cells.items():
            gains = np.array([[rows[s, h]["gain"] for h in receivers] for s in sources])
            losses = np.array([rows[sources[0], h]["baseline"] for h in receivers])
            values[name] = (gains.mean() / losses.mean(), np.einsum("bi,ij,bj->b", ws, gains, wh) / (wh @ losses))
        for label, a, b in pairs:
            point = values[a][0] - values[b][0]
            low, high = basic_interval(point, values[a][1] - values[b][1])
            output.append(
                dict(
                    field=field,
                    displacement=delta,
                    contrast=label,
                    difference=float(point),
                    low=float(low),
                    high=float(high),
                )
            )
    return output


def plots(summary, out, fv):
    cells = ("justrl-on-justrl", "polaris-on-justrl", "justrl-on-polaris", "polaris-on-polaris")
    colors = fv.series_colors("paper", 2)
    for field, filename, xlabel in (
        ("native", "native-sgd-test", "Parameter displacement norm"),
        ("restored-adam", "restored-adam-test", "Fraction of historical Adam displacement"),
    ):
        fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.7), sharey=True)
        fig.subplots_adjust(left=0.13, right=0.97, bottom=0.20, top=0.91, wspace=0.36, hspace=0.60)
        for ax, cell in zip(axes.flat, cells, strict=True):
            rows = sorted(
                [r for r in summary if r["branch"] == cell and r["field"] == field], key=lambda r: r["displacement"]
            )
            x = [r["displacement"] for r in rows]
            ax.plot(x, [100 * r["value"] for r in rows], "o-", color=colors[0], label="Observed")
            ax.fill_between(
                x, [100 * r["low"] for r in rows], [100 * r["high"] for r in rows], color=colors[0], alpha=0.16
            )
            ax.plot(x, [100 * r["predicted"] for r in rows], "s--", color=colors[1], label="Linear prediction")
            ax.axhline(0, color=fv.NEUTRALS["baseline"], linewidth=0.6)
            ax.set_title(cell, fontsize=9)
            frame(ax, xlabel, "Test KL reduction (%)")
        fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False)
        save(
            fig,
            out / filename,
            "Teacher-on-student at step 240; three future B8 sources; 48 untouched test questions.",
            fv,
            caption_fontsize=8,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--adam", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    grouped, hashes, metadata = [], {}, []
    for cell in json.loads(args.cells.read_text()):
        paths = {key: Path(cell[key]) for key in ("trajectory", "updates")}
        data = {key: json.loads(path.read_text()) for key, path in paths.items()}
        for key in ("student", "teacher", "records", "stage", "cap"):
            if data["updates"]["signature"]["task"][key] != data["trajectory"]["signature"]["task"][key]:
                raise ValueError(f"Update/trajectory input mismatch: {cell['label']}/{key}")
        _, rows, _ = analyze_updates(data["updates"], data["trajectory"])
        grouped.extend(dict(r, branch=cell["label"]) for r in rows)
        name = cell["label"] if cell["student"] != cell["teacher"] else cell["student"]
        paths["adam"] = args.adam / f"{name}-s240.adam.json"
        adam = json.loads(paths["adam"].read_text())
        for key in ("student", "teacher", "records", "stage", "cap"):
            if adam["signature"]["task"][key] != data["updates"]["signature"]["task"][key]:
                raise ValueError(f"Adam/update input mismatch: {cell['label']}/{key}")
        grouped.extend(adam_groups(adam, cell["label"]))
        metadata.extend(
            dict(branch=cell["label"], **{k: v for k, v in t.items() if k not in ("records", "predictions")})
            for t in adam["trials"]
        )
        hashes.update({str(path): digest(path) for path in paths.values()})
    summary = aggregate(grouped)
    selected, selected_summary = calibrated_test(grouped)
    for name, rows in (
        ("update-groups", grouped),
        ("test-summary", summary),
        ("paired-cell-contrasts", paired_cell_contrasts(grouped)),
        ("adam-step-metadata", metadata),
        ("calibration-selected-groups", selected),
        ("calibration-selected-summary", selected_summary),
        ("direction-contrasts", paired_direction_contrasts([r for r in grouped if r["field"] != "restored-adam"])),
    ):
        write_csv(args.out / (name + ".csv"), rows)
    write_json(args.out / "sources.json", hashes)
    write_json(
        args.out / "methods.json",
        dict(
            replicates=4000,
            interval="signed basic, independent original source/receiver groups",
            calibration="first six heldout B8 groups",
            test="last six heldout B8 groups",
            adam="historical moments, hyperparameters, weight decay, clip=1; B8 counterfactual source",
            scales="fractions of one Adam displacement, not new learning rates",
            component_predictions="NaN when live/frozen Gram does not include parallel direction",
        ),
    )
    sys.path.insert(0, str(DEFAULT_STYLE))
    import fvstyle as fv

    fv.apply_rc("paper")
    plots(summary, args.out, fv)
    write_json(args.out / "figure-audits.json", AUDITS)


if __name__ == "__main__":
    main()
