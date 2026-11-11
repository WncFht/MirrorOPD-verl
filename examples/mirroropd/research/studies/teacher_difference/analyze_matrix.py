# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent train/heldout propagation for generic and crossed fixed banks.

Semantic ledger: each row is one checkpoint/teacher/bank; no teacher endpoints
are interpreted as a continuous training path. Bootstrap units are original
groups, with independent source and receiver draws and paired condition seeds.
Geometric ledger: dot intervals show signed D / native receiver KL. Frozen
requests use that KL only as a reference, not as a gradient-flow identity.
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
from .results import load_results, write_csv
from .runtime import write_json


def panel_statistics(gram, source, held, loss_t, loss_h, *, replicates=4000):
    """Independent-group propagation; retain signed cross terms and intervals."""
    gram = np.asarray(gram)
    nt, nh = len(source), len(held)
    if not nt or not nh:
        raise ValueError("Both independent panels must contain groups")
    tt, hh, ht = gram[np.ix_(source, source)], gram[np.ix_(held, held)], gram[np.ix_(held, source)]
    lt, lh = np.mean(loss_t), np.mean(loss_h)
    d = ht.mean()
    et, eh = tt.mean(), hh.mean()
    rng = np.random.default_rng(20261109)
    wt, wh = bootstrap_weights(nt, replicates, rng), bootstrap_weights(nh, replicates, rng)
    sample = np.einsum("bi,ij,bj->b", wh, ht, wt) / (wh @ np.asarray(loss_h))
    low, high = basic_interval(d / lh, sample)
    return dict(
        L_train=float(lt),
        L=float(lh),
        D=float(d),
        D_over_L=float(d / lh),
        low=low,
        high=high,
        cosine=float(d / np.sqrt(et * eh)),
        q_train=float(et / lt),
        q=float(eh / lh),
        source_groups=nt,
        heldout_groups=nh,
    )


def describe(groups, gram, *, field="live", future=None):
    """Small adapter contract: group has rows, loss, pressure and second arrays."""
    selected = [i for i, g in enumerate(groups) if g["field"] == field]
    source = [i for i in selected if groups[i]["split"] == "train" and (future is None or groups[i]["group"] in future)]
    held = [i for i in selected if groups[i]["split"] == "heldout"]
    sizes = {len(groups[i]["rows"]) for i in selected}
    if len(sizes) != 1:
        raise ValueError("Equal-sized original groups required for this estimator")
    live_loss = {(g["split"], g["group"]): np.mean(g["losses"]) for g in groups if g["field"] == "live"}
    losses = lambda indices: [live_loss[(groups[i]["split"], groups[i]["group"])] for i in indices]
    result = panel_statistics(gram, source, held, losses(source), losses(held))
    pressures = np.array([v for i in held for v in groups[i]["pressure"]])
    seconds = np.array([v for i in held for v in groups[i]["second"]])
    p, second = pressures.mean(), seconds.mean()
    energy = np.asarray(gram)[np.ix_(held, held)].mean()
    n = sum(len(groups[i]["rows"]) for i in held)
    result.update(
        P=float(p),
        A=float(second / p),
        C=float(energy / second),
        P_over_L=float(p / result["L"]),
        Xi=float(energy / p),
        questions=n,
        self_fraction=float(second / n / energy),
        distinct_record_coherence=float((n * energy / second - 1) / (n - 1)) if n > 1 else float("nan"),
        distinct_record_q=float((n * energy - second) / (n - 1) / result["L"]) if n > 1 else float("nan"),
        max_question_energy_share=float(seconds.max() / seconds.sum()),
        median_question_A=float(np.median(seconds[pressures > 0] / pressures[pressures > 0])),
    )
    return result


def generic_rows(manifest):
    entries = list(load_results(manifest))
    rows, sources, baselines = [], {}, {}
    for task, data, path in entries:
        sources[str(path)] = digest(path)
        groups = [
            dict(
                field="live",
                split=g["split"],
                group=g["label"],
                rows=[r["row_id"] for r in g["records"]],
                losses=[r["loss"] for r in g["records"]],
                pressure=[r["pressure"] for r in g["records"]],
                second=[r["gradient_norm_sq"] for r in g["records"]],
            )
            for g in data["groups"]
            if g["domain"] == "math"
        ]
        ix = [i for i, g in enumerate(data["groups"]) if g["domain"] == "math"]
        gram = np.asarray(data["geometry"]["gram"])[np.ix_(ix, ix)]
        row = dict(
            task=task["id"],
            bank=task["bank"],
            branch=task["branch"],
            teacher=task["target"],
            stage=task["stage"],
            label=task["label"],
            field="live",
            role="all-train",
            **describe(groups, gram),
        )
        rows.append(row)
        if task["stage"] == 0:
            key = (task["model"], task["bank"], task["teacher"])
            if key in baselines:
                raise ValueError(f"Ambiguous initial baseline: {key}")
            baselines[key] = (row, groups)
        row["_key"] = (task["model"], task["bank"], task["teacher"])
        row["_groups"] = groups
    for row in rows:
        key, groups = row.pop("_key"), row.pop("_groups")
        if key not in baselines:
            row.update(
                L_initial=float("nan"),
                remaining_fraction=float("nan"),
                remaining_low=float("nan"),
                remaining_high=float("nan"),
            )
            continue
        initial, initial_groups = baselines[key]
        h = [g for g in groups if g["split"] == "heldout"]
        h0 = [g for g in initial_groups if g["split"] == "heldout"]
        if [g["rows"] for g in h] != [g["rows"] for g in h0]:
            raise ValueError("Initial/current row ordering differs")
        w = bootstrap_weights(len(h), 4000, np.random.default_rng(20261109))
        ratios = (w @ [np.mean(g["losses"]) for g in h]) / (w @ [np.mean(g["losses"]) for g in h0])
        lo, hi = np.quantile(ratios, (0.025, 0.975))
        row.update(
            L_initial=initial["L"],
            remaining_fraction=row["L"] / initial["L"],
            remaining_low=float(lo),
            remaining_high=float(hi),
        )
    return rows, sources


def crossed_rows(cells):
    rows, sources = [], {}
    for cell in json.loads(cells.read_text()):
        path = Path(cell["trajectory"])
        data = json.loads(path.read_text())
        sources[str(path)] = digest(path)
        task = data["signature"]["task"]
        bank = json.loads(Path(task["records"]).read_text())
        future = {r["group"] for r in bank if r["split"] == "train" and r["batch_index"] >= task["stage"] // 16}
        for field in ("live", "frozen"):
            for role, selected in (("all-train", None), ("future-train", future)):
                rows.append(
                    dict(
                        student=cell["student"],
                        teacher=cell["teacher"],
                        label=cell["label"],
                        bank=task["bank"],
                        stage=240,
                        field=field,
                        role=role,
                        **describe(data["groups"], data["gram"], field=field, future=selected),
                    )
                )
    return rows, sources


def factorial_contrasts(cells, *, replicates=4000):
    """Paired positive factor ratios at the same questions and B8 resamples."""
    entries = {}
    row_order = None
    for cell in json.loads(cells.read_text()):
        data = json.loads(Path(cell["trajectory"]).read_text())
        groups = data["groups"]
        indices = [i for i, g in enumerate(groups) if g["split"] == "heldout" and g["field"] == "live"]
        order = [groups[i]["rows"] for i in indices]
        if row_order is not None and order != row_order:
            raise ValueError("Factorial comparison requires paired receiver questions")
        row_order = order
        w = bootstrap_weights(len(indices), replicates, np.random.default_rng(20261109))
        loss = np.array([np.mean(groups[i]["losses"]) for i in indices])
        for field in ("live", "frozen"):
            ix = [i for i, g in enumerate(groups) if g["split"] == "heldout" and g["field"] == field]
            if [groups[i]["rows"] for i in ix] != order:
                raise ValueError("Live/frozen receiver question order differs")
            gram = np.asarray(data["gram"])[np.ix_(ix, ix)]
            p = np.array([np.mean(groups[i]["pressure"]) for i in ix])
            second = np.array([np.mean(groups[i]["second"]) for i in ix])
            point_w = np.full((1, len(ix)), 1 / len(ix))

            def statistics(weights, gram=gram, p=p, second=second, loss=loss):
                energy = np.einsum("bi,ij,bj->b", weights, gram, weights)
                pressure, moment, risk = weights @ p, weights @ second, weights @ loss
                return dict(
                    P=pressure,
                    A=moment / pressure,
                    C=energy / moment,
                    P_over_L=pressure / risk,
                    Xi=energy / pressure,
                    q=energy / risk,
                )

            entries[(cell["student"], cell["teacher"], field)] = (statistics(point_w), statistics(w))
    contrasts = []
    for field in ("live", "frozen"):
        comparisons = []
        for student in ("justrl", "polaris"):
            comparisons.append((f"teacher-J-over-P-on-{student}", [(student, "justrl", 1), (student, "polaris", -1)]))
        for teacher in ("justrl", "polaris"):
            comparisons.append((f"student-J-over-P-for-{teacher}", [("justrl", teacher, 1), ("polaris", teacher, -1)]))
        comparisons.append(
            (
                "teacher-by-student-interaction",
                [
                    ("justrl", "justrl", 1),
                    ("justrl", "polaris", -1),
                    ("polaris", "justrl", -1),
                    ("polaris", "polaris", 1),
                ],
            )
        )
        for name, terms in comparisons:
            for metric in ("P", "A", "C", "P_over_L", "Xi", "q"):
                point = np.prod([entries[(s, t, field)][0][metric][0] ** power for s, t, power in terms])
                sample = np.prod([entries[(s, t, field)][1][metric] ** power for s, t, power in terms], axis=0)
                low, high = np.quantile(sample, (0.025, 0.975))
                contrasts.append(
                    dict(
                        field=field, contrast=name, metric=metric, ratio=float(point), low=float(low), high=float(high)
                    )
                )
    return contrasts


def plot_rows(rows, out, fv, crossed):
    banks = sorted({r["bank"] for r in rows})
    for bank in banks:
        selected = (
            [r for r in rows if r["bank"] == bank and r["stage"] == 320]
            if not crossed
            else [r for r in rows if r["bank"] == bank and r["role"] == "future-train"]
        )
        if not selected:
            continue
        labels = (
            [f"{r['teacher']} on {r['student']} {r['field']}" for r in selected]
            if crossed
            else [str(i + 1) for i in range(len(selected))]
        )
        write_csv(out / (bank + "-figure-key.csv"), [dict(index=i + 1, **r) for i, r in enumerate(selected)])
        keys = ["L", "self_fraction", "D_over_L"]
        titles = ["Remaining native KL", "Self-energy fraction", "Train to heldout D / KL"]
        if not crossed:
            keys.insert(1, "remaining_fraction")
            titles.insert(1, "KL / initial KL")
        fig, axes = plt.subplots(1, len(keys), figsize=(8.5, max(3.4, 0.19 * len(selected) + 1)), sharey=True)
        fig.subplots_adjust(left=0.26 if crossed else 0.07, right=0.98, bottom=0.23, top=0.91, wspace=0.48)
        y = np.arange(len(selected))
        color = fv.HUES["vteal"]
        for ax, key, label in zip(axes, keys, titles, strict=True):
            ax.plot([r[key] for r in selected], y, "o", color=color, markersize=3, label="Estimate / 95% interval")
            if key == "D_over_L":
                ax.hlines(y, [r["low"] for r in selected], [r["high"] for r in selected], color=color)
                ax.axvline(0, color=fv.NEUTRALS["baseline"], linewidth=0.6, label="Zero transfer")
            elif key == "remaining_fraction":
                ax.hlines(
                    y, [r["remaining_low"] for r in selected], [r["remaining_high"] for r in selected], color=color
                )
            elif key == "L":
                ax.set_xscale("log")
            frame(ax, label, "Condition index")
            ax.set_yticks(y, labels)
        axes[0].invert_yaxis()
        fig.legend(*axes[-1].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False)
        save(
            fig,
            out / (bank + "-transfer"),
            "Indices: accompanying figure-key.csv. Signed intervals resample original groups.",
            fv,
            caption_fontsize=8.5,
        )


def plot_factorial(rows, out, fv):
    selected = [r for r in rows if r["role"] == "future-train"]
    fig, axes = plt.subplots(2, 3, figsize=(7.5, 5.7))
    fig.subplots_adjust(left=0.11, right=0.97, bottom=0.18, top=0.90, wspace=0.60, hspace=0.58)
    colors = fv.series_colors("paper", 2)
    for i, field in enumerate(("live", "frozen")):
        for ax, metric, title in zip(
            axes[i], ("A", "C", "D_over_L"), ("Accessibility A", "Coherence C", "Future to heldout D / KL"), strict=True
        ):
            for teacher, color in zip(("justrl", "polaris"), colors, strict=True):
                values = [
                    next(
                        r
                        for r in selected
                        if r["teacher"] == teacher and r["student"] == student and r["field"] == field
                    )
                    for student in ("justrl", "polaris")
                ]
                ax.plot([0, 1], [r[metric] for r in values], "o-", color=color, label=teacher + " teacher")
                if metric == "D_over_L":
                    ax.vlines([0, 1], [r["low"] for r in values], [r["high"] for r in values], color=color)
            ax.set_xticks([0, 1], ["JustRL", "Polaris"])
            ax.set_xlim(-0.2, 1.2)
            frame(ax, "Student training path", title)
            ax.set_title(field + " request", fontsize=9)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False)
    save(
        fig,
        out / "teacher-student-factors",
        "Step240; paired 96 heldout questions. Frozen D / KL uses current risk only as reference.",
        fv,
        caption_fontsize=8.5,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--manifest", type=Path)
    inputs.add_argument("--cells", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows, sources = generic_rows(args.manifest) if args.manifest else crossed_rows(args.cells)
    write_csv(args.out / "cross-batch.csv", rows)
    if args.cells:
        write_csv(args.out / "factorial-contrasts.csv", factorial_contrasts(args.cells))
    write_json(args.out / "sources.json", sources)
    write_json(
        args.out / "methods.json",
        dict(
            replicates=4000,
            unit="original equal-size groups",
            interval="signed basic; paired percentile for remaining KL",
            qwen_source="all fixed train-panel groups; future status not inferred from stage number",
            frozen="Native KL is reference only; frozen q is not a current-loss gradient rate",
        ),
    )
    sys.path.insert(0, str(DEFAULT_STYLE))
    import fvstyle as fv

    fv.apply_rc("paper")
    plot_rows(rows, args.out, fv, args.cells is not None)
    if args.cells:
        plot_factorial(rows, args.out, fv)
    write_json(args.out / "figure-audits.json", AUDITS)


if __name__ == "__main__":
    main()
