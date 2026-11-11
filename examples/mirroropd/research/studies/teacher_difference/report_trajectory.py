# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Remaining-risk and amplitude-invariant full-trajectory transfer analysis.

Original B8 groups are the resampling units; source and heldout draws are
independent and shared across fields/stages/teachers. Basic bootstrap bands
measure fixed-bank sampling uncertainty, not independent training runs.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .metrics import moment_summary
from .report_groups import basic_interval, bootstrap_weights
from .report_stages import DEFAULT_STYLE, digest, frame, save
from .results import write_csv
from .runtime import write_json


def statistics(data, source, held, ws, wh, field):
    groups, gram = data["groups"], np.asarray(data["gram"])
    n = len(groups) // 2
    offset = 0 if field == "live" else n
    s, h = np.asarray(source) + offset, np.asarray(held) + offset
    loss = np.array([np.mean(g["losses"]) for g in groups])
    pressure = np.array([np.mean(g["pressure"]) for g in groups])
    ss, hh, sh = gram[np.ix_(s, s)], gram[np.ix_(h, h)], gram[np.ix_(s, h)]
    a = np.einsum("bi,ij,bj->b", ws, ss, ws)
    b = np.einsum("bi,ij,bj->b", wh, hh, wh)
    d = np.einsum("bi,ij,bj->b", ws, sh, wh)
    ls, lh = ws @ loss[s], wh @ loss[h]
    ps, ph = ws @ pressure[s], wh @ pressure[h]
    return dict(
        D=d,
        D_over_L=d / lh,
        M=d / np.sqrt(ls * lh),
        pressure_transfer=d / np.sqrt(ps * ph),
        gamma=d / np.sqrt(np.maximum(a * b, 1e-30)),
        source_q=a / ls,
        held_q=b / lh,
    )


def analyze(data, *, replicates=4000):
    groups, gram = data["groups"], np.asarray(data["gram"])
    n = len(groups) // 2
    meta = dict(branch=data["signature"]["task"]["target"], stage=data["signature"]["task"]["stage"])
    bank = json.loads(Path(data["signature"]["task"]["records"]).read_text())
    batch_index = {r["group"]: r["batch_index"] for r in bank if r["split"] == "train"}
    held = [i for i, g in enumerate(groups[:n]) if g["split"] == "heldout"]
    train = [i for i, g in enumerate(groups[:n]) if g["split"] == "train"]
    future = [i for i in train if batch_index[groups[i]["group"]] >= meta["stage"] // 16]
    panels, transfer, rotations = [], [], []
    for split, indices in (("train", train), ("heldout", held)):
        for field, offset in (("live", 0), ("frozen", n)):
            ix = [i + offset for i in indices]
            p = np.mean([v for i in ix for v in groups[i]["pressure"]])
            second = np.mean([v for i in ix for v in groups[i]["second"]])
            norm = gram[np.ix_(ix, ix)].mean()
            risk = np.mean([v for i in ix for v in groups[i]["losses"]])
            summary = moment_summary(
                pressure=p, second=second, mean_norm=norm, samples=sum(len(groups[i]["rows"]) for i in ix)
            )
            panels.append(dict(**meta, split=split, field=field, L=risk, q=norm / risk, P_over_L=p / risk, **summary))
        live, frozen = indices, [i + n for i in indices]
        info = np.array([v for i in frozen for v in groups[i]["request"]]).mean(0)
        dot = info[2]
        rotations.append(
            dict(
                **meta,
                split=split,
                P_live=info[0],
                P_initial=info[1],
                initial_L=info[3],
                alpha=dot / info[1],
                logit_cosine=dot / np.sqrt(info[0] * info[1]),
                request_unexplained=1 - dot**2 / (info[0] * info[1]),
                parameter_cosine=gram[np.ix_(live, frozen)].mean()
                / np.sqrt(gram[np.ix_(live, live)].mean() * gram[np.ix_(frozen, frozen)].mean()),
            )
        )
    for role, source in (("all-train", train), ("future-train", future)):
        if not source:
            continue
        rng = np.random.default_rng(20261109)
        ws = bootstrap_weights(len(source), replicates, rng)
        wh = bootstrap_weights(len(held), replicates, rng)
        for field in ("live", "frozen"):
            point = statistics(
                data, source, held, np.ones((1, len(source))) / len(source), np.ones((1, len(held))) / len(held), field
            )
            samples = statistics(data, source, held, ws, wh, field)
            row = dict(**meta, role=role, field=field, source_groups=len(source), held_groups=len(held))
            for key in point:
                row[key] = float(point[key][0])
                row[key + "_low"], row[key + "_high"] = basic_interval(row[key], samples[key])
            # Cross independent sets make D and its risk/pressure-normalizations
            # free of within-set diagonal self-inner bias. Cosine CI uses delete-one.
            if min(len(source), len(held)) > 1:
                from scipy.stats import t

                values = []
                for side, indices in (("s", source), ("h", held)):
                    jack = []
                    for omitted in indices:
                        s = [i for i in source if i != omitted] if side == "s" else source
                        h = [i for i in held if i != omitted] if side == "h" else held
                        jack.append(
                            statistics(data, s, h, np.ones((1, len(s))) / len(s), np.ones((1, len(h))) / len(h), field)[
                                "gamma"
                            ][0]
                        )
                    values.append((len(indices) - 1) * np.var(jack))
                error = t.ppf(0.975, min(len(source), len(held)) - 1) * np.sqrt(sum(values))
                row["gamma_low"], row["gamma_high"] = row["gamma"] - error, row["gamma"] + error
            transfer.append(row)
    return panels, transfer, rotations


def decomposition(data, source, held):
    """Separate the globally fitted B0 component from its logit residual.

    Fit amplitudes on each split's logit requests, then use the joint Gram
    to propagate both components through the SAME current Jacobian.
    """
    groups, gram = data["groups"], np.asarray(data["gram"])
    n = len(groups) // 2

    def alpha(indices):
        info = np.array([v for i in indices for v in groups[i + n]["request"]]).mean(0)
        return info[2] / info[1]

    a, b = alpha(source), alpha(held)
    s0, h0 = [i + n for i in source], [i + n for i in held]
    d00 = gram[np.ix_(s0, h0)].mean()
    d = gram[np.ix_(source, held)].mean()
    parallel = a * b * d00
    source_parallel = a * (gram[np.ix_(s0, held)].mean() - b * d00)
    held_parallel = b * (gram[np.ix_(source, h0)].mean() - a * d00)
    return dict(
        alpha_source=a,
        alpha_held=b,
        D=d,
        parallel=parallel,
        source_parallel=source_parallel,
        held_parallel=held_parallel,
        residual=d - parallel - source_parallel - held_parallel,
    )


def paired_contrasts(artifacts, *, replicates=4000):
    results = []
    stages = sorted(set(stage for _, stage in artifacts))
    for stage in stages:
        if not all((branch, stage) in artifacts for branch in ("justrl", "polaris")):
            continue
        left, right = [artifacts[(branch, stage)] for branch in ("justrl", "polaris")]
        n = len(left["groups"]) // 2
        assert [(g["rows"], g["split"]) for g in left["groups"]] == [(g["rows"], g["split"]) for g in right["groups"]]
        source = [i for i, g in enumerate(left["groups"][:n]) if g["split"] == "train"]
        held = [i for i, g in enumerate(left["groups"][:n]) if g["split"] == "heldout"]
        bank = json.loads(Path(left["signature"]["task"]["records"]).read_text())
        batch_index = {r["group"]: r["batch_index"] for r in bank if r["split"] == "train"}
        future = [i for i in source if batch_index[left["groups"][i]["group"]] >= stage // 16]
        for role, selected in (("all-train", source), ("future-train", future)):
            if not selected:
                continue
            rng = np.random.default_rng(20261109)
            ws, wh = bootstrap_weights(len(selected), replicates, rng), bootstrap_weights(len(held), replicates, rng)
            samples = [statistics(d, selected, held, ws, wh, "live") for d in (left, right)]
            points = [
                statistics(
                    d,
                    selected,
                    held,
                    np.ones((1, len(selected))) / len(selected),
                    np.ones((1, len(held))) / len(held),
                    "live",
                )
                for d in (left, right)
            ]
            for key in ("D_over_L", "M", "pressure_transfer"):
                point = points[0][key][0] - points[1][key][0]
                low, high = basic_interval(point, samples[0][key] - samples[1][key])
                results.append(
                    dict(
                        stage=stage,
                        role=role,
                        metric=key,
                        contrast="JustRL minus Polaris",
                        value=point,
                        low=low,
                        high=high,
                    )
                )
    return results


def plot(rows, out, fv, *, filename, keys, labels, field="live", split=None, role=None):
    fig, axes = plt.subplots(1, len(keys), figsize=(min(6.8, 3.4 * len(keys)), 2.8), squeeze=False)
    fig.subplots_adjust(left=0.22 if len(keys) == 1 else 0.09, right=0.98, bottom=0.27, top=0.86, wspace=0.55)
    colors = fv.series_colors("paper", 2)
    for branch, color in zip(("justrl", "polaris"), colors, strict=True):
        selected = sorted(
            [
                r
                for r in rows
                if r["branch"] == branch
                and r.get("field", field) == field
                and (split is None or r.get("split") == split)
                and (role is None or r.get("role") == role)
            ],
            key=lambda r: r["stage"],
        )
        for ax, key, label in zip(axes[0], keys, labels, strict=True):
            x, y = [r["stage"] for r in selected], [r[key] for r in selected]
            ax.plot(x, y, "o-", color=color, label=branch, markersize=3)
            if selected and key + "_low" in selected[0]:
                ax.fill_between(
                    x,
                    [r[key + "_low"] for r in selected],
                    [r[key + "_high"] for r in selected],
                    color=color,
                    alpha=0.16,
                )
            frame(ax, "Optimizer updates", label)
            ax.set_xticks(sorted(set(r["stage"] for r in rows if role is None or r.get("role") == role)))
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False)
    save(fig, out / filename, "H1024 trajectories; B8 cluster intervals where shaded.", fv, caption_fontsize=7)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(DEFAULT_STYLE))
    import fvstyle as fv

    from .report_stages import AUDITS

    fv.apply_rc("paper")
    out = args.root / "analysis"
    out.mkdir(exist_ok=True)
    panels, transfers, rotations, sources = [], [], [], {}
    artifacts, components = {}, []
    for path in sorted(args.root.glob("*-s[0-9][0-9][0-9].json")):
        data = json.loads(path.read_text())
        a, b, c = analyze(data)
        task = data["signature"]["task"]
        artifacts[(task["target"], task["stage"])] = data
        n = len(data["groups"]) // 2
        source = [i for i, g in enumerate(data["groups"][:n]) if g["split"] == "train"]
        held = [i for i, g in enumerate(data["groups"][:n]) if g["split"] == "heldout"]
        components.append(dict(branch=task["target"], stage=task["stage"], **decomposition(data, source, held)))
        panels.extend(a)
        transfers.extend(b)
        rotations.extend(c)
        sources[str(path)] = digest(path)
    factor_rows = []
    for stage in sorted(set(r["stage"] for r in panels)):
        pair = {
            r["branch"]: r for r in panels if r["stage"] == stage and r["split"] == "heldout" and r["field"] == "live"
        }
        if len(pair) == 2:
            a, b = pair["justrl"], pair["polaris"]
            factor_rows.append(
                dict(
                    stage=stage,
                    pressure_over_risk_ratio=a["P_over_L"] / b["P_over_L"],
                    A_ratio=a["A"] / b["A"],
                    C_ratio=a["C"] / b["C"],
                    q_ratio=a["q"] / b["q"],
                )
            )
    write_csv(out / "factor-comparison.csv", factor_rows)
    write_csv(out / "panels.csv", panels)
    write_csv(out / "transfer.csv", transfers)
    for row in rotations:
        supplemental = args.root / f"{row['branch']}-s{row['stage']:03d}.alignment.json"
        if supplemental.exists():
            payload = json.loads(supplemental.read_text())
            info = np.array([r["moments"] for r in payload["records"] if r["split"] == row["split"]]).mean(0)
            np.testing.assert_allclose(
                info[:3], [row["P_live"], row["P_initial"], row["alpha"] * row["P_initial"]], rtol=1e-6, atol=1e-12
            )
            row.update(
                local_cosine=info[2] / info[3],
                perpendicular_fraction=info[4] / info[0],
                positive_rescale_residual=info[5] / info[0],
            )
            sources[str(supplemental)] = digest(supplemental)
    write_csv(out / "requests.csv", rotations)
    write_csv(out / "paired-contrasts.csv", paired_contrasts(artifacts))
    write_csv(out / "transfer-components.csv", components)
    write_json(out / "sources.json", sources)
    plot(
        panels,
        out,
        fv,
        filename="risk",
        keys=["L", "P", "q"],
        labels=["Remaining KL / H", "Pressure P", "Gradient energy / KL"],
        split="heldout",
    )
    plot(
        panels,
        out,
        fv,
        filename="pac",
        keys=["P_over_L", "A", "C"],
        labels=["P / L", "Accessibility A", "Coherence C"],
        split="heldout",
    )
    for role in ("all-train", "future-train"):
        plot(
            transfers,
            out,
            fv,
            filename="transfer-" + role,
            keys=["D_over_L", "M", "gamma"],
            labels=["D / held KL", r"$D/\sqrt{L_T L_H}$", "Train-held cosine"],
            role=role,
        )
        plot(
            transfers,
            out,
            fv,
            filename="pressure-" + role,
            keys=["pressure_transfer"],
            labels=[r"$D/\sqrt{P_T P_H}$"],
            role=role,
        )
    late = [row for row in transfers if row["stage"] > 0]
    if late:
        plot(
            late,
            out,
            fv,
            filename="transfer-late",
            keys=["D_over_L", "M", "gamma"],
            labels=["D / held KL", r"$D/\sqrt{L_T L_H}$", "Train-held cosine"],
            role="all-train",
        )
    plot(
        rotations,
        out,
        fv,
        filename="request-directions",
        keys=["alpha", "logit_cosine", "parameter_cosine"],
        labels=["Best amplitude fit to B0", "Logit request cosine", "Parameter mean cosine"],
        split="heldout",
    )
    plot(
        panels,
        out,
        fv,
        filename="frozen-pac",
        keys=["A", "C", "Xi"],
        labels=["Frozen accessibility A", "Frozen coherence C", "Frozen Xi"],
        field="frozen",
        split="heldout",
    )
    if rotations and all("local_cosine" in row for row in rotations):
        plot(
            rotations,
            out,
            fv,
            filename="local-directions",
            keys=["local_cosine", "perpendicular_fraction"],
            labels=["Token-local weighted cosine", "Perpendicular request fraction"],
            split="heldout",
        )
    write_json(out / "figure-audits.json", AUDITS)


if __name__ == "__main__":
    main()
