# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Response decomposition with paired B8 uncertainty and separate probe error.

Semantic ledger: each point is a teacher/checkpoint on the same frozen Math
panel; shaded bands resample original B8 groups, not tokens or sign probes.
Geometric ledger: shared update axes; energy ratios are dimensionless, A_token
uses parameter-gradient energy per logit pressure. Signed transfer terms share
the native remaining heldout KL denominator and sum to native transfer.
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


def positive_interval(samples):
    """Percentile intervals preserve positive support for skewed energy ratios."""
    return [float(v) for v in np.quantile(samples, (0.025, 0.975))]


def arrays(data, split):
    records = {row["row_id"]: row for row in data["records"]}
    groups = [g for g in data["groups"] if g["field"] == "live" and g["split"] == split]
    cap, k = data["signature"]["task"]["cap"], data["signature"]["probes"]
    moments, native, parallel, cross, probes = [], [], [], [], []
    for group in groups:
        rows = [records[key] for key in group["rows"]]
        moments.append(np.mean([r["moments"] for r in rows], axis=0))
        native.append(np.mean([r["norm_sq"]["live"] for r in rows]))
        parallel.append(np.mean([r["norm_sq"]["parallel"] for r in rows]))
        cross.append(np.mean([r["native_parallel_inner"] for r in rows]))
        probes.append([cap * np.mean([r["norm_sq"][f"sign-{j}"] for r in rows]) for j in range(k)])
    return dict(
        moments=np.asarray(moments),
        native=np.asarray(native),
        parallel=np.asarray(parallel),
        cross=np.asarray(cross),
        probes=np.asarray(probes),
        groups=groups,
    )


def factors(data, split, weights):
    a = arrays(data, split)
    loss, p, p0, pp, perp, reverse, _ = (weights @ a["moments"]).T
    native, parallel, cross = [weights @ a[key] for key in ("native", "parallel", "cross")]
    u = weights @ a["probes"].mean(1)
    cap = data["signature"]["task"]["cap"]
    return dict(
        L=loss,
        P=p,
        A=native / p,
        U=u,
        A_token=u / p,
        R_agg=native / u,
        aggregation_over_diagonal=cap * native / u,
        diagonal=u / cap,
        offdiagonal=native - u / cap,
        second=native,
        parallel_second=parallel,
        perpendicular_second=native + parallel - 2 * cross,
        parameter_mixed=2 * (cross - parallel),
        parameter_energy_retained=native / (native + 2 * parallel - 2 * cross),
        parameter_component_cosine=np.divide(
            cross - parallel,
            np.sqrt(np.maximum(parallel * (native + parallel - 2 * cross), 0)),
            out=np.full_like(native, np.nan),
            where=(perp > 0) & (parallel * (native + parallel - 2 * cross) > 0),
        ),
        perpendicular_pressure=perp / p,
        reverse_pressure=reverse / p,
        parallel_A=parallel / np.maximum(pp, 1e-300),
        perpendicular_A=np.divide(
            native + parallel - 2 * cross, perp, out=np.full_like(native, np.nan), where=perp > 0
        ),
        P0=p0,
    )


def component_transfer(data, role, ws=None, wh=None):
    groups, gram = data["groups"], np.asarray(data["gram"])
    task = data["signature"]["task"]
    bank = json.loads(Path(task["records"]).read_text())
    batch_index = {r["group"]: r["batch_index"] for r in bank if r["split"] == "train"}
    source = [
        i
        for i, g in enumerate(groups)
        if g["field"] == "live"
        and g["split"] == "train"
        and (role == "all-train" or batch_index[g["group"]] >= task["stage"] // 16)
    ]
    held = [i for i, g in enumerate(groups) if g["field"] == "live" and g["split"] == "heldout"]
    if not source or not held:
        return {}
    lookup = {(g["group"], g["split"], g["field"]): i for i, g in enumerate(groups)}
    # Indices here follow the original interleaved group sequence.
    mapped = {i: lookup[(g["group"], g["split"], "parallel")] for i, g in enumerate(groups) if g["field"] == "live"}
    if ws is None:
        ws = np.ones((1, len(source))) / len(source)
        wh = np.ones((1, len(held))) / len(held)
    records = {r["row_id"]: r for r in data["records"]}
    lh = wh @ np.array([np.mean([records[key]["moments"][0] for key in groups[i]["rows"]]) for i in held])

    def inner(h, s):
        return np.einsum("bi,ij,bj->b", wh, gram[np.ix_(h, s)], ws)

    hp, sp = [mapped[i] for i in held], [mapped[i] for i in source]
    native, pp = inner(held, source), inner(hp, sp)
    p_perp, perp_p = inner(hp, source) - pp, inner(held, sp) - pp
    perpendicular = native - pp - p_perp - perp_p
    absolute = np.abs(pp) + np.abs(p_perp) + np.abs(perp_p) + np.abs(perpendicular)
    return {
        "native": native / lh,
        "parallel_parallel": pp / lh,
        "parallel_perpendicular": p_perp / lh,
        "perpendicular_parallel": perp_p / lh,
        "perpendicular_perpendicular": perpendicular / lh,
        "parallel_source": (pp + perp_p) / lh,
        "perpendicular_source": (p_perp + perpendicular) / lh,
        "signed_retention": native / np.maximum(absolute, 1e-300),
    }


def analyze(data, replicates=4000):
    task = data["signature"]["task"]
    meta = dict(branch=task["target"], stage=task["stage"])
    rows, mc, samples = [], [], {}
    for split in ("train", "heldout"):
        a = arrays(data, split)
        if not len(a["groups"]):
            continue
        n = len(a["groups"])
        w = bootstrap_weights(n, replicates, np.random.default_rng(20261109))
        point = factors(data, split, np.ones((1, n)) / n)
        sample = factors(data, split, w)
        samples[split] = sample
        row = dict(**meta, split=split, groups=n)
        for key, value in point.items():
            row[key] = float(value[0])
            interval = (
                basic_interval(row[key], sample[key])
                if key in ("offdiagonal", "parameter_mixed", "parameter_component_cosine")
                else positive_interval(sample[key])
            )
            row[key + "_low"], row[key + "_high"] = interval
        probe_u = a["probes"].mean(0)
        row["U_mc_relative_se"] = float(probe_u.std(ddof=1) / np.sqrt(len(probe_u)) / probe_u.mean())
        individual_pressure = np.array([r["moments"][1] for r in data["records"] if r["split"] == split])
        row["pressure_effective_questions"] = float(individual_pressure.sum() ** 2 / (individual_pressure**2).sum())
        row["maximum_pressure_share"] = float(individual_pressure.max() / individual_pressure.sum())
        individual = [r for r in data["records"] if r["split"] == split]
        token_energy = np.array(
            [np.mean([r["norm_sq"][f"sign-{j}"] for j in range(data["signature"]["probes"])]) for r in individual]
        )
        trajectory_energy = np.array([r["norm_sq"]["live"] for r in individual])
        row["maximum_token_energy_share"] = float(token_energy.max() / token_energy.sum())
        row["token_energy_effective_questions"] = float(token_energy.sum() ** 2 / (token_energy**2).sum())
        row["maximum_trajectory_energy_share"] = float(trajectory_energy.max() / trajectory_energy.sum())
        rows.append(row)
        for k in (2, 4, 8, 16, 32):
            if k <= len(probe_u):
                mc.append(
                    dict(
                        **meta,
                        split=split,
                        probes=k,
                        U=float(probe_u[:k].mean()),
                        relative_se=float(probe_u[:k].std(ddof=1) / np.sqrt(k) / probe_u[:k].mean()),
                    )
                )
    transfer = []
    train = arrays(data, "train")["groups"]
    bank = json.loads(Path(task["records"]).read_text())
    batch_index = {r["group"]: r["batch_index"] for r in bank if r["split"] == "train"}
    for role in ("all-train", "future-train"):
        n = sum(role == "all-train" or batch_index[g["group"]] >= task["stage"] // 16 for g in train)
        nh = len(arrays(data, "heldout")["groups"])
        if not n or not nh:
            continue
        rng = np.random.default_rng(20261109)
        ws, wh = bootstrap_weights(n, replicates, rng), bootstrap_weights(nh, replicates, rng)
        point, sample = component_transfer(data, role), component_transfer(data, role, ws, wh)
        for key, value in point.items():
            low, high = basic_interval(value[0], sample[key])
            transfer.append(dict(**meta, role=role, component=key, value=float(value[0]), low=low, high=high))
    return rows, transfer, mc, samples


def frozen_factors(data, frozen, replicates=4000):
    """Compare native and initial requests through the same current Jacobian."""
    assert data["signature"]["task"] == frozen["signature"]["task"]
    task = data["signature"]["task"]
    results = []
    by_id = {r["row_id"]: r for r in frozen["records"]}
    for split in ("train", "heldout"):
        a = arrays(data, split)
        moments, energy, probes = [], [], []
        for group in a["groups"]:
            rows = [by_id[key] for key in group["rows"]]
            moments.append(np.mean([r["moments"][2] for r in rows]))
            energy.append(np.mean([r["norm_sq"]["frozen"] for r in rows]))
            probes.append(
                [
                    task["cap"] * np.mean([r["norm_sq"][f"frozen-sign-{j}"] for r in rows])
                    for j in range(frozen["signature"]["probes"])
                ]
            )
        moments, energy, probes = np.array(moments), np.array(energy), np.array(probes)
        weights = bootstrap_weights(len(moments), replicates, np.random.default_rng(20261109))
        outputs = []
        for w in (np.ones((1, len(moments))) / len(moments), weights):
            u, p, second = w @ probes.mean(1), w @ moments, w @ energy
            live = factors(data, split, w)
            outputs.append(
                dict(
                    A_token_frozen=u / p,
                    A_frozen=second / p,
                    R_agg_frozen=second / u,
                    live_over_frozen_token=live["A_token"] / (u / p),
                )
            )
        point, sample = outputs
        row = dict(branch=task["target"], stage=task["stage"], split=split)
        for key, value in point.items():
            row[key] = float(value[0])
            row[key + "_low"], row[key + "_high"] = positive_interval(sample[key])
        m = probes.mean(0)
        row["U_mc_relative_se"] = float(m.std(ddof=1) / np.sqrt(len(m)) / m.mean())
        results.append(row)
    return results


def plot_factors(rows, out, fv, name, keys, labels, *, logarithmic=False):
    fig, axes = plt.subplots(1, len(keys), figsize=(6.8, 3.1), squeeze=False)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.27, top=0.85, wspace=0.62)
    for branch, color in zip(("justrl", "polaris"), fv.series_colors("paper", 2), strict=True):
        selected = sorted(
            [r for r in rows if r["branch"] == branch and r["split"] == "heldout"], key=lambda r: r["stage"]
        )
        for ax, key, label in zip(axes[0], keys, labels, strict=True):
            ax.plot([r["stage"] for r in selected], [r[key] for r in selected], "o-", label=branch, color=color)
            ax.fill_between(
                [r["stage"] for r in selected],
                [r[key + "_low"] for r in selected],
                [r[key + "_high"] for r in selected],
                color=color,
                alpha=0.16,
            )
            if len(selected) == 1:
                ax.vlines([selected[0]["stage"]], selected[0][key + "_low"], selected[0][key + "_high"], color=color)
            frame(ax, "Optimizer updates", label)
            if logarithmic:
                ax.set_yscale("log")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False)
    save(fig, out / name, "96 heldout trajectories; B8 intervals; probe error separate.", fv, caption_fontsize=8)


def question_sensitivity(data):
    """Finite-panel leave-one-question-out energies; never trim the estimand."""
    cap, k = data["signature"]["task"]["cap"], data["signature"]["probes"]
    rows = [r for r in data["records"] if r["split"] == "heldout"]
    p = np.array([r["moments"][1] for r in rows])
    u = cap * np.array([np.mean([r["norm_sq"][f"sign-{j}"] for j in range(k)]) for r in rows])
    e = np.array([r["norm_sq"]["live"] for r in rows])
    output = []
    for i, row in enumerate(rows):
        pi, ui, ei = p.sum() - p[i], u.sum() - u[i], e.sum() - e[i]
        output.append(
            dict(
                branch=data["signature"]["task"]["target"],
                stage=data["signature"]["task"]["stage"],
                row_id=row["row_id"],
                token_energy_share=u[i] / u.sum(),
                token_response=u[i] / p[i],
                trajectory_response=e[i] / p[i],
                trajectory_energy_share=e[i] / e.sum(),
                A_without=ei / pi,
                A_token_without=ui / pi,
                R_agg_without=ei / ui,
            )
        )
    return output


def plot_concentration(sensitivity, out, fv):
    """A ranked curve element is a question's share of fixed-panel energy."""
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.28, top=0.85, wspace=0.32)
    for branch, color in zip(("justrl", "polaris"), fv.series_colors("paper", 2), strict=True):
        rows = [r for r in sensitivity if r["branch"] == branch and r["stage"] == 240]
        for ax, metric in zip(axes, ("token_energy_share", "trajectory_energy_share"), strict=True):
            if rows:
                energy = np.sort([r[metric] for r in rows])[::-1]
                ax.plot(np.arange(1, len(rows) + 1), 100 * np.cumsum(energy), color=color, label=branch)
    frame(axes[0], "Questions ranked by energy", "Cumulative token energy (%)")
    frame(axes[1], "Questions ranked by energy", "Cumulative trajectory energy (%)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False)
    save(
        fig,
        out / "energy-concentration",
        "Update240 heldout; rankings are separate for each teacher and energy.",
        fv,
        caption_fontsize=8,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    out = args.root / "analysis"
    out.mkdir(exist_ok=True)
    rows, transfer, mc, sources, paired, frozen_rows = [], [], [], {}, [], []
    by_stage = {}
    sensitivity = []
    for path in sorted(args.root.glob("*-s[0-9][0-9][0-9].json")):
        data = json.loads(path.read_text())
        a, b, c, samples = analyze(data)
        sensitivity.extend(question_sensitivity(data))
        rows.extend(a)
        transfer.extend(b)
        mc.extend(c)
        task = data["signature"]["task"]
        by_stage[(task["target"], task["stage"])] = (a, samples)
        sources[str(path)] = digest(path)
        frozen_path = args.root.parent / "frozen-norms" / f"{task['target']}-s{task['stage']:03d}.norms.json"
        if frozen_path.exists():
            frozen = json.loads(frozen_path.read_text())
            if frozen["signature"]["task"] == data["signature"]["task"]:
                frozen_rows.extend(frozen_factors(data, frozen))
                sources[str(frozen_path)] = digest(frozen_path)
    for (branch, stage), (a, samples) in by_stage.items():
        if branch != "justrl" or ("polaris", stage) not in by_stage:
            continue
        b, other = by_stage[("polaris", stage)]
        left, right = [next(r for r in item if r["split"] == "heldout") for item in (a, b)]
        for key in ("A", "A_token", "R_agg"):
            point = left[key] / right[key]
            low, high = positive_interval(samples["heldout"][key] / other["heldout"][key])
            paired.append(dict(stage=stage, metric=key, ratio=point, low=low, high=high))
    for name, values in (
        ("factors", rows),
        ("transfer", transfer),
        ("probe-convergence", mc),
        ("paired-ratios", paired),
        ("frozen-factors", frozen_rows),
        ("question-sensitivity", sensitivity),
    ):
        if values:
            write_csv(out / (name + ".csv"), values)
    write_json(
        out / "methods.json",
        dict(
            resampling="4000 paired B8 draws; source and heldout independent",
            positive_factors="percentile intervals; skewed positive ratios",
            signed_transfer="basic intervals; preserve signed values",
            probe_error="independent Monte Carlo SE; repeated probes do not add questions",
        ),
    )
    write_json(out / "sources.json", sources)
    sys.path.insert(0, str(DEFAULT_STYLE))
    import fvstyle as fv

    fv.apply_rc("paper")
    plot_concentration(sensitivity, out, fv)
    plot_factors(
        rows,
        out,
        fv,
        "accessibility",
        ["A", "A_token", "R_agg"],
        ["Trajectory A", "Token response A", "Aggregation ratio"],
    )
    plot_factors(
        rows,
        out,
        fv,
        "token-aggregation",
        ["aggregation_over_diagonal", "offdiagonal"],
        ["Energy / token diagonal", "Token off-diagonal energy"],
    )
    plot_factors(
        rows,
        out,
        fv,
        "component-response",
        ["parallel_A", "perpendicular_A"],
        ["Parallel response / pressure", "Perpendicular response / pressure"],
    )
    plot_factors(
        rows,
        out,
        fv,
        "component-energy",
        ["parallel_second", "perpendicular_second", "parameter_mixed"],
        ["Parallel parameter energy", "Perpendicular parameter energy", "Mixed parameter energy"],
    )
    for role in ("all-train", "future-train"):
        components = (
            "parallel_parallel",
            "parallel_perpendicular",
            "perpendicular_parallel",
            "perpendicular_perpendicular",
        )
        fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.8), sharey=True)
        fig.subplots_adjust(left=0.10, right=0.98, bottom=0.40, top=0.96, wspace=0.28)
        colors = fv.series_colors("paper", 4)
        for ax, branch in zip(axes, ("justrl", "polaris"), strict=True):
            for i, (component, color) in enumerate(zip(components, colors, strict=True)):
                selected = sorted(
                    [
                        r
                        for r in transfer
                        if r["branch"] == branch
                        and r["role"] == role
                        and r["component"] == component
                        and r["stage"] > 0
                    ],
                    key=lambda r: r["stage"],
                )
                if not selected:
                    continue
                x = np.arange(len(selected)) + (i - 1.5) * 0.16
                y = np.array([r["value"] for r in selected])
                ax.plot(
                    x,
                    y,
                    "o-",
                    color=color,
                    markersize=3,
                    label=component.replace("perpendicular", "perp").replace("parallel", "par"),
                )
                ax.fill_between(x, [r["low"] for r in selected], [r["high"] for r in selected], color=color, alpha=0.14)
                if len(selected) == 1:
                    ax.vlines(x, [r["low"] for r in selected], [r["high"] for r in selected], color=color)
                ax.set_xticks(np.arange(len(selected)), [r["stage"] for r in selected])
            ax.axhline(0, color=fv.NEUTRALS["baseline"], linewidth=0.6)
            frame(ax, branch + " updates", "Component / native held KL")
        fig.legend(
            *axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.07), ncol=2, frameon=False
        )
        save(
            fig,
            out / ("transfer-" + role),
            "Updates > 0. First index: receiver; second: source. Terms sum to native.",
            fv,
            caption_fontsize=8,
        )
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0))
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.27, top=0.85, wspace=0.38)
    for branch, color in zip(("justrl", "polaris"), fv.series_colors("paper", 2), strict=True):
        selected = sorted(
            [r for r in mc if r["branch"] == branch and r["stage"] == 240 and r["split"] == "heldout"],
            key=lambda r: r["probes"],
        )
        if not selected:
            continue
        x = [r["probes"] for r in selected]
        axes[0].plot(x, [r["U"] / selected[-1]["U"] for r in selected], "o-", color=color, label=branch)
        axes[1].plot(x, [100 * r["relative_se"] for r in selected], "o-", color=color, label=branch)
    frame(axes[0], "Sign probes per question", "U / final-probe estimate")
    frame(axes[1], "Sign probes per question", "Monte Carlo relative SE (%)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False)
    save(
        fig,
        out / "probe-convergence",
        "Update240, 96 heldout questions; probe count does not add questions.",
        fv,
        caption_fontsize=8,
    )
    write_json(out / "figure-audits.json", AUDITS)


if __name__ == "__main__":
    main()
