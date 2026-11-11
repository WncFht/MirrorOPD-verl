"""Join exact parameter fingerprints to every original explorer held-out curve.

Teacher matrices and checkpoints are dependent observations. Regressions split
by teacher identity, keeping input modes together. Native cross-size weight
spectra are descriptive; delta predictors require compatible coordinates.
Plots use family color and added-stage marker; axes are common within figures.
"""

import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
OUT = ROOT / "spectrum-expanded-v4"
sys.path.insert(0, str(REPO.parent / "dotfiles/.agents/skills/figure-viz/assets/mpl"))
import fvstyle as fv  # noqa: E402

FEATURES = [
    "norm_over_base",
    "stable_rank_fraction",
    "rank90_fraction",
    "top1pct",
    "top5pct",
    "kappa_spec",
    "rho_sigma",
]
GROUPS = ["r1", "r7", "q3-1.7", "q3-1.7-nonthinking", "llama3-3b-off"]
TRANSFORMS = {"norm_over_base", "kappa_spec", "rho_sigma"}


def dump(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, allow_nan=False))


def write_csv(name, rows):
    if not rows:
        return
    with (OUT / name).open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def med(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    return float(np.median(v)) if v else None


def aggregate(rows):
    scalars = ["frobenius", "stable_rank", "effective_rank", "rank90", *FEATURES, "top16"]
    return {k: med(rows, k) for k in scalars}


def fingerprint():
    inventory = json.loads((OUT / "inventory.json").read_text())["runs"]
    curves = json.loads((OUT / "explorer-data.json").read_text())["runs"]
    profiles = []
    joined = []
    risk_checks = []
    for r in inventory:
        files = sorted((OUT / "runs" / r["run_id"]).glob("*.json"))
        assert len(files) == 6, (r["run_id"], len(files))
        matrices = [json.loads(p.read_text()) for p in files]
        compatible = all(m["compatible"] for m in matrices)
        tid = hashlib.sha256(r["teacher_path"].encode()).hexdigest()[:16]
        teacher_full = [json.loads(p.read_text()) for p in (OUT / "weights" / tid).glob("*.json")]
        assert len(teacher_full) == 6
        profile = dict(
            run_id=r["run_id"],
            student=r["student"],
            teacher=r["teacher"],
            label=r["label"],
            teacher_path=r["teacher_path"],
            lineage=r["lineage"],
            added_stage=r["added_stage"],
            compatible=compatible,
            budget=r["budget"],
            teacher_full=aggregate(teacher_full),
            teacher_delta=aggregate([m["teacher_delta"] for m in matrices]) if compatible else None,
            student_stages={},
        )
        steps = sorted({s["step"] for m in matrices for s in m["stages"]})
        for step in steps:
            stages = [next(s for s in m["stages"] if s["step"] == step) for m in matrices]
            assert len(stages) == 6
            profile["student_stages"][str(step)] = {
                "delta": aggregate([s["delta"] for s in stages]),
                "missing": aggregate([s["missing"] for s in stages]) if compatible else None,
                "update_cosine": med(stages, "update_cosine"),
                "left_overlap16": med(stages, "left_overlap16"),
                "right_overlap16": med(stages, "right_overlap16"),
            }
        profiles.append(profile)
        curve = next(c for c in curves if c["run_id"] == r["run_id"])
        for panel, points in curve["panels"].items():
            keyed = {p["step"]: p for p in points}
            for step in [64, 320]:
                if step not in keyed:
                    continue
                start, p = keyed[0], keyed[step]
                if panel != "Training scored":
                    groups = {
                        "Math": ["M01", "M02", "M03", "M04"],
                        "Code": ["Code01", "Code02"],
                        "IF": ["IF01", "IF02"],
                        "All held-out": ["M01", "M02", "M03", "M04", "Code01", "Code02", "IF01", "IF02"],
                    }[panel]
                    for checkstep, expected in [(0, start["rkl"]), (step, p["rkl"])]:
                        rawpath = Path(r["initial_source"]).parent / f"fixed-{checkstep:03d}.json"
                        records = json.loads(rawpath.read_text())
                        selected = [x for x in records if x["group"] in groups]
                        actual = sum(x["stats"]["all"]["rkl"] for x in selected) / (len(selected) * 1024)
                        assert np.isclose(actual, expected, rtol=1e-12, atol=1e-14), (r["run_id"], panel, checkstep)
                        risk_checks.append(
                            dict(
                                run_id=r["run_id"],
                                panel=panel,
                                step=checkstep,
                                source=str(rawpath),
                                sha256=hashlib.sha256(rawpath.read_bytes()).hexdigest(),
                                rkl=actual,
                                n=len(selected),
                            )
                        )
                late = keyed.get(step - 64) if step >= 128 else None
                train = {p["step"]: p for p in curve["panels"]["Training scored"]}
                row = dict(
                    run_id=r["run_id"],
                    student=r["student"],
                    teacher=r["teacher"],
                    teacher_path=r["teacher_path"],
                    label=r["label"],
                    added_stage=r["added_stage"],
                    lineage=r["lineage"],
                    compatible=compatible,
                    step=step,
                    panel=panel,
                    initial_rkl=start["rkl"],
                    final_rkl=p["rkl"],
                    remaining=p["rkl"] / start["rkl"],
                    absolute_decrease=start["rkl"] - p["rkl"],
                    log_absorption=p["log_absorption"],
                    train_initial=train[0]["rkl"],
                    train_final=train[step]["rkl"],
                    train_log_absorption=train[step]["log_absorption"],
                    transfer_gap=train[step]["log_absorption"] - p["log_absorption"],
                    late_absolute_decrease=late["rkl"] - p["rkl"] if late else None,
                    late_relative_decrease=1 - p["rkl"] / late["rkl"] if late else None,
                    late_log_decrease=p["log_absorption"] - late["log_absorption"] if late else None,
                )
                row.update(
                    {"weight_" + k: profile["teacher_full"][k] for k in ["stable_rank", "top16", "rank90_fraction"]}
                )
                for feature in FEATURES:
                    row[feature] = profile["teacher_delta"][feature] if compatible else None
                stage = profile["student_stages"].get(str(step))
                row["student_update_norm"] = stage["delta"]["norm_over_base"] if stage else None
                row["student_top1pct"] = stage["delta"]["top1pct"] if stage else None
                row["missing_norm"] = stage["missing"]["norm_over_base"] if stage and compatible else None
                row["update_cosine"] = stage["update_cosine"] if stage else None
                joined.append(row)
    dump("raw-risk-verification.json", risk_checks)
    dump("model-profiles.json", profiles)
    write_csv("heldout-feature-join.csv", joined)
    dump(
        "coverage-summary.json",
        dict(
            runs=len(profiles),
            teachers=len({p["teacher_path"] for p in profiles}),
            compatible_runs=sum(p["compatible"] for p in profiles),
            compatible_teacher_pairs=len(
                {(r["base"], r["teacher_path"]) for r, p in zip(inventory, profiles, strict=True) if p["compatible"]}
            ),
            student_stage_observations=sum(len(p["student_stages"]) for p in profiles),
            full_weight_checkpoints=len(list((OUT / "weights").iterdir())),
            matrices=sum(1 for p in (OUT / "weights").glob("*/*.json")),
        ),
    )
    return profiles, joined


def spectral_neighbors(profiles):
    groups = {}
    for p in profiles:
        if not p["compatible"]:
            continue
        matrices = [json.loads(f.read_text()) for f in (OUT / "runs" / p["run_id"]).glob("*.json")]
        key = matrices[0]["base"]
        groups.setdefault(key, {})[p["teacher_path"]] = (p, {m["tensor"]: m for m in matrices})
    neighbors = []
    for base, members in groups.items():
        for teacher, (profile, matrices) in members.items():
            distances = []
            for other, (q, qmatrices) in members.items():
                if teacher == other:
                    continue
                ds = []
                for key, m in matrices.items():
                    a = np.array(m["teacher_delta"]["singular_values"])
                    b = np.array(qmatrices[key]["teacher_delta"]["singular_values"])
                    ds.append(float(np.linalg.norm(a / np.linalg.norm(a) - b / np.linalg.norm(b))))
                distances.append(
                    dict(
                        teacher=q["teacher"],
                        label=q["label"],
                        added_stage=q["added_stage"],
                        mean_normalized_spectrum_distance=float(np.mean(ds)),
                        norm_ratio=q["teacher_delta"]["norm_over_base"] / profile["teacher_delta"]["norm_over_base"],
                    )
                )
            neighbors.append(
                dict(
                    base=base,
                    teacher=profile["teacher"],
                    label=profile["label"],
                    neighbors=sorted(distances, key=lambda x: x["mean_normalized_spectrum_distance"]),
                )
            )
    dump("spectral-neighbors.json", neighbors)


def transform(feature, v):
    return np.log10(v) if feature in TRANSFORMS else v


def partial_rank(rows, feature, target):
    if len(rows) < 6:
        return None
    x = rankdata([r[feature] for r in rows])
    y = rankdata([r[target] for r in rows])
    groups = sorted({r["student"] for r in rows})
    z = np.array([[1, np.log10(r["initial_rkl"]), *[int(r["student"] == g) for g in groups[1:]]] for r in rows])
    if len(rows) - np.linalg.matrix_rank(z) < 3:
        return None
    xr = x - z @ np.linalg.lstsq(z, x, rcond=None)[0]
    yr = y - z @ np.linalg.lstsq(z, y, rcond=None)[0]
    return float(np.corrcoef(xr, yr)[0, 1]) if np.std(xr) > 1e-10 and np.std(yr) > 1e-10 else None


def associations(joined):
    results = []
    for step in [64, 320]:
        for panel in ["All held-out", "Math", "Code", "IF"]:
            for group in [*GROUPS, "pooled_compatible"]:
                rows = [
                    r
                    for r in joined
                    if r["step"] == step
                    and r["panel"] == panel
                    and r["compatible"]
                    and (group == "pooled_compatible" or r["student"] == group)
                ]
                if len(rows) < 4:
                    continue
                for feature in FEATURES:
                    for target in ["final_rkl", "remaining", "absolute_decrease", "late_log_decrease"]:
                        subset = [r for r in rows if r[feature] is not None and r[target] is not None]
                        if len(subset) < 4:
                            continue
                        x = [r[feature] for r in subset]
                        y = [r[target] for r in subset]
                        rho = float(spearmanr(x, y).statistic) if np.std(x) > 0 and np.std(y) > 0 else None
                        results.append(
                            dict(
                                step=step,
                                panel=panel,
                                group=group,
                                feature=feature,
                                target=target,
                                n_rows=len(subset),
                                n_teacher=len({r["teacher_path"] for r in subset}),
                                spearman=rho,
                                adjusted_rank_correlation=partial_rank(subset, feature, target),
                            )
                        )
    write_csv("associations.csv", results)
    return results


def predict(joined):
    allresults = []
    predictions = []
    for step in [64, 320]:
        for scope in ["compatible", "published_direct_parent"]:
            rows = [
                r
                for r in joined
                if r["step"] == step
                and r["panel"] == "All held-out"
                and r["compatible"]
                and all(r[k] is not None and r[k] > 0 for k in FEATURES)
                and (scope == "compatible" or r["lineage"] == "published_direct_parent")
            ]
            if len({r["teacher_path"] for r in rows}) < 6:
                continue
            groups = sorted({r["student"] for r in rows})
            base = np.array([[np.log10(r["initial_rkl"]), *[int(r["student"] == g) for g in groups[1:]]] for r in rows])
            geometry = np.array([[transform(k, r[k]) for k in FEATURES] for r in rows])
            y = np.log10([r["final_rkl"] for r in rows])
            variants = [
                ("initial_KL+group", []),
                *[(f"+{k}", [i]) for i, k in enumerate(FEATURES)],
                ("+all_geometry", list(range(len(FEATURES)))),
            ]
            for model, indices in variants:
                x = np.column_stack([base, geometry[:, indices]])
                pred = np.zeros(len(rows))
                for teacher in {r["teacher_path"] for r in rows}:
                    test = np.array([r["teacher_path"] == teacher for r in rows])
                    train = ~test
                    mean = x[train].mean(0)
                    std = x[train].std(0)
                    std[std < 1e-12] = 1
                    a = np.column_stack([np.ones(train.sum()), (x[train] - mean) / std])
                    b = np.column_stack([np.ones(test.sum()), (x[test] - mean) / std])
                    penalty = np.eye(a.shape[1])
                    penalty[0, 0] = 0
                    beta = np.linalg.solve(a.T @ a + penalty, a.T @ y[train])
                    pred[test] = b @ beta
                # Weight a teacher equally, irrespective of the number of input modes.
                errors = {
                    t: float(
                        np.mean(
                            np.abs(
                                pred[[r["teacher_path"] == t for r in rows]] - y[[r["teacher_path"] == t for r in rows]]
                            )
                        )
                    )
                    for t in {r["teacher_path"] for r in rows}
                }
                allresults.append(
                    dict(
                        step=step,
                        scope=scope,
                        model=model,
                        n_rows=len(rows),
                        n_teacher=len(errors),
                        teacher_balanced_MAE_dex=float(np.mean(list(errors.values()))),
                        row_RMSE_dex=float(np.sqrt(np.mean((pred - y) ** 2))),
                        ridge_alpha=1.0,
                    )
                )
                for r, v in zip(rows, pred, strict=True):
                    predictions.append(
                        dict(
                            step=step,
                            scope=scope,
                            model=model,
                            run_id=r["run_id"],
                            teacher_path=r["teacher_path"],
                            actual_log10_K=np.log10(r["final_rkl"]),
                            predicted_log10_K=float(v),
                        )
                    )
    write_csv("teacher-heldout-prediction.csv", predictions)
    dump("prediction-summary.json", allresults)
    return allresults


def savefig(fig, name, ignore=None):
    name += "-v3"
    fig.canvas.draw()
    audit = [str(x) for x in fv.audit(fig, print_slot=11, scoped_ignore=ignore or [])]
    dump(name + ".audit.json", audit)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT / f"{name}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(name, "audit", audit, flush=True)


def figures(profiles, joined, associations, prediction):
    fv.apply_rc("paper", fontsize=10)
    colors = dict(zip(["R1", "Qwen", "Llama"], fv.series_colors("paper", 3), strict=True))

    def family(r):
        return "R1" if r["student"].startswith("r") else "Qwen" if r["student"].startswith("q") else "Llama"

    def marker(r):
        return "s" if r["added_stage"] == "added SFT" else "o" if r["added_stage"].startswith("RL") else "^"

    selected = [r for r in joined if r["step"] == 64 and r["panel"] == "All held-out" and r["compatible"]]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.17, top=0.82, wspace=0.22)
    for ax, feature, label, log in zip(
        axes,
        ["norm_over_base", "top1pct", "kappa_spec"],
        ["Teacher update / initial weight norm", "Top 1% update energy", "$\\kappa_{spec}$"],
        [True, False, True],
        strict=True,
    ):
        for fam, color in colors.items():
            for symbol in ["s", "o", "^"]:
                points = [r for r in selected if family(r) == fam and marker(r) == symbol]
                if points:
                    ax.scatter(
                        [r[feature] for r in points],
                        [r["remaining"] for r in points],
                        color=color,
                        marker=symbol,
                        s=48,
                        alpha=0.8,
                    )
        if log:
            ax.set_xscale("log")
        ax.set(xlabel=label, yscale="log", ylabel="Held-out KL remaining at step64", ylim=(0.005, 1.3))
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], marker="o", linestyle="None", color=c, label=k) for k, c in colors.items()]
    handles += [
        Line2D([], [], marker=m, linestyle="None", color=fv.NEUTRALS["baseline"], label=stage_label)
        for m, stage_label in [("s", "Added SFT"), ("o", "RL"), ("^", "Other history")]
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=6, frameon=False)
    fig.text(
        0.5,
        0.01,
        "Six matrices per pair; shared teacher modes remain dependent; fixed H1024 risk",
        ha="center",
        fontsize=8,
        color=fv.NEUTRALS["baseline"],
    )
    savefig(fig, "feature-vs-absorption")
    # Every teacher curve: feature table plus initial/final absolute risk.
    unique = []
    seen = set()
    for r in selected:
        k = (r["student"], r["teacher_path"])
        if k not in seen:
            unique.append(r)
            seen.add(k)
    fig, axes = plt.subplots(1, 3, figsize=(14, max(8, len(unique) * 0.36)), sharey=True)
    fig.subplots_adjust(left=0.29, right=0.98, bottom=0.12, top=0.9, wspace=0.4)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.65, 0.98), ncol=6, frameon=False)
    for i, r in enumerate(unique):
        c = colors[family(r)]
        axes[0].scatter(r["norm_over_base"], i, color=c, marker=marker(r), s=35)
        axes[1].scatter(r["top1pct"], i, color=c, marker=marker(r), s=35)
        axes[2].plot([r["final_rkl"], r["initial_rkl"]], [i, i], color=c, alpha=0.65)
        axes[2].scatter(r["initial_rkl"], i, facecolors="none", edgecolors=c, s=25)
        axes[2].scatter(r["final_rkl"], i, color=c, marker=marker(r), s=30)
    axes[0].set_yticks(range(len(unique)), [r["student"] + ": " + r["teacher"] for r in unique])
    axes[0].invert_yaxis()
    axes[0].set(xscale="log", xlabel="Teacher update / initial norm")
    axes[1].set(xlabel="Top 1% update energy")
    axes[2].set(xscale="log", xlabel="Absolute held-out RKL: initial (open), step64 (filled)")
    fig.text(
        0.5,
        0.008,
        "All compatible explorer conditions; same teacher under different input modes shown separately",
        ha="center",
        fontsize=8,
        color=fv.NEUTRALS["baseline"],
    )
    # Rows identify models; factorized global legend identifies family and stage.
    # Multiple artists in one row are endpoints of one observation, not series.
    ignore = [("palette-overflow", ax) for ax in axes]
    ignore += [("missing-axis-label", ax) for ax in axes[1:]]
    savefig(fig, "model-fingerprint-overview", ignore)
    # Full trajectory retains absolute level and magnitude of changes for compatible pairs.
    data = json.loads((OUT / "explorer-data.json").read_text())["runs"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 6.2), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.29, top=0.9, wspace=0.2)
    scopes = [("r1", "R1-1.5B"), ("q3-1.7", "Qwen thinking"), ("q3-1.7-nonthinking", "Qwen non-thinking")]
    labels = sorted({p["teacher"] for p in profiles if p["compatible"] and p["student"] in {s[0] for s in scopes}})
    palette = fv.series_colors("paper", 4)
    from matplotlib.colors import to_rgb

    registry = {
        label: tuple(v * (1 - 0.16 * (i // 4)) for v in to_rgb(palette[i % 4])) for i, label in enumerate(labels)
    }
    fv.register_colors(*registry.values())
    for ax, (student, title) in zip(axes, scopes, strict=True):
        pr = [p for p in profiles if p["student"] == student and p["compatible"]]
        colors2 = [registry[p["teacher"]] for p in pr]
        for p, color in zip(pr, colors2, strict=True):
            c = next(r for r in data if r["run_id"] == p["run_id"])
            points = c["panels"]["All held-out"]
            ax.plot([v["step"] for v in points], [v["rkl"] for v in points], label=p["teacher"], color=color)
        ax.set(
            xlabel="OPD updates",
            yscale="log",
            ylabel="Absolute held-out RKL (nats / H1024)",
            title=title,
            ylim=(0.0003, 1.5),
        )
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=2, fontsize=8, frameon=False)
    fig.text(
        0.5,
        0.012,
        "Original complete single-pass curves; no continuation stitching; all saved intermediate readouts retained",
        ha="center",
        fontsize=8,
        color=fv.NEUTRALS["baseline"],
    )
    savefig(fig, "matched-spectrum-kl-dynamics")


def main():
    assert (OUT / "measurement-complete.json").exists()
    profiles, joined = fingerprint()
    spectral_neighbors(profiles)
    a = associations(joined)
    p = predict(joined)
    figures(profiles, joined, a, p)
    dump(
        "analysis-protocol.json",
        dict(
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            independent_unit="teacher weight identity; modes grouped in leave-one-teacher-out; no matrix replication",
            targets="64,320 actual updates; fixed-H1024 RKL; late256:320 only; domains separately",
            prediction="ridge alpha1 fixed before results; training-fold standardization; log10 K endpoint",
            multiplicity="exploratory associations; no confirmatory p values; no classifier fitted",
            limitations=[
                "six k/v projection matrices, not full-model spectrum",
                "public parent models often unpinned",
                "same-size need not mean direct lineage",
                "no new RKL or forward-KL inference",
            ],
        ),
    )
    print(json.dumps(json.loads((OUT / "coverage-summary.json").read_text()), indent=2))
    print(json.dumps(p, indent=2))


if __name__ == "__main__":
    main()
