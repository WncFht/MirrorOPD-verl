"""Cached exact spectra versus absolute-risk dynamics; descriptive CPU analysis.

Clusters use average linkage on log10 absolute RKL at 0:16:320, interpolating
in log risk without extrapolation. Colors identify curve groups. Correlation
heatmaps retain initial-student/input-mode strata. No natural classes assumed.
"""

import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.special import comb
from scipy.stats import spearmanr


def adjusted_rand_score(a, b):
    table = np.zeros((len(set(a)), len(set(b))), dtype=int)
    for x, y in zip(a, b, strict=True):
        table[int(x) - 1, int(y) - 1] += 1
    same = comb(table, 2).sum()
    sa = comb(table.sum(0), 2).sum()
    sb = comb(table.sum(1), 2).sum()
    expected = sa * sb / comb(len(a), 2)
    return float((same - expected) / ((sa + sb) / 2 - expected))


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "spectrum-expanded-v4"
OUT = ROOT / "curve-spectrum-v1"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT.parents[2] / "dotfiles/.agents/skills/figure-viz/assets/mpl"))
import fvstyle as fv  # noqa: E402


def spectral(d):
    s = np.array(d["singular_values"])
    p = s * s / (s @ s)
    n = len(s)
    cs = p.cumsum()
    o = dict(
        norm=d["frobenius"],
        stable_rank=d["stable_rank"],
        entropy_rank=float(np.exp(-sum(p[p > 0] * np.log(p[p > 0])))),
        participation_rank=float(1 / (p @ p)),
        gini=float(2 * np.trapezoid(np.r_[0, cs], dx=1 / n) - 1),
        log_condition=float(np.log10(s[0] / max(s[-1], 1e-30))),
    )
    for q in [0.1, 0.5, 0.9, 0.95, 0.99]:
        o["rank" + str(int(q * 100)) + "_fraction"] = float((np.searchsorted(cs, q) + 1) / n)
    for q in [0.01, 0.05, 0.1, 0.25, 0.5]:
        o["energy_top" + str(int(q * 100))] = float(p[: int(np.ceil(q * n))].sum())
    o["effective_rank_fraction"] = o["entropy_rank"] / n
    o["participation_rank_fraction"] = o["participation_rank"] / n
    o["tail50_energy"] = 1 - o["energy_top50"]
    for k in ["norm_over_base", "kappa_spec", "rho_sigma"]:
        if k in d:
            o[k] = d[k]
    if "frame_diag" in d:
        total = sum(d[k] for k in ["frame_diag", "frame_off", "frame_null"])
        for k in ["frame_diag", "frame_off", "frame_null"]:
            o[k + "_fraction"] = d[k] / total
    return o


def dump(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, allow_nan=False))


def save(fig, name, ignore=None):
    fig.canvas.draw()
    audit = [str(x) for x in fv.audit(fig, print_slot=12, scoped_ignore=ignore or [])]
    dump(name + "-v2.audit.json", audit)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT / (name + "-v2." + ext), dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(name, audit, flush=True)


def main():
    curves = json.loads((SRC / "explorer-data.json").read_text())["runs"]
    profiles = json.loads((SRC / "model-profiles.json").read_text())
    rows = []
    matrices = []
    vectors = []
    grid = np.arange(0, 321, 16)
    for c in curves:
        pts = c["panels"]["All held-out"]
        x = np.array([p["step"] for p in pts])
        y = np.array([p["rkl"] for p in pts])
        if 320 not in x:
            continue
        p = next(p for p in profiles if p["run_id"] == c["run_id"])
        v = np.interp(grid, x, np.log10(y))
        vectors.append(v)
        r = {k: p[k] for k in ["run_id", "student", "teacher", "teacher_path", "compatible", "added_stage", "lineage"]}
        r.update(
            initial_K=10 ** v[0],
            final_K=10 ** v[-1],
            remaining=10 ** (v[-1] - v[0]),
            absolute_drop=10 ** v[0] - 10 ** v[-1],
            early_log_drop=v[0] - v[4],
            late_log_drop=v[16] - v[20],
            late_absolute_drop=10 ** v[16] - 10 ** v[20],
            late_vs_early_rate=(v[16] - v[20]) / max(v[0] - v[4], 1e-12),
            area_absolute=float(np.trapezoid(10**v, grid) / 320),
        )
        ms = []
        for file in sorted((SRC / "runs" / p["run_id"]).glob("*.json")):
            m = json.loads(file.read_text())
            entry = dict(run_id=p["run_id"], tensor=m["tensor"])
            for name, key in [("base", "base_full"), ("teacher", "teacher_full"), ("delta", "teacher_delta")]:
                if m.get(key):
                    entry.update({name + "_" + k: v for k, v in spectral(m[key]).items()})
            for st in m["stages"]:
                for name, key in [("student", "delta"), ("missing", "missing")]:
                    if st.get(key):
                        entry.update({f"{name}{st['step']}_" + k: v for k, v in spectral(st[key]).items()})
                for k in ["update_cosine", "left_overlap16", "right_overlap16"]:
                    if st.get(k) is not None:
                        entry[f"{k}_{st['step']}"] = st[k]
            ms.append(entry)
            matrices.append(entry)
        keys = set().union(*(set(e) for e in ms)) - {"run_id", "tensor"}
        for k in keys:
            r[k] = float(np.median([e[k] for e in ms if k in e]))
        train = c["panels"]["Training scored"]
        tv = np.interp(grid, [p["step"] for p in train], np.log10([p["rkl"] for p in train]))
        r["late_train_log_drop"] = tv[16] - tv[20]
        r["late_generalization_gap"] = r["late_train_log_drop"] - r["late_log_drop"]
        rows.append(r)
    V = np.array(vectors)
    labels = fcluster(linkage(V, method="average"), 4, criterion="maxclust")
    order = sorted(set(labels), key=lambda k: np.median(V[labels == k, -1]))
    mapping = {v: i + 1 for i, v in enumerate(order)}
    for r, label in zip(rows, labels, strict=True):
        r["absolute_curve_group"] = mapping[label]
    sensitivity = []
    for transform, name in [(V, "absolute"), (V - V[:, 0, None], "relative_shape")]:
        for method in ["average", "complete", "ward"]:
            lab = fcluster(linkage(transform, method=method), 4, criterion="maxclust")
            sensitivity.append(
                dict(distance=name, linkage=method, ARI_vs_absolute_average=float(adjusted_rand_score(labels, lab)))
            )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "curve-feature-table.csv", index=False)
    pd.DataFrame(matrices).to_csv(OUT / "matrix-feature-table.csv", index=False)
    dump(
        "curve-clusters.json",
        dict(
            grid=grid.tolist(),
            log10_absolute_risk=V.tolist(),
            runs=[{k: r[k] for k in ["run_id", "student", "teacher", "absolute_curve_group"]} for r in rows],
            sensitivity=sensitivity,
        ),
    )
    targets = ["final_K", "remaining", "late_log_drop", "late_generalization_gap"]
    features = [c for c in df.columns if c.startswith(("delta_", "update_cosine_", "left_overlap", "right_overlap"))]
    associations = []
    for group in ["r1", "q3-1.7", "q3-1.7-nonthinking"]:
        a = df[(df.student == group) & df.compatible]
        for feature in features:
            for target in targets:
                b = a[[feature, target]].dropna()
                if len(b) >= 4 and b[feature].nunique() > 1:
                    associations.append(
                        dict(
                            group=group,
                            feature=feature,
                            target=target,
                            n=len(b),
                            rho=float(spearmanr(b[feature], b[target]).statistic),
                        )
                    )
    pd.DataFrame(associations).to_csv(OUT / "within-group-associations.csv", index=False)
    fv.apply_rc("paper", fontsize=10)
    colors = fv.series_colors("paper", 4)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.1, top=0.94, hspace=0.45, wspace=0.2)
    for ax, k in zip(axes.flat, range(1, 5), strict=True):
        ids = [i for i, r in enumerate(rows) if r["absolute_curve_group"] == k]
        for i in ids:
            ax.plot(grid, 10 ** V[i], color=colors[k - 1], alpha=0.4, lw=1)
        ax.plot(grid, 10 ** np.median(V[ids], axis=0), color=colors[k - 1], lw=2.5)
        ax.set(
            yscale="log",
            xlabel="OPD updates",
            ylabel="Absolute held-out RKL",
            title=f"Group {k}: {len(ids)} conditions",
        )
        print("group", k, "; ".join(rows[i]["student"] + ":" + rows[i]["teacher"] for i in ids), flush=True)
    fig.text(
        0.5,
        0.02,
        "Average-linkage of log absolute RKL; four descriptive groups; thick line = pointwise median",
        ha="center",
        fontsize=8,
    )
    save(
        fig,
        "absolute-curve-groups",
        [(rule, ax) for ax in axes.flat for rule in ["palette-overflow", "legend-coverage"]],
    )
    selected = [
        "delta_norm_over_base",
        "delta_energy_top1",
        "delta_effective_rank_fraction",
        "delta_participation_rank_fraction",
        "delta_rank90_fraction",
        "delta_gini",
        "delta_frame_diag_fraction",
        "delta_frame_null_fraction",
        "delta_rho_sigma",
        "update_cosine_320",
        "left_overlap16_320",
        "right_overlap16_320",
    ]
    names = [
        "Update magnitude",
        "Top-1% energy",
        "Entropy rank / rank",
        "Participation rank / rank",
        "90% energy rank / rank",
        "Energy Gini",
        "Base-frame diagonal",
        "Base-frame null",
        "Singular-value displacement",
        "Student/teacher cosine (320)",
        "Left top16 overlap (320)",
        "Right top16 overlap (320)",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 7), sharey=True)
    fig.subplots_adjust(left=0.27, right=0.88, bottom=0.23, top=0.9, wspace=0.1)
    for ax, group in zip(axes, ["r1", "q3-1.7", "q3-1.7-nonthinking"], strict=True):
        arr = np.full((len(selected), 3), np.nan)
        for i, f in enumerate(selected):
            for j, t in enumerate(targets[:3]):
                a = [a for a in associations if a["group"] == group and a["feature"] == f and a["target"] == t]
                if a:
                    arr[i, j] = a[0]["rho"]
        im = ax.imshow(arr, cmap="BrBG", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(3), ["Final absolute KL", "Remaining ratio", "Late log decrease"], rotation=55, ha="right")
        ax.set_yticks(range(len(selected)), names)
        ax.set_title(group)
        ax.set_xlabel("Curve metric")
        for i in range(len(selected)):
            for j in range(3):
                if np.isfinite(arr[i, j]):
                    ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", fontsize=8)
    cb = fig.colorbar(im, cax=fig.add_axes([0.91, 0.29, 0.015, 0.52]))
    cb.set_label("Within-mode Spearman correlation")
    fig.text(
        0.5,
        0.025,
        "Exploratory associations: 5 R1, 6 thinking, 6 non-thinking teachers; no significance selection",
        ha="center",
        fontsize=8,
    )
    save(fig, "spectrum-dynamics-associations", [("missing-axis-label", ax) for ax in axes])
    dump(
        "protocol.json",
        dict(
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            n_curves=len(rows),
            features=len(df.columns),
            matrix_rows=len(matrices),
            interpolation="linear log10 risk on 0:16:320; observed320 required",
            clustering="average-linkage Euclidean log absolute risk; k4 descriptive, no inferred natural class count",
            limitations=[
                "six k/v matrices",
                "dependent teacher modes",
                "exploratory, no multiplicity adjustment",
                "student features are posttraining descriptors",
            ],
        ),
    )
    print("complete", len(rows), len(df.columns), len(matrices), flush=True)


if __name__ == "__main__":
    main()
