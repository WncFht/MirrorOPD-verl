"""Aggregate continuous-parent geometry; each point is one identified condition.

Ledger: family=color; method=marker; module=column; norms aggregate squared
energies before ratios. Spectral sketches are lower bounds with two seeds.
Figures show measured parameter geometry and historical functional risk.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(
    0, "/mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/fht_data/dotfiles/.agents/skills/figure-viz/assets/mpl"
)
import fvstyle as fv


def js(a, b):
    a, b = np.asarray(a), np.asarray(b)
    a, b = a / a.sum(), b / b.sum()
    m = (a + b) / 2
    return float(sum(np.sum(x[x > 0] * np.log(x[x > 0] / m[x > 0])) for x in [a, b]) / 2)


def ancillary(c):
    vals = defaultdict(list)
    for v in c.get("curves", []):
        domain = v["domain"]
        if domain not in ["all", "macro4", "math", "code", "if"]:
            continue
        bank = v["bank"]
        step = int(v.get("optimizer_updates", v.get("step", 0)))
        vals[bank, step].append((domain, float(v["rkl"])))
    curves = defaultdict(dict)
    for (bank, step), rows in vals.items():
        macro = [x for d, x in rows if d in ["all", "macro4"]]
        curves[bank][step] = float(np.mean(macro or [x for _, x in rows]))
    preferred = "public_teacher" if c["family"] == "MiMo" else "teacher-bank"
    if preferred not in curves:
        return None
    line = curves[preferred]
    end = min(max(line), 320)
    if end not in line:
        end = max(line)
    initial, final = line[0], line[end]
    early = line.get(64, np.nan)
    return dict(
        initial_K=initial,
        final_K=final,
        remaining=final / initial,
        absolute_drop=initial - final,
        early_log_drop=np.log(initial / early),
        late_log_drop=np.log(early / final),
        curve_step=end,
        panel="MiMo macro teacher bank" if c["family"] == "MiMo" else "controlled macro teacher bank",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--families", nargs="+")
    args = ap.parse_args()
    root, out = args.root, args.output
    out.mkdir(parents=True, exist_ok=True)
    inv = json.loads((root / "direct-pairs-v1/inventory.json").read_text())
    if args.families:
        inv["pairs"] = [p for p in inv["pairs"] if p["family"] in args.families]
        inv["conditions"] = [c for c in inv["conditions"] if c["family"] in args.families]
    curves = pd.read_csv(root / "curve-spectrum-v1/curve-feature-table.csv").set_index("run_id")
    explorer_runs = {
        r["run_id"]: r for r in json.loads((root / "spectrum-expanded-v4/explorer-data.json").read_text())["runs"]
    }
    modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    teachers, stages, module_rows, sketches, controls = [], [], [], [], []
    configs = json.loads((root / "direct-pairs-v1/coordinate-configs.json").read_text())
    coverage = []
    for p in inv["pairs"]:
        rows = [
            json.loads(f.read_text()) for f in sorted((root / "direct-pairs-v1/matrices" / p["pair_id"]).glob("*.json"))
        ]
        config = configs[p["pair_id"]]
        expected = 7 * config["num_hidden_layers"]
        coverage.append(dict(pair_id=p["pair_id"], teacher=p["teacher"], observed=len(rows), expected=expected))
        if len(rows) != expected:
            raise ValueError(f"Incomplete {p['teacher']}: {len(rows)}/{expected}")
        ed = sum(r["delta_energy"] for r in rows)
        eb = sum(r["base_energy"] for r in rows)
        numel = sum(r["elements"] for r in rows)
        l1 = sum(r["delta_L1"] for r in rows)
        sk = [s["energy_fraction"] for r in rows for s in r["sketches"]]
        seedspread = [
            abs(r["sketches"][0]["energy_fraction"] - r["sketches"][1]["energy_fraction"])
            for r in rows
            if r["sketches"]
        ]
        t = dict(
            pair_id=p["pair_id"],
            family=p["family"],
            teacher=p["teacher"],
            student=p["student"],
            label=p["label"],
            cohort=p["cohort"],
            matrices=len(rows),
            delta_norm=ed**0.5,
            norm_over_base=(ed / eb) ** 0.5,
            entry_support_fraction=l1 * l1 / (ed * numel),
            sampled_kurtosis_median=float(np.median([r["sample_kurtosis"] for r in rows])),
            sketch_top16_median=float(np.median(sk)),
            seed_spread_max=max(seedspread),
        )
        layer_energy = np.array(
            [
                sum(r["delta_energy"] for r in rows if int(r["tensor"].split(".")[2]) == layer)
                for layer in range(config["num_hidden_layers"])
            ]
        )
        weights = layer_energy / ed
        t["layer_effective_fraction"] = float(np.exp(-np.sum(weights * np.log(weights))) / len(weights))
        t["depth_center"] = float(np.sum(weights * np.linspace(0, 1, len(weights))))
        t["delta_base_cosine"] = sum(r["weight_dot_delta"] for r in rows) / (ed * eb) ** 0.5
        for module in modules:
            t[module + "_energy_fraction"] = (
                sum(r["delta_energy"] for r in rows if r["tensor"].split(".")[-2] == module) / ed
            )
        t["attention_fraction"] = sum(t[m + "_energy_fraction"] for m in modules[:4])
        teachers.append(t)
        groups = defaultdict(list)
        for r in rows:
            module = r["tensor"].split(".")[-2]
            for s in r["sketches"]:
                sketches.append(
                    dict(
                        pair_id=p["pair_id"],
                        teacher=p["teacher"],
                        module=module,
                        family=p["family"],
                        label=p["label"],
                        **s,
                    )
                )
            for s in r["stages"]:
                groups[s["run_id"], s["step"]].append((r, s))
        for (run_id, step), group in groups.items():
            es = sum(s["student_energy"] for _, s in group)
            dot = sum(s["dot"] for _, s in group)
            missing = sum(s["missing_energy"] for _, s in group)
            assert abs(missing - (ed + es - 2 * dot)) < 1e-8 * max(missing, 1)
            teacher_modules, student_modules = [], []
            for module in modules:
                rr = [(r, s) for r, s in group if r["tensor"].split(".")[-2] == module]
                te = sum(r["delta_energy"] for r, _ in rr)
                se = sum(s["student_energy"] for _, s in rr)
                dd = sum(s["dot"] for _, s in rr)
                teacher_modules.append(te)
                student_modules.append(se)
                module_rows.append(
                    dict(
                        run_id=run_id,
                        step=step,
                        teacher=p["teacher"],
                        family=p["family"],
                        module=module,
                        teacher_fraction=te / ed,
                        student_fraction=se / es,
                        gain=dd / te,
                        cosine=dd / (te * se) ** 0.5,
                    )
                )
            row = dict(
                **t,
                run_id=run_id,
                step=step,
                student_norm=es**0.5,
                student_over_teacher=(es / ed) ** 0.5,
                update_cosine=dot / (es * ed) ** 0.5,
                amplitude_gain=dot / ed,
                missing_energy_ratio=missing / ed,
                student_teacher_axis_fraction=dot * dot / (ed * es),
                module_JS=js(teacher_modules, student_modules),
            )
            stages.append(row)
        for f in (root / "direct-pairs-v1/direction-controls" / p["pair_id"]).glob("*.json"):
            r = json.loads(f.read_text())
            for s in r["stages"]:
                record = dict(
                    pair_id=p["pair_id"],
                    teacher=p["teacher"],
                    family=p["family"],
                    label=p["label"],
                    run_id=s["run_id"],
                    step=s["step"],
                    tensor=r["tensor"],
                    top16_gain=s["top16_gain"],
                    top16_cosine=s["top16_cosine"],
                    matched_fraction=s["matched_fraction"],
                    top16_energy=s["top16_energy"],
                    student_energy=s["student_energy"],
                    random_abs_cosine_mean=float(np.mean([abs(x["cosine"]) for x in s["random_frames"]])),
                    random_gain_RMS=float(np.mean([x["gain"] ** 2 for x in s["random_frames"]]) ** 0.5),
                )
                for i, b in enumerate(s["bands"]):
                    record[f"band{i}_gain"] = b["gain"]
                controls.append(record)
    condition_rows = []
    for c in inv["conditions"]:
        p = next(p for p in inv["pairs"] if any(z["run_id"] == c["run_id"] for z in p["conditions"]))
        t = next(t for t in teachers if t["pair_id"] == p["pair_id"])
        row = dict(t, run_id=c["run_id"])
        if c["run_id"] in curves.index:
            cr = curves.loc[c["run_id"]]
            row.update(
                {
                    k: cr[k]
                    for k in ["initial_K", "final_K", "remaining", "absolute_drop", "early_log_drop", "late_log_drop"]
                }
            )
            row.update(panel="historical H1024", curve_step=c["budget"])
            for k in ["delta_energy_top1", "delta_energy_top5", "delta_effective_rank_fraction", "delta_stable_rank"]:
                row[k] = cr.get(k, np.nan)
        elif c["run_id"] in explorer_runs:
            line = explorer_runs[c["run_id"]]["panels"]["All held-out"]
            initial, final = line[0]["rkl"], line[-1]["rkl"]
            row.update(
                initial_K=initial,
                final_K=final,
                remaining=final / initial,
                absolute_drop=initial - final,
                curve_step=line[-1]["step"],
                panel="historical H1024",
            )
        else:
            extra = ancillary(c)
            if extra:
                row.update(extra)
        condition_rows.append(row)
    tf, sf, cf, mf, skf, rf = map(pd.DataFrame, [teachers, stages, condition_rows, module_rows, sketches, controls])
    for name, frame in [
        ("teachers", tf),
        ("students", sf),
        ("conditions", cf),
        ("modules", mf),
        ("sketches", skf),
        ("controls", rf),
    ]:
        frame.to_csv(out / (name + ".csv"), index=False)
    associations = []
    features = [
        "norm_over_base",
        "entry_support_fraction",
        "sampled_kurtosis_median",
        "sketch_top16_median",
        "delta_energy_top1",
        "delta_energy_top5",
        "delta_effective_rank_fraction",
    ]
    for family, group in cf[(cf.cohort == "continuous") & (cf.panel == "historical H1024")].groupby("family"):
        for target in ["final_K", "remaining", "late_log_drop"]:
            for feature in features:
                g = group.drop_duplicates("teacher").dropna(subset=[feature, target])
                if len(g) >= 3:
                    rho, pval = spearmanr(g[feature], g[target])
                    associations.append(dict(family=family, target=target, feature=feature, n=len(g), rho=rho, p=pval))
    pd.DataFrame(associations).to_csv(out / "associations.csv", index=False)
    # Fixed ridge, standardized on each teacher-held-out training fold; endpoints are never predictors.
    g = cf[(cf.cohort == "continuous") & (cf.panel == "historical H1024") & (cf.curve_step == 320)].copy()
    predictions = []
    for featset, cols in [
        ("initial", ["initial_K"]),
        ("initial+norm", ["initial_K", "norm_over_base"]),
        ("initial+concentration", ["initial_K", "sketch_top16_median"]),
        (
            "initial+geometry",
            ["initial_K", "norm_over_base", "sketch_top16_median", "entry_support_fraction", "sampled_kurtosis_median"],
        ),
    ]:
        x = g[cols].to_numpy(float, copy=True)
        x[:, 0] = np.log(x[:, 0])
        if "norm_over_base" in cols:
            x[:, cols.index("norm_over_base")] = np.log(x[:, cols.index("norm_over_base")])
        family = pd.get_dummies(g.family).to_numpy(float, copy=True)
        y = np.log(g.final_K.to_numpy(float, copy=True))
        for teacher in g.teacher.unique():
            test = (g.teacher == teacher).to_numpy()
            train = ~test
            mu, sd = x[train].mean(0), x[train].std(0)
            sd[sd < 1e-12] = 1
            z = np.column_stack([np.ones(len(x)), (x - mu) / sd, family])
            reg = np.eye(z.shape[1])
            reg[0, 0] = 0
            teacher_counts = g.teacher.value_counts()
            weights = np.array([1 / teacher_counts[t] for t in g.teacher])[train]
            beta = np.linalg.solve(z[train].T @ (z[train] * weights[:, None]) + reg, z[train].T @ (y[train] * weights))
            for idx in np.flatnonzero(test):
                predictions.append(
                    dict(
                        features=featset,
                        teacher=teacher,
                        run_id=g.iloc[idx].run_id,
                        actual=y[idx],
                        predicted=float(z[idx] @ beta),
                    )
                )
    pred = pd.DataFrame(predictions)
    pred.to_csv(out / "teacher-heldout-predictions.csv", index=False)
    pred["squared_error"] = (pred.actual - pred.predicted) ** 2
    teacher_error = pred.groupby(["teacher", "features"]).squared_error.mean().unstack()
    errors = {k: float(np.sqrt(v.mean())) for k, v in teacher_error.items()}
    rng = np.random.default_rng(20261110)
    indices = rng.integers(0, len(teacher_error), size=(10000, len(teacher_error)))
    error_differences = {}
    for key in teacher_error.columns:
        if key == "initial":
            continue
        delta = (teacher_error[key] - teacher_error["initial"]).to_numpy()
        error_differences[key] = dict(
            mean_MSE_difference=float(delta.mean()),
            bootstrap95=np.quantile(delta[indices].mean(1), [0.025, 0.975]).tolist(),
        )
    # Cross-family method classification uses only parent groups containing both labels.
    pairmap = {p["pair_id"]: p for p in inv["pairs"]}
    tf["parent"] = [pairmap[x]["base"] for x in tf.pair_id]
    eligible = [k for k, v in tf[tf.cohort == "continuous"].groupby("parent") if v.label.nunique() == 2]
    cl = tf[(tf.cohort == "continuous") & tf.parent.isin(eligible)].copy()
    classification = []
    for label, cols in [
        ("norm", ["norm_over_base"]),
        ("concentration", ["sketch_top16_median"]),
        ("norm+concentration", ["norm_over_base", "sketch_top16_median"]),
    ]:
        x = cl[cols].to_numpy(float, copy=True)
        if "norm_over_base" in cols:
            x[:, cols.index("norm_over_base")] = np.log(x[:, cols.index("norm_over_base")])
        y = (cl.label == "SFT").to_numpy(float, copy=True)
        for family in cl.family.unique():
            test = (cl.family == family).to_numpy()
            train = ~test
            mu = x[train].mean(0)
            sd = x[train].std(0)
            sd[sd < 1e-12] = 1
            z = np.column_stack([np.ones(len(x)), (x - mu) / sd])
            weights = np.where(y[train] > 0, 0.5 / y[train].sum(), 0.5 / (1 - y[train]).sum()) * train.sum()

            def objective(b, z=z, train=train, weights=weights, y=y):
                logits = z[train] @ b
                return np.sum(weights * (np.logaddexp(0, logits) - y[train] * logits)) + 0.5 * np.sum(b[1:] ** 2)

            beta = minimize(objective, np.zeros(z.shape[1]), method="BFGS").x
            for idx in np.flatnonzero(test):
                classification.append(
                    dict(
                        features=label,
                        family=family,
                        teacher=cl.iloc[idx].teacher,
                        actual=int(y[idx]),
                        probability=float(1 / (1 + np.exp(-z[idx] @ beta))),
                    )
                )
    cp = pd.DataFrame(classification)
    cp.to_csv(out / "leave-family-out-method.csv", index=False)
    classifier_accuracy = {k: float(np.mean((v.probability >= 0.5) == v.actual)) for k, v in cp.groupby("features")}
    summary = dict(
        pairs=len(tf),
        continuous_pairs=int((tf.cohort == "continuous").sum()),
        conditions=len(cf),
        matrices=int(tf.matrices.sum()),
        student_matrix_records=int(
            sum(
                len(s["stages"])
                for p in inv["pairs"]
                for f in (root / "direct-pairs-v1/matrices" / p["pair_id"]).glob("*.json")
                for s in [json.loads(f.read_text())]
            )
        ),
        control_matrices=int(len(rf[["pair_id", "tensor"]].drop_duplicates())),
        maximum_sketch_seed_spread=float(tf.seed_spread_max.max()),
        ridge_log_final_RMSE=errors,
        ridge_MSE_differences=error_differences,
        leave_family_out_method_accuracy=classifier_accuracy,
        method_classification_teachers=len(cl),
        coverage=coverage,
        scope="all attention/MLP matrices, embeddings/head/norm excluded",
        notes=("H1024 and macro-bank risks separate; randomized top16 energies lower bounds; controls parameter-only"),
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    plot(out, tf, sf, cf, mf, skf, rf)


def plot(out, tf, sf, cf, mf, skf, rf):
    fv.apply_rc("paper")
    colors = dict(zip(["R1", "Qwen", "Llama", "MiMo"], fv.series_colors("paper", 4), strict=True))
    colors = {k: v for k, v in colors.items() if k in tf.family.unique()}
    audits = {}

    def save(fig, name, caption):
        fig.text(0.5, 0.012, caption, ha="center", fontsize=8, color=fv.NEUTRALS["baseline"])
        fig.set_layout_engine("constrained", rect=(0, 0.065, 1, 0.93))
        fig.canvas.draw()
        audits[name] = [str(x) for x in fv.audit(fig, print_slot=fig.get_figwidth())]
        for ext in ["png", "pdf", "svg"]:
            fig.savefig(out / (name + "." + ext), dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(1, len(colors), figsize=(16, 4), layout="constrained")
    for ax, (family, color) in zip(axes, colors.items(), strict=True):
        data = cf[(cf.family == family) & (cf.cohort == "continuous")]
        for _, r in data.iterrows():
            ax.scatter(r.norm_over_base, r.remaining, c=color, marker="o" if r.label == "RL" else "s", s=45)
            ax.annotate(
                r.teacher.replace("controlled-", ""),
                (r.norm_over_base, r.remaining),
                xytext={
                    "brorl": (6, 14),
                    "nemotron": (-50, 10),
                    "justrl": (-34, -17),
                    "controlled-CCS-rho0": (5, -18),
                    "controlled-CCS-rho1": (5, 12),
                    "coreward-i": (4, 16),
                    "gt": (4, -15),
                    "MiMo-Zero": (5, -16),
                }.get(r.teacher, (3, 4)),
                textcoords="offset points",
                fontsize=7,
            )
        ax.set(
            xscale="log",
            yscale="log",
            xlabel="Teacher update / initial weight norm",
            ylabel="Remaining functional RKL",
            title=family,
        )
        ax.margins(0.35, 0.4)
        ax.set_ylim(0.01, 1.5)
    save(
        fig,
        "norm-versus-fit",
        "Continuous pairs; circles RL, squares SFT; MiMo and controlled-bank risks use separate protocols",
    )
    latest = sf.sort_values("step").drop_duplicates("run_id", keep="last")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for family, color in colors.items():
        g = latest[(latest.family == family) & (latest.cohort == "continuous")]
        axes[0].scatter(g.student_over_teacher, g.update_cosine, color=color, label=family)
        joined = g.merge(cf[["run_id", "remaining"]], on="run_id")
        axes[1].scatter(joined.missing_energy_ratio, joined.remaining, color=color, label=family)
    axes[0].set(xscale="log", xlabel="Student / teacher update norm", ylabel="All-layer update cosine")
    axes[1].set(yscale="log", xlabel="Parameter residual energy / teacher energy", ylabel="Remaining functional RKL")
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.0, 1))
    save(fig, "parameter-versus-function", "All-layer energy aggregation; model endpoints describe learning paths")
    data = tf.sort_values(["family", "teacher"])
    matrix = []
    labels = []
    mods = ["q_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    for _, r in data.iterrows():
        matrix.append([float(skf[(skf.pair_id == r.pair_id) & (skf.module == m)].energy_fraction.mean()) for m in mods])
        labels.append(r.family + ": " + r.teacher)
    fig, ax = plt.subplots(figsize=(9, 9), layout="constrained")
    im = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set(
        xticks=range(len(mods)),
        xticklabels=mods,
        yticks=range(len(labels)),
        yticklabels=labels,
        xlabel="Middle-layer module",
        ylabel="Actual student to teacher pair",
    )
    fig.colorbar(im, ax=ax, label="Rank16 captured energy (lower bound)")
    save(
        fig,
        "module-concentration",
        "Two seeded randomized SVDs per matrix; exact full SVD remains available for six k/v matrices",
    )
    cr = rf.sort_values("step").drop_duplicates(["run_id", "tensor"], keep="last")
    fig, axes = plt.subplots(1, len(colors), figsize=(15, 4), layout="constrained")
    for ax, (family, color) in zip(axes, colors.items(), strict=True):
        g = cr[cr.family == family]
        ax.scatter(g.random_abs_cosine_mean, np.abs(g.top16_cosine), color=color, s=16)
        ax.set(
            xscale="log",
            yscale="log",
            xlabel="Random-frame absolute cosine",
            ylabel="True-frame absolute cosine",
            title=family,
        )
    for ax in axes:
        ax.set_ylim(0.002, 1)
        ax.set_xlim(1e-5, 0.005)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    save(
        fig,
        "same-spectrum-controls",
        "Same teacher top16 singular values; three randomized left/right frames; six k/v matrices per pair",
    )
    (out / "figure-audits.json").write_text(json.dumps(audits, indent=2))


if __name__ == "__main__":
    main()
