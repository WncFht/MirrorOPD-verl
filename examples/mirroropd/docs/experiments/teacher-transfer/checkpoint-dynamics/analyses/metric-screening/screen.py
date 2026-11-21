"""Explore every existing pair-level feature against held-out output matching.

Ledger: a scatter point is one pair; color is student family and input mode, shape teacher stage;
an outer ring independently marks cross-size. Heatmap rows are explicitly named metrics
and columns bank/domain/outcome definitions. Pair, never matrix, is the unit of
correlation. Missing cross-size teacher geometry remains null. Within-family
results correlate family-centered ranks, not independently replicated students.
All figures use the same -1..1 correlation scale; no significance selection.
"""

import csv
import hashlib
import html
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file() and (p / "verl").is_dir())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pair_identity as identity  # noqa: E402

from examples.mirroropd.research.core.plotting import save_figure  # noqa: E402
from examples.mirroropd.research.studies.absorption_dynamics.cohort_comparison import place_labels  # noqa: E402
from examples.mirroropd.research.studies.absorption_dynamics.geometry_plotting import setup  # noqa: E402

OUT = Path(__file__).resolve().parent
SOURCE = (
    ROOT
    / "examples/mirroropd/docs/experiments/teacher-transfer/checkpoint-dynamics/data/metric-synthesis-20261113.json"
)
BANKS = ["initial", "teacher"]
DOMAINS = ["math", "code", "if", "all"]
ENDS = [f"{b}/{d}/{s}" for s in ["absolute", "remaining"] for b in BANKS for d in DOMAINS]
COHORTS = ["all", "same_coordinate", "within_student", *identity.FAMILIES]
CORE = [
    "overlap64_16_final",
    "capture_optimal64_16_final",
    "overlap64_480_final",
    "late_increment64_anchor16",
    "student_rank90_fraction",
    "student_stable_rank",
    "student_update_sd",
    "student_weight_sd",
    "student_delta_sigma",
    "student_rho_sigma",
    "student_kappa_spec",
    "weight_frame_angle64",
    "weight_shape_residual",
    "student_global_amplitude",
    "teacher_global_amplitude",
    "teacher_rank90_fraction",
    "teacher_weight_sd",
    "teacher_spectrum_remaining",
    "parameter_norm_ratio",
    "parameter_cosine",
    "parameter_residual_energy_ratio",
    "teacher_update_overlap64",
    "progress_norm_step16",
    "progress_norm_step64",
]
EXTRA = [
    "late_increment_norm_per_update",
    "late_relative_weight",
    "late_relative_final",
    "frame64_16_max_angle_deg",
    "weight_frame_angle128",
    "student_rank95_fraction",
    "weight_entropy_rank",
    "weight_maximum_perturbation",
    "teacher_delta_sigma",
    "teacher_kappa_spec",
    "early_gap64",
    "final_gap64",
]


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")


def scalar(v):
    return v.get("median") if isinstance(v, dict) else v


def family(n):
    return identity.IDENTITIES[n]["family"]


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def centered(x, families):
    r = rankdata(x)
    for f in set(families):
        mask = np.array([v == f for v in families])
        r[mask] -= r[mask].mean()
    return r


def collect():
    raw = json.loads(SOURCE.read_text())
    pairs, feature_meta = {}, {}
    for n, p in raw["pairs"].items():
        assert p["expected"] == p["measured"] == len(p["matrices"]) == 196
        scopes = {"all": {k: scalar(v) for k, v in p["summary"].items()}}
        for group in ["modules", "depths"]:
            scopes.update({k: {f: scalar(v) for f, v in s.items()} for k, s in p[group].items()})
        for scope, values in scopes.items():
            ms = [
                m
                for m in p["matrices"]
                if scope == "all"
                or m["module"] == scope
                or (scope == "shallow" and m["layer"] <= 8)
                or (scope == "middle" and 9 <= m["layer"] <= 18)
                or (scope == "deep" and m["layer"] >= 19)
            ]
            for denominator, label in [("base_energy", "weight"), ("student_energy", "final")]:
                values[f"late_relative_{label}"] = float(
                    np.median(
                        [m["values"]["late_increment_norm_per_update"] / np.sqrt(m["values"][denominator]) for m in ms]
                    )
                )
            for i, step in enumerate([16, 32, 48, 64, 128, 192, 320, 480, 640]):
                if step != 640:
                    values[f"progress_norm_step{step}"] = float(
                        np.median(
                            [
                                np.sqrt(
                                    m["spectral_trajectory"][i]["update_energy"]
                                    / m["spectral_trajectory"][-1]["update_energy"]
                                )
                                for m in ms
                            ]
                        )
                    )
                for field in ms[0]["spectral_trajectory"][i]:
                    if field not in ["step", "update_energy"]:
                        vs = [m["spectral_trajectory"][i].get(field) for m in ms]
                        if all(v is not None for v in vs):
                            values[f"trajectory_{field}_step{step}"] = float(np.median(vs))
            for f, v in values.items():
                assert v is None or np.isfinite(v), (n, scope, f)
                feature_meta.setdefault(
                    f,
                    dict(
                        group="trajectory"
                        if f.startswith("trajectory_")
                        else "progress"
                        if f.startswith("progress_")
                        else "geometry"
                    ),
                )
        outcomes = {}
        for b in BANKS:
            for d in DOMAINS:
                o = p["rkl_dynamics"][f"{b}/{d}"]
                assert o["initial"] > 0 and o["final"] > 0
                outcomes[f"{b}/{d}/absolute"] = o["final"]
                outcomes[f"{b}/{d}/remaining"] = o["final"] / o["initial"]
                for field, key in [("initial", "baseline"), ("at128", "rkl128")]:
                    name = f"{key}_{b}_{d}"
                    scopes["all"][name] = o[field]
                    feature_meta[name] = dict(group="functional", endpoint_dependent=field != "initial")
        pairs[n] = dict(
            family=family(n),
            stage=identity.IDENTITIES[n]["teacher_stage"],
            student_group=identity.IDENTITIES[n]["student"],
            cross_size=n == "E102",
            scopes=scopes,
            outcomes=outcomes,
        )
    return dict(
        pairs=pairs,
        features=feature_meta,
        outcomes=ENDS,
        source=str(SOURCE.relative_to(ROOT)),
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    )


def rows_for(data, scope, feature, outcome, cohort):
    return [
        (n, p["scopes"].get(scope, {}).get(feature), p["outcomes"][outcome], p["student_group"])
        for n, p in data["pairs"].items()
        if p["scopes"].get(scope, {}).get(feature) is not None
        and (cohort != "same_coordinate" or not p["cross_size"])
        and (cohort not in identity.FAMILIES or p["family"] == cohort)
    ]


def measure(rows, cohort):
    x, y = [r[1] for r in rows], [r[2] for r in rows]
    rho = corr(x, y)
    if cohort == "within_student" and rho is not None:
        a, b = [centered(v, [r[3] for r in rows]) for v in [x, y]]
        rho = float(np.corrcoef(a, b)[0, 1]) if np.ptp(a) and np.ptp(b) else None
    return rho


def batch_correlations(rows, cohort):
    if len(rows) < 3:
        return [None] * len(ENDS)
    x = rankdata([r[1] for r in rows])
    y = rankdata(np.array([r[2] for r in rows]), axis=0)
    if cohort == "within_student":
        families = [r[3] for r in rows]
        for f in set(families):
            mask = np.array([v == f for v in families])
            x[mask] -= x[mask].mean()
            y[mask] -= y[mask].mean(axis=0)
    x -= x.mean()
    y -= y.mean(axis=0)
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y, axis=0))
    return [float(np.dot(x, y[:, j]) / denom[j]) if denom[j] > 1e-14 else None for j in range(len(ENDS))]


def screen(data):
    records = []
    for scope in data["pairs"]["E100"]["scopes"]:
        features = sorted({f for p in data["pairs"].values() for f in p["scopes"][scope]})
        for f in features:
            for cohort in COHORTS:
                selected = rows_for(data, scope, f, ENDS[0], cohort)
                rows = [(n, x, [data["pairs"][n]["outcomes"][o] for o in ENDS], fam) for n, x, _, fam in selected]
                rhos = batch_correlations(rows, cohort)
                loo = (
                    [batch_correlations(rows[:i] + rows[i + 1 :], cohort) for i in range(len(rows))]
                    if scope == "all"
                    else []
                )
                for j, outcome in enumerate(ENDS):
                    vs = [r[j] for r in loo if r[j] is not None]
                    records.append(
                        dict(
                            scope=scope,
                            feature=f,
                            outcome=outcome,
                            cohort=cohort,
                            n=len(rows),
                            rho=rhos[j],
                            loo_min=min(vs) if vs else None,
                            loo_max=max(vs) if vs else None,
                            pairs=",".join(r[0] for r in rows),
                        )
                    )
    with (OUT / "correlations.csv").open("w") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return records


def heatmap(features, scopes, lookup, cohort, name, caption):
    plt, fv = setup()
    labels = [f if s == "all" else f"{s}: {f}" for s, f in zip(scopes, features, strict=True)]
    values = np.array(
        [[lookup.get((s, f, o, cohort)) for o in ENDS] for s, f in zip(scopes, features, strict=True)], float
    )
    fig, ax = plt.subplots(figsize=(15, max(6, 0.25 * len(features) + 2.5)), layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0.065, 1, 0.90))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(fv.NEUTRALS["vmist"])
    im = ax.imshow(np.ma.masked_invalid(values), vmin=-1, vmax=1, aspect="auto", cmap=cmap)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    short = [
        f"{'I' if b == 'initial' else 'T'}:{d.upper()}\n{'abs' if t == 'absolute' else 'rel'}"
        for b, d, t in (o.split("/") for o in ENDS)
    ]
    ax.set_xticks(range(16), short, fontsize=8)
    for i in range(len(features)):
        for j in range(16):
            if np.isfinite(values[i, j]):
                v = values[i, j]
                ax.text(
                    j,
                    i,
                    f"{v:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if abs(v) > 0.65 else fv.NEUTRALS["vink"],
                )
    ax.axvline(7.5, color=fv.NEUTRALS["vink"], lw=1)
    fig.colorbar(im, ax=ax, label="Rank correlation", shrink=0.6)
    save_figure(fig, OUT / name, caption, caption_size=8)


def scatters(data, features, outcome, name):
    plt, fv = setup()
    fig, axs = plt.subplots(3, 4, figsize=(19, 13), layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0.06, 1, 0.84))
    for ax, f in zip(axs.flat, features, strict=False):
        points = []
        points = [
            (n, p["scopes"]["all"].get(f), p["outcomes"][outcome])
            for n, p in data["pairs"].items()
            if p["scopes"]["all"].get(f) is not None
        ]
        identity.scatter(ax, points, fv, size=40)
        xs = [r[1] for r in points]
        if xs and min(xs) > 0 and max(xs) / min(xs) > 25:
            ax.set_xscale("log")
        ys = [p["outcomes"][outcome] for p in data["pairs"].values()]
        ax.set(yscale="log", ylim=(min(ys) * 0.55, max(ys) * 1.8))
        ax.margins(x=0.22)
        ax.set_xlabel(f, fontsize=9)
        ax.set_ylabel(outcome, fontsize=9)
        r = corr(xs, [r[2] for r in points])
        same = [p for p in points if p[0] != "E102"]
        r2 = corr([p[1] for p in same], [p[2] for p in same])
        ax.set_title(f"r={r:+.2f}, n={len(points)}; without E102={r2:+.2f}", fontsize=10)
        place_labels(fig, ax, points)
    for ax in list(axs.flat)[len(features) :]:
        ax.set_visible(False)
    identity.legend(fig, data["pairs"], fv, top=True)
    save_figure(
        fig, OUT / name, "Every eligible pair is labeled. Teacher-coordinate metrics omit E102; no values imputed."
    )


def render(data, records):
    lookup = {(r["scope"], r["feature"], r["outcome"], r["cohort"]): r["rho"] for r in records}
    if "--data-only" in sys.argv:
        return
    if "--data-only" not in sys.argv:
        for cohort in ["all", "same_coordinate", "within_student"]:
            heatmap(
                EXTRA,
                ["all"] * len(EXTRA),
                lookup,
                cohort,
                f"new-candidates-{cohort}",
                "Late-update amplitude, weight directions and spectral gaps; all eligible pairs included.",
            )
        for bank in BANKS:
            for scale in ["absolute", "remaining"]:
                scatters(data, EXTRA, f"{bank}/math/{scale}", f"scatter-new-{bank}-{scale}")
        if "--extras-only" in sys.argv:
            return
    for cohort in ["all", "same_coordinate", "within_student"]:
        heatmap(
            CORE,
            ["all"] * len(CORE),
            lookup,
            cohort,
            f"overview-{cohort}",
            f"{cohort}; I=initial bank, T=teacher bank; abs=final RKL, rel=final/step0; pair-level summaries.",
        )
    features = sorted(f for f, m in data["features"].items() if m["group"] == "geometry")
    for i in range(0, len(features), 32):
        fs = features[i : i + 32]
        heatmap(
            fs,
            ["all"] * len(fs),
            lookup,
            "all",
            f"full-metrics-{i // 32 + 1:02d}",
            "All available pairs; I/T banks; abs=final, rel=final/step0. Undefined / constant correlations are gray.",
        )
    temporal = sorted(f for f, m in data["features"].items() if m["group"] in ["trajectory", "progress"])
    for i in range(0, len(temporal), 30):
        fs = temporal[i : i + 30]
        heatmap(
            fs,
            ["all"] * len(fs),
            lookup,
            "all",
            f"trajectory-{i // 30 + 1:02d}",
            "Saved-checkpoint trajectories versus final outcomes; endpoint-reference metrics are retrospective.",
        )
    fs = sorted(f for f, m in data["features"].items() if m["group"] == "functional")
    heatmap(
        fs,
        ["all"] * len(fs),
        lookup,
        "all",
        "functional-controls",
        "Baseline / step128 output gaps are functional comparators; step128 uses training-time evaluation.",
    )
    core = [
        "overlap64_16_final",
        "capture_optimal64_16_final",
        "student_rank90_fraction",
        "student_weight_sd",
        "teacher_amplitude",
        "teacher_update_overlap64",
    ]
    for kind, scopes in [
        ("modules", list(data["pairs"]["E100"]["scopes"])[1:8]),
        ("depths", ["shallow", "middle", "deep"]),
    ]:
        for i in range(0, len(core), 3):
            fs = [f for f in core[i : i + 3] for s in scopes]
            ss = [s for f in core[i : i + 3] for s in scopes]
            heatmap(
                fs,
                ss,
                lookup,
                "all",
                f"{kind}-{i // 3 + 1}",
                "One point per pair after module/depth aggregation; matrices are not independent replicates.",
            )
    for i in range(2):
        for b in BANKS:
            for scale in ["absolute", "remaining"]:
                scatters(data, CORE[12 * i : 12 * i + 12], f"{b}/math/{scale}", f"scatter-math-{b}-{scale}-{i + 1}")
    for b in BANKS:
        for d in ["code", "if", "all"]:
            scatters(data, CORE[:6] + CORE[14:20], f"{b}/{d}/absolute", f"scatter-{b}-{d}")


def gallery(data, records):
    data = {**data, "pairs": {n: {**p, "identity": identity.IDENTITIES[n]} for n, p in data["pairs"].items()}}
    # Embed every metric and point locally; selectors never require network access.
    reduced = [r for r in records if r["scope"] == "all"]
    payload = json.dumps(dict(data=data, stats=reduced), ensure_ascii=False).replace("</", "<\\/")
    pictures = "\n".join(
        f'<details><summary>{html.escape(p.stem)}</summary><a href="{p.stem}.pdf">PDF</a>'
        f'<img loading="lazy" src="{p.name}"></details>'
        for p in sorted(OUT.glob("*.png"))
    )
    template = (OUT / "viewer-template.html").read_text()
    (OUT / "index.html").write_text(template.replace("/*PAYLOAD*/", payload).replace("<!--GALLERY-->", pictures))


def main():
    OUT.mkdir(exist_ok=True)
    if "--render-only" in sys.argv:
        data = json.loads((OUT / "screen-data.json").read_text())
        with (OUT / "correlations.csv").open() as fp:
            records = list(csv.DictReader(fp))
        for r in records:
            r["rho"] = float(r["rho"]) if r["rho"] else None
            r["n"] = int(r["n"])
            for k in ["loo_min", "loo_max"]:
                r[k] = float(r[k]) if r[k] else None
    else:
        data = collect()
        write(OUT / "screen-data.json", data)
        records = screen(data)
    render(data, records)
    if "--data-only" in sys.argv:
        return
    gallery(data, records)
    write(
        OUT / "provenance.json",
        dict(
            source=data["source"],
            source_sha256=data["source_sha256"],
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            identity_sha256=hashlib.sha256(Path(identity.__file__).read_bytes()).hexdigest(),
            pairs=list(data["pairs"]),
            features=len(data["features"]),
            correlations=len(records),
            files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob("*.png")},
        ),
    )


if __name__ == "__main__":
    main()
