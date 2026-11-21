"""Replot existing FP64 measurements without model execution.

Ledger: colors identify student family and Qwen input mode throughout.
Marker shapes identify teacher stages; an outer ring marks cross-size scatter
points. Curve styles distinguish teacher stages within a row. Each scatter point is a pair,
not a matrix replicate. Spectral CDFs use all 28 k_proj matrices; other summaries
use all 196 matrices. CDF x is retained rank / available rank, y is squared
singular-value energy. Bands are matrix IQR, not sampling confidence intervals.
Selection is retrospective: low/high residual examples in each student family,
plus JustRL's cross-domain counterexample and the sole cross-size measured pair.
"""

import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file() and (p / "verl").is_dir())
sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pair_identity as identity  # noqa: E402

from examples.mirroropd.research.core.plotting import save_figure  # noqa: E402
from examples.mirroropd.research.studies.absorption_dynamics.geometry_plotting import setup  # noqa: E402

place_labels = identity.place_labels

OUT = Path(__file__).resolve().parent
DOC = ROOT / "examples/mirroropd/docs/experiments/teacher-transfer/checkpoint-dynamics"
GROUPS = [
    ("R1-Distill-Qwen-1.5B", ["E94", "E95", "E96"]),
    ("Qwen3-1.7B (thinking / non-thinking)", ["E100", "E101", "E102"]),
    ("Llama-3.2-3B-Instruct", ["E103", "E104"]),
]
LABELS = {
    "E94": "E94 DeepScaleR (RL)",
    "E95": "E95 JustRL (RL)",
    "E96": "E96 MiniMath (SFT)",
    "E100": "E100 e3 | RL | thinking",
    "E101": "E101 Samsung Math | SFT | thinking",
    "E102": "E102 4B-Instruct | non-thinking | cross-size",
    "E103": "E103 Co-rewarding (RL)",
    "E104": "E104 arithmetic (SFT candidate)",
}
ANCHORS = [16, 64, 192, 320, 480]
CDF_X = np.linspace(0, 1, 201)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def median(pair, field):
    value = pair["summary"].get(field)
    return value.get("median") if isinstance(value, dict) else value


def collect():
    source = DOC / "data/metric-synthesis-20261113.json"
    data = json.loads(source.read_text())
    selected = {n for _, names in GROUPS for n in names}
    pairs = {}
    for name, p in data["pairs"].items():
        assert p["expected"] == p["measured"] == len(p["matrices"]) == 196
        assert p["complete_layers"] == list(range(28))
        matrices = p["matrices"]
        progress = []
        for m in matrices:
            stages = m["spectral_trajectory"]
            assert [s["step"] for s in stages] == [16, 32, 48, 64, 128, 192, 320, 480, 640]
            progress.append([np.sqrt(s["update_energy"] / stages[-1]["update_energy"]) for s in stages])
        progress = np.asarray(progress)
        origin = "SFT" if name in ("E96", "E101", "E104") else "Cross-size" if name == "E102" else "RL"
        family = "R1" if int(name[1:]) <= 98 else "Qwen" if int(name[1:]) <= 102 else "Llama"
        row = dict(
            origin=origin,
            family=family,
            selected=name in selected,
            initial_rkl=p["rkl_dynamics"]["teacher/math"]["initial"],
            early_overlap=median(p, "overlap64_16_final"),
            early_capture=median(p, "capture_optimal64_16_final"),
            late_overlap=median(p, "overlap64_480_final"),
            early_progress=float(np.median(progress[:, 0])),
            rank90_fraction=median(p, "student_rank90_fraction"),
            weight_sd=median(p, "student_weight_sd"),
            teacher_amplitude=median(p, "teacher_global_amplitude"),
            student_amplitude=median(p, "student_global_amplitude"),
            teacher_rank90_fraction=median(p, "teacher_rank90_fraction"),
            parameter_residual=median(p, "parameter_residual_energy_ratio"),
            teacher_overlap=median(p, "teacher_update_overlap64"),
            parameter_cosine=median(p, "parameter_cosine"),
            parameter_norm_ratio=median(p, "parameter_norm_ratio"),
        )
        row["risks"] = [
            dict(
                step=e["optimizer_steps"],
                **{
                    f"{bank}_{domain}": e["metrics"][f"eval/rkl/public_{bank}/{domain}/question_mean"]
                    for bank in ("initial", "teacher")
                    for domain in ("math", "code", "if", "all")
                },
            )
            for e in p["risks"]
        ]
        row["outcomes"] = p["rkl_dynamics"]
        row["frames"] = {
            str(k): [dict(step=anchor, **p["summary"][f"overlap{k}_{anchor}_final"]) for anchor in ANCHORS]
            for k in (16, 32, 64, 128, 256)
        }
        row["topk"] = {
            role: [median(p, f"{role}_energy{k}") for k in (16, 32, 64, 128, 256)] for role in ("teacher", "final")
        }
        row["progress"] = dict(
            steps=[s["step"] for s in matrices[0]["spectral_trajectory"]], median=np.median(progress, axis=0).tolist()
        )
        row["sigma0"] = [x for x in data["sigma0_intervention"][name] if x["step"] == 640 and x["domain"] == "math"]
        pairs[name] = row
    # Exact spectrum shapes: match original compressed record hashes to synthesis.
    lookup = {(name, m["tensor"]): m["source_sha256"] for name, p in data["pairs"].items() for m in p["matrices"]}
    sources = [
        ROOT / "examples/mirroropd/research-evidence/absorption-parallel-20261112-v5" / name
        for name in ("geometry-optimized-back-attempt1", "geometry-optimized-front-attempt2")
    ]
    if any(not (directory / "protocol.json").is_file() for directory in sources):
        raise FileNotFoundError("The original matrix records are unavailable. Replot using comparison-data.json.")
    records, evidence = {}, []
    for directory in sources:
        protocol = json.loads((directory / "protocol.json").read_text())
        for index, task in enumerate(protocol["tasks"]):
            name, tensor = task["condition"], task["tensor"]
            if name not in selected or not tensor.endswith("self_attn.k_proj.weight"):
                continue
            path = directory / f"matrix-{index:05d}.json.gz"
            payload = path.read_bytes()
            sha = hashlib.sha256(payload).hexdigest()
            assert sha == lookup[name, tensor]
            raw = json.loads(gzip.decompress(payload))
            assert raw["task"] == task and raw["dtype"] == "float64"
            if name not in records:
                records[name] = dict(teacher=[], opd=[])
            for role, obj in (("teacher", raw.get("teacher")), ("opd", raw["stages"][-1])):
                if obj is None:
                    continue
                s = np.asarray(obj["singular_values"])
                assert np.all(np.diff(s) <= 1e-10) and np.all(s >= 0)
                assert np.isclose(np.sum(s * s), obj["energy"], rtol=1e-8)
                cdf = np.r_[0, np.cumsum(s * s) / np.sum(s * s)]
                records[name][role].append(np.interp(CDF_X, np.linspace(0, 1, len(cdf)), cdf))
            evidence.append(dict(path=str(path.relative_to(ROOT)), sha256=sha, pair=name, tensor=tensor))
    for name in selected:
        pairs[name]["cdf"] = {}
        for role, rows in records[name].items():
            assert len(rows) == (0 if name == "E102" and role == "teacher" else 28)
            if rows:
                pairs[name]["cdf"][role] = dict(
                    zip(("p25", "median", "p75"), np.quantile(rows, [0.25, 0.5, 0.75], axis=0).tolist(), strict=True)
                )
    return dict(
        pairs=pairs,
        cdf_x=CDF_X.tolist(),
        provenance=dict(
            synthesis=str(source.relative_to(ROOT)),
            synthesis_sha256=digest(source),
            raw_k_projection_records=evidence,
            arithmetic="existing exact FP64 SVD",
            selection="retrospective, fixed-student comparisons; cross-size is not an isolated origin intervention",
        ),
    )


def correlations(data):
    fields = [
        "initial_rkl",
        "early_overlap",
        "early_capture",
        "early_progress",
        "late_overlap",
        "rank90_fraction",
        "weight_sd",
        "teacher_amplitude",
        "teacher_rank90_fraction",
        "teacher_overlap",
        "parameter_residual",
    ]
    result = []
    for field in fields:
        for bank in ("initial", "teacher"):
            for domain in ("math", "all"):
                rows = [p for p in data["pairs"].values() if p[field] is not None]
                x = np.array(
                    [p["outcomes"][f"{bank}/{domain}"]["initial"] if field == "initial_rkl" else p[field] for p in rows]
                )
                y = np.array([p["outcomes"][f"{bank}/{domain}"]["final"] for p in rows])
                relative = y / np.array([p["outcomes"][f"{bank}/{domain}"]["initial"] for p in rows])
                loo = [float(spearmanr(np.delete(x, i), np.delete(y, i)).statistic) for i in range(len(x))]
                result.append(
                    dict(
                        feature=field,
                        bank=bank,
                        domain=domain,
                        n=len(x),
                        spearman=float(spearmanr(x, y).statistic),
                        spearman_relative_remaining=float(spearmanr(x, relative).statistic),
                        leave_one_out_min=min(loo),
                        leave_one_out_max=max(loo),
                    )
                )
    # Descriptive rank residualization, not a causal adjustment or significance test.
    partial = []
    for field in ("early_overlap", "early_capture"):
        for bank in ("initial", "teacher"):
            rows = list(data["pairs"].values())
            for controls in (("initial_distance",), ("initial_distance", "early_progress")):
                z = [np.ones(len(rows))]
                for control in controls:
                    v = [
                        p["outcomes"][f"{bank}/math"]["initial"] if control == "initial_distance" else p[control]
                        for p in rows
                    ]
                    z.append(rankdata(v))
                z = np.array(z).T
                x = rankdata([p[field] for p in rows])
                y = rankdata([p["outcomes"][f"{bank}/math"]["final"] for p in rows])
                xr, yr = [v - z @ np.linalg.lstsq(z, v, rcond=None)[0] for v in (x, y)]
                partial.append(
                    dict(
                        feature=field,
                        bank=bank,
                        controls=controls,
                        n=len(rows),
                        rank_residual_correlation=float(np.corrcoef(xr, yr)[0, 1]),
                    )
                )
    return dict(spearman=result, descriptive_partial=partial)


def layout(rows, cols, width=15, height=11):
    plt, _ = setup()
    fig, axes = plt.subplots(rows, cols, figsize=(width, height), layout="constrained", squeeze=False)
    fig.get_layout_engine().set(rect=(0, 0.06, 1, 0.94))
    return fig, axes


def row_colors(names, fv):
    return {n: fv.HUES[identity.FAMILIES[identity.IDENTITIES[n]["family"]]] for n in names}


def row_style(name):
    return dict(
        marker=identity.STAGES[identity.IDENTITIES[name]["teacher_stage"]],
        markersize=3,
        linestyle=["-", "--", "-."][int(name[1:]) % 3],
    )


def render(data, stats):
    plt, fv = setup()
    pairs = data["pairs"]
    fig, axes = layout(3, 3)
    for row, (family, names) in enumerate(GROUPS):
        colors = row_colors(names, fv)
        for name in names:
            p = pairs[name]
            for col, bank in enumerate(("initial", "teacher")):
                ax = axes[row, col]
                ax.plot(
                    [r["step"] for r in p["risks"]],
                    [r[f"{bank}_math"] for r in p["risks"]],
                    color=colors[name],
                    label=LABELS[name],
                    **row_style(name),
                )
                ax.set(yscale="log", ylim=(1e-4, 2), xlabel="Optimizer updates", ylabel="Math held-out RKL")
                ax.set_title(f"{family}\n{bank.capitalize()} bank", fontsize=11)
            ax = axes[row, 2]
            series = p["frames"]["64"]
            ax.plot(
                ANCHORS + [640],
                [r["median"] for r in series] + [1],
                color=colors[name],
                **row_style(name),
                label=LABELS[name],
            )
            ax.fill_between(
                ANCHORS, [r["p25"] for r in series], [r["p75"] for r in series], color=colors[name], alpha=0.10
            )
            ax.set(xlabel="Optimizer updates", ylabel="Adjusted right-frame overlap", ylim=(0, 1.03))
            ax.set_title(f"{family}\nCumulative update vs its step640, k64", fontsize=11)
        axes[row, 2].legend(fontsize=8, loc="lower right")
    save_figure(
        fig,
        OUT / "01-absorption-and-direction",
        "640 updates, seed42. Frame: 196-matrix median / IQR; reference is each run's own final update.",
    )

    fig, axes = layout(3, 3)
    for row, (family, names) in enumerate(GROUPS):
        colors = row_colors(names, fv)
        for col, role in enumerate(("teacher", "opd")):
            ax = axes[row, col]
            for name in names:
                cdf = pairs[name]["cdf"].get(role)
                if not cdf:
                    continue
                ax.plot(
                    CDF_X, cdf["median"], color=colors[name], label=LABELS[name], linestyle=row_style(name)["linestyle"]
                )
                ax.fill_between(CDF_X, cdf["p25"], cdf["p75"], color=colors[name], alpha=0.10)
            ax.plot([0, 1], [0, 1], color=fv.NEUTRALS["baseline"], ls=":", lw=0.8, label="Equal energy per direction")
            ax.set(
                xlabel="Retained rank / available rank",
                ylabel="Cumulative squared singular-value energy",
                xlim=(0, 1),
                ylim=(0, 1.02),
            )
            title = "Teacher displacement" if role == "teacher" else "Final OPD update"
            ax.set_title(f"{family}\n{title} spectrum", fontsize=11)
        ax = axes[row, 2]
        for i, name in enumerate(names):
            p = pairs[name]
            ax.scatter(
                100 * p["student_amplitude"],
                i,
                color=colors[name],
                s=45,
                marker=identity.STAGES[identity.IDENTITIES[name]["teacher_stage"]],
            )
            if p["teacher_amplitude"] is not None:
                ax.scatter(100 * p["teacher_amplitude"], i, color=colors[name], s=60, marker="x")
                ax.plot(
                    100 * np.array([p["student_amplitude"], p["teacher_amplitude"]]), [i, i], color=colors[name], lw=1
                )
        ax.set_yticks(range(len(names)), names)
        ax.set(xscale="log", xlim=(0.03, 7), xlabel="Displacement / initial weight norm (%)")
        ax.set_title(f"{family}\nGlobal norm: stage marker OPD; x teacher", fontsize=11)
        axes[row, 0].legend(fontsize=8, loc="lower right")
        axes[row, 1].legend(fontsize=8, loc="lower right")
    save_figure(
        fig,
        OUT / "02-spectra-and-amplitude",
        "CDF: 28 k_proj matrices, median / IQR. Norm: 196 matrices combined. E102 teacher delta is undefined.",
    )

    fields = [
        ("initial_rkl", "Initial teacher-bank Math RKL", True),
        ("early_overlap", "Step16 to final frame overlap, k64", False),
        ("early_capture", "Step16 captures final energy / optimal", False),
        ("early_progress", "Step16 / final update norm", False),
        ("rank90_fraction", "Final update rank90 / available rank", False),
        ("teacher_amplitude", "Teacher displacement / initial weight norm", True),
        ("teacher_overlap", "OPD vs teacher update-frame overlap, k64", False),
        ("parameter_residual", "Final parameter residual energy / initial", False),
    ]
    fig, axes = layout(2, 4, width=19, height=10)
    fig.get_layout_engine().set(rect=(0, 0.06, 1, 0.86))
    for ax, (field, label, log) in zip(axes.flat, fields, strict=True):
        points = []
        identity.scatter(
            ax,
            [(n, p[field], p["outcomes"]["teacher/math"]["final"]) for n, p in pairs.items() if p[field] is not None],
            fv,
            size=45,
        )
        for name, p in pairs.items():
            if p[field] is None:
                continue
            y = p["outcomes"]["teacher/math"]["final"]
            points.append((name, p[field], y))
        ax.set(yscale="log", ylim=(1e-4, 0.8), xlabel=label, ylabel="Final teacher-bank Math RKL")
        if log:
            ax.set_xscale("log")
        ax.margins(x=0.18)
        rho = next(
            r for r in stats["spearman"] if r["feature"] == field and r["bank"] == "teacher" and r["domain"] == "math"
        )
        ax.set_title(f"Spearman r = {rho['spearman']:+.2f}; n = {rho['n']}", fontsize=11)
        place_labels(fig, ax, points)
    identity.legend(fig, pairs, fv, top=True)
    save_figure(
        fig,
        OUT / "03-pair-level-associations",
        "One point per pair. All 196 matrices summarized per pair; descriptive associations, not causal effects.",
    )

    selected = [name for _, names in GROUPS for name in names]
    fig, axes = layout(1, 3, width=16, height=7.2)
    fig.get_layout_engine().set(rect=(0, 0.08, 1, 0.79))
    for bank, marker in (("initial", "o"), ("teacher", "s")):
        color = fv.series_colors("paper", 2)[bank == "teacher"]
        offset = -0.12 if bank == "initial" else 0.12
        values = [next(x for x in pairs[name]["sigma0"] if x["bank"] == bank) for name in selected]
        axes[0, 0].scatter(
            np.arange(len(selected)) + offset,
            [s["delta"] for s in values],
            color=color,
            marker=marker,
            s=35,
            label=f"{bank} bank",
        )
        axes[0, 1].scatter(
            np.arange(len(selected)) + offset,
            [100 * s["relative_change"] for s in values],
            color=color,
            marker=marker,
            s=35,
            label=f"{bank} bank",
        )
    for ax, ylabel in zip(axes[0, :2], ("Edited - original Math RKL", "Math RKL change (%)"), strict=True):
        ax.axhline(0, color=fv.NEUTRALS["baseline"], lw=0.8)
        ax.set_xticks(range(len(selected)), selected, rotation=45)
        ax.set(ylabel=ylabel)
        ax.legend(loc="upper left", fontsize=9)
    axes[0, 0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[0, 0].set_title("Restore initial singular values: absolute effect", fontsize=11)
    axes[0, 1].set_title("Restore initial singular values: relative effect", fontsize=11)
    identity.scatter(
        axes[0, 2], [(n, p["weight_sd"], p["outcomes"]["teacher/math"]["final"]) for n, p in pairs.items()], fv, size=40
    )
    identity.legend(fig, pairs, fv, top=True)
    axes[0, 2].set(
        xscale="log",
        yscale="log",
        xlabel="Complete-weight normalized spectrum drift",
        ylabel="Final teacher-bank Math RKL",
        ylim=(1e-4, 0.8),
    )
    axes[0, 2].margins(x=0.18)
    axes[0, 2].set_title("Spectrum drift and outcome (11 pairs)", fontsize=11)
    place_labels(
        fig, axes[0, 2], [(n, p["weight_sd"], p["outcomes"]["teacher/math"]["final"]) for n, p in pairs.items()]
    )
    save_figure(
        fig,
        OUT / "04-spectrum-restoration",
        "Paired HF evaluation at step640; existing measurements. Right-panel colors / shapes follow figure 03.",
    )


def export_csv(data):
    scalar_keys = [k for k, v in data["pairs"]["E100"].items() if not isinstance(v, dict | list)]
    with (OUT / "pair-summary.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=["pair", *scalar_keys, "math_initial_final", "math_teacher_final"])
        writer.writeheader()
        for name, p in data["pairs"].items():
            writer.writerow(
                dict(
                    pair=name,
                    **{k: p[k] for k in scalar_keys},
                    math_initial_final=p["outcomes"]["initial/math"]["final"],
                    math_teacher_final=p["outcomes"]["teacher/math"]["final"],
                )
            )


def main():
    output = OUT / "comparison-data.json"
    if "--recollect" in sys.argv or not output.exists():
        data = collect()
        write_json(output, data)
    else:
        data = json.loads(output.read_text())
    for n, p in data["pairs"].items():
        p.update(identity.IDENTITIES[n])
        p["origin"] = p["teacher_stage"]
    write_json(output, data)
    stats = correlations(data)
    write_json(OUT / "correlations.json", stats)
    export_csv(data)
    render(data, stats)
    write_json(
        OUT / "provenance.json",
        dict(
            source=data["provenance"],
            script_sha256=digest(Path(__file__)),
            identity_sha256=digest(Path(identity.__file__)),
            comparison_data_sha256=digest(output),
            numpy=np.__version__,
            selected_pairs=[n for _, ns in GROUPS for n in ns],
            files={p.name: digest(p) for p in OUT.glob("*.png")},
        ),
    )


if __name__ == "__main__":
    main()
