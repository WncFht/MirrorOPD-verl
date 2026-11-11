"""Teacher-band learning, per-question late risk and output diagnostics.

Student/teacher parameter directions are paired Frobenius-orthogonal SVD
atoms, independent of base thin-SVD diag/off/null terminology. Error bars
resample paired questions, not training seeds. All original attempts retained.
"""

import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "spectrum-expanded-v4"
BAND = ROOT / "teacher-band-capture-v2"
OUT = ROOT / "curve-spectrum-v1" / "band-analysis-v3"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT.parents[2] / "dotfiles/.agents/skills/figure-viz/assets/mpl"))
import fvstyle as fv  # noqa: E402


def save(fig, name, ignore=None):
    fig.canvas.draw()
    audit = [str(x) for x in fv.audit(fig, print_slot=12, scoped_ignore=ignore or [])]
    (OUT / (name + ".audit.json")).write_text(json.dumps(audit))
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT / (name + "." + ext), dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(name, audit, flush=True)


def main():
    assert (BAND / "complete.json").exists()
    inventory = json.loads((SRC / "inventory.json").read_text())["runs"]
    ancillary = json.loads((SRC / "ancillary-complete.json").read_text())["conditions"]
    byrun = {r["run_id"]: r for r in inventory + ancillary}
    bands = []
    stages = []
    for file in BAND.glob("*/*.json"):
        m = json.loads(file.read_text())
        r = byrun[m["run_id"]]
        for s in m["stages"]:
            row = dict(
                run_id=r["run_id"],
                student=r.get("student", r["run_id"]),
                teacher=r.get("teacher", r["run_id"]),
                tensor=m["tensor"],
                step=s["step"],
            )
            row.update(
                {
                    k: s[k]
                    for k in [
                        "student_energy",
                        "teacher_energy",
                        "missing_energy",
                        "matched_fraction",
                        "off_fraction",
                        "outside_fraction",
                    ]
                }
            )
            row["matched_enrichment"] = (
                s["matched_energy"] / s["isotropic_matched_expectation"] if s["isotropic_matched_expectation"] else None
            )
            row["student_teacher_norm_ratio"] = np.sqrt(s["student_energy"] / s["teacher_energy"])
            stages.append(row)
            for i, b in enumerate(s["bands"]):
                bands.append(row | {"band": i, "total_teacher_energy": row["teacher_energy"]} | b)
    bd = pd.DataFrame(bands)
    sd = pd.DataFrame(stages)
    bd.to_csv(OUT / "teacher-band-matrices.csv", index=False)
    sd.to_csv(OUT / "teacher-frame-matrices.csv", index=False)
    # Pair summary shares physical direction across modes; original run risk stays separate.
    summary = bd.groupby(["run_id", "student", "teacher", "step", "band"]).median(numeric_only=True).reset_index()
    summary.to_csv(OUT / "teacher-band-summary.csv", index=False)
    sm = sd.groupby(["run_id", "student", "teacher", "step"]).median(numeric_only=True).reset_index()
    sm.to_csv(OUT / "teacher-frame-summary.csv", index=False)
    rng = np.random.default_rng(20261111)
    late = []
    outputs = []
    for r in inventory:
        folder = Path(r["initial_source"]).parent
        files = {step: folder / f"fixed-{step:03d}.json" for step in [0, 64, 256, 320]}
        data = {step: json.loads(p.read_text()) for step, p in files.items() if p.exists()}
        for step, recs in data.items():
            held = [x for x in recs if x["split"] == "heldout"]
            if not held:
                held = [
                    x for x in recs if x["group"] in ["M01", "M02", "M03", "M04", "Code01", "Code02", "IF01", "IF02"]
                ]
            for domain in ["all", "math", "code", "if"]:
                subset = [x for x in held if domain == "all" or x["domain"] == domain]
                if not subset:
                    continue
                for window in ["all", "early", "middle", "late"]:
                    stats = [x["stats"][window] for x in subset if window in x["stats"]]
                    if not stats:
                        continue
                    keys = set.intersection(*(set(x) for x in stats)) - {"tokens"}
                    for metric in sorted(keys):
                        vals = [x[metric] / x["tokens"] for x in stats if x["tokens"]]
                        if vals:
                            outputs.append(
                                dict(
                                    run_id=r["run_id"],
                                    student=r["student"],
                                    teacher=r["teacher"],
                                    step=step,
                                    domain=domain,
                                    window=window,
                                    metric=metric,
                                    value=float(np.mean(vals)),
                                    questions=len(vals),
                                )
                            )
        if not all(s in data for s in [256, 320]):
            continue
        keyed = {s: {(x["group"], x["id"]): x for x in data[s]} for s in [256, 320]}
        assert set(keyed[256]) == set(keyed[320])
        for domain in ["all", "math", "code", "if"]:
            changes = []
            old = []
            for key, a in keyed[256].items():
                if a["group"] not in ["M01", "M02", "M03", "M04", "Code01", "Code02", "IF01", "IF02"]:
                    continue
                if domain != "all" and a["domain"] != domain:
                    continue
                b = keyed[320][key]
                changes.append((a["stats"]["all"]["rkl"] - b["stats"]["all"]["rkl"]) / 1024)
                old.append(a["stats"]["all"]["rkl"] / 1024)
            if not changes:
                continue
            d = np.array(changes)
            boot = d[rng.integers(0, len(d), (2000, len(d)))].mean(1)
            ci = np.quantile(boot, [0.025, 0.975])
            late.append(
                dict(
                    run_id=r["run_id"],
                    student=r["student"],
                    teacher=r["teacher"],
                    domain=domain,
                    questions=len(d),
                    absolute_drop=float(d.mean()),
                    relative_drop=float(d.mean() / np.mean(old)),
                    ci_low=float(ci[0]),
                    ci_high=float(ci[1]),
                    fraction_questions_improved=float((d > 0).mean()),
                    mean_abs_question_change=float(abs(d).mean()),
                    net_change_retention=float(abs(d.sum()) / abs(d).sum()) if abs(d).sum() else None,
                )
            )
    pd.DataFrame(outputs).to_csv(OUT / "output-diagnostics.csv", index=False)
    ld = pd.DataFrame(late)
    ld.to_csv(OUT / "late-question-paired.csv", index=False)
    fv.apply_rc("paper", fontsize=10)
    colors = fv.series_colors("paper", 4)
    from matplotlib.colors import to_rgb

    labels = sorted({r["teacher"] for r in inventory if r["architecture_compatible"]})
    registry = {
        label: tuple(v * (1 - 0.18 * (i // 4)) for v in to_rgb(colors[i % 4])) for i, label in enumerate(labels)
    }
    fv.register_colors(*registry.values())
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.5), sharey=True)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.23, top=0.84, wspace=0.25)
    for ax, student, title in zip(
        axes,
        ["r1", "q3-1.7", "q3-1.7-nonthinking"],
        ["R1 paired directions", "Qwen thinking", "Qwen non-thinking"],
        strict=True,
    ):
        # Duplicated input modes use identical weights; recover by teacher identity from inventory.
        rows = [r for r in inventory if r["student"] == student and r["architecture_compatible"]]
        for i, r in enumerate(rows):
            ident = hashlib.sha256((r["base"] + "|" + r["teacher_path"] + "|" + r["run_id"]).encode()).hexdigest()[:16]
            records = []
            for file in (BAND / ident).glob("*.json"):
                m = json.loads(file.read_text())
                records.extend(s for s in m["stages"] if s["step"] == 320)
            if not records:
                continue
            gains = np.median([[b["amplitude_gain"] for b in s["bands"]] for s in records], axis=0)
            ax.plot(
                range(4),
                gains,
                marker="o",
                label=r["teacher"],
                color=registry[r["teacher"]],
                linestyle="--" if i >= 4 else "-",
            )
        ax.set(
            xlabel="Teacher-delta singular band",
            ylabel="Signed amplitude gain",
            title=title,
        )
        ax.set_yscale("symlog", linthresh=0.001)
        ax.set_xticks(range(4), ["0–1%", "1–5%", "5–20%", "20–100%"], rotation=30)
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=2, frameon=False)
    fig.text(
        0.5,
        0.025,
        "Gain = <student update, teacher band> / ||teacher band||²; gain 1 recovers band amplitude",
        ha="center",
        fontsize=8,
    )
    save(fig, "teacher-band-amplitude", [("palette-overflow", ax) for ax in axes])
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.5))
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.28, top=0.91, wspace=0.3)
    for ax, student in zip(axes, ["r1", "q3-1.7", "q3-1.7-nonthinking"], strict=True):
        allowed = {r["teacher"] for r in inventory if r["student"] == student and r["architecture_compatible"]}
        a = ld[(ld.student == student) & (ld.domain == "all") & ld.teacher.isin(allowed)].copy()
        a = a.sort_values("absolute_drop")
        ax.errorbar(
            a.absolute_drop,
            range(len(a)),
            xerr=[a.absolute_drop - a.ci_low, a.ci_high - a.absolute_drop],
            fmt="o",
            color=colors[0],
            capsize=3,
        )
        ax.set_yticks(range(len(a)), a.teacher)
        ax.set(xlabel="Absolute RKL drop, updates256→320", ylabel="Teacher", title=student)
        ax.axvline(0, color=fv.NEUTRALS["baseline"], lw=0.7)
        ax.set_ylim(-0.5, 5.5)
    fig.text(
        0.5,
        0.04,
        "Paired-question bootstrap95%; original fixed H1024; independent training-seed variability not estimated",
        ha="center",
        fontsize=8,
    )
    save(fig, "late-absolute-improvement", [("legend-coverage", ax) for ax in axes])
    print(
        "complete", len(bd), "band records", len(outputs), "output metrics", len(late), "late paired panels", flush=True
    )


if __name__ == "__main__":
    main()
