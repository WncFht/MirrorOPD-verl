"""Weekly section 4: freeze existing evidence and render pair-level comparisons.

Ledger: one point is one trained pair. Color is student family/input protocol;
marker is teacher stage; an outer ring is cross-size. X is dimensionless teacher
projection amplitude or a median update-spectrum metric; Y is held-out Math RKL
or relative reduction. Final RKL uses log Y; relative reduction uses linear Y.
The fitted curve is OLS in natural-log coordinates, restricted to observed X.
Six family hues and stage combinations are intentional; hue-budget and
palette-overflow checks are exempted while other audits run.
No training or GPU measurement runs here. Frozen inputs support replotting.
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from scipy.stats import spearmanr

REPO = next(p for p in Path(__file__).resolve().parents if (p / "examples/mirroropd/docs").is_dir())
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--refresh-snapshot", action="store_true")
args = parser.parse_args()
STYLE = REPO.parent / "dotfiles/.agents/skills/figure-viz/assets/mpl"
sys.path.insert(0, str(STYLE))
import fvstyle as fv  # noqa: E402

PROGRESS = REPO / "examples/mirroropd/docs/progress"
DATA = PROGRESS / "data/20261116-section4"
OUT = PROGRESS / "figures/20261116-section4"
SOURCE = REPO / "examples/mirroropd/docs/experiments/teacher-transfer/checkpoint-dynamics/analyses"
DATA.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
INPUTS = [
    "pairs.json",
    "priority-panel-pairs.json",
    "priority-panel-correlations.csv",
    "group-confounding.json",
    "fit-diagnostics.json",
    "correlations.csv",
]
files = []
for name in INPUTS:
    src = SOURCE / "expanded-lightweight" / name
    dest = DATA / name
    if args.refresh_snapshot or not dest.exists():
        shutil.copyfile(src, dest)
    files.append(
        dict(
            source=str(src.relative_to(REPO)),
            snapshot=str(dest.relative_to(REPO)),
            sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
        )
    )
if args.refresh_snapshot or not (DATA / "pair_identity.py").exists():
    shutil.copyfile(SOURCE / "pair_identity.py", DATA / "pair_identity.py")
if Path(__file__).resolve() != DATA / "plot_section4.py":
    shutil.copyfile(__file__, DATA / "plot_section4.py")
sys.path.insert(0, str(DATA))
sys.path.insert(0, str(REPO))
import pair_identity as identity  # noqa: E402

pairs = json.loads((DATA / "pairs.json").read_text())
panel = json.loads((DATA / "priority-panel-pairs.json").read_text())
fv.apply_rc("paper")


def finish(fig, name, names, extra=()):
    identity.legend(fig, names, fv, extra=extra)
    fig.text(
        0.5,
        0.015,
        "One point per pair; fixed-bank question-mean Math RKL",
        ha="center",
        color=fv.NEUTRALS["baseline"],
        fontsize=9,
    )
    fig.canvas.draw()
    (OUT / f"{name}.audit.txt").write_text(str(fv.audit(fig, ignore=["hue-budget", "palette-overflow"])) + "\n")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=180)
    plt.close(fig)


names = [n for n, p in pairs.items() if "teacher_global_amplitude" in p["features"]]
fig, axes = plt.subplots(2, 2, figsize=(8.2, 9.5))
fig.subplots_adjust(left=0.09, right=0.97, bottom=0.22, top=0.94, hspace=0.48, wspace=0.34)
fits = {}
for col, bank in enumerate(("teacher", "initial")):
    for row, target in enumerate(("final", "relative")):
        ax = axes[row, col]
        points = [
            (n, pairs[n]["features"]["teacher_global_amplitude"], pairs[n]["outcomes"][f"{bank}/math"][target])
            for n in names
        ]
        identity.scatter(ax, points, fv)
        ax.set_xscale("log")
        x = np.array([p[1] for p in points])
        y = np.array([p[2] for p in points])
        rho = float(spearmanr(x, y).statistic)
        if row == 0:
            ax.set_yscale("log")
            design = np.column_stack([np.ones(len(x)), np.log(x)])
            beta = np.linalg.lstsq(design, np.log(y), rcond=None)[0]
            r2 = float(1 - np.square(np.log(y) - design @ beta).sum() / np.square(np.log(y) - np.log(y).mean()).sum())
            grid = np.geomspace(x.min(), x.max(), 100)
            ax.plot(grid, np.exp(beta[0]) * grid ** beta[1], color=fv.NEUTRALS["vink"], ls="--", lw=1.2)
            fits[bank] = dict(
                n=len(names), spearman=rho, scale=float(np.exp(beta[0])), exponent=float(beta[1]), log_r2=r2
            )
            ax.set_title(f"{bank.capitalize()} bank: rho={rho:.3f}\nlog-R²={r2:.3f}", fontsize=11)
            ax.set_ylabel("Final Math RKL")
            ax.set_ylim(6e-5, 1.1)
        else:
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.set_title(f"{bank.capitalize()} bank: rho={rho:.3f}, n={len(names)}", fontsize=11)
            ax.set_ylim(0, 1.16)
            ax.set_ylabel("Relative RKL reduction")
        ax.set_xlabel("Teacher global amplitude")
        ax.margins(x=0.18, y=0.22)
        identity.place_labels(
            fig,
            ax,
            [p for p in points if p[0] in {"E94", "E96", "E103", "E105", "E106", "E107", "E111", "E113", "E117"}],
        )
# Move two labels away from the fitted curve and the nearby unlabelled point.
for ax, label, offset in [(axes[0, 0], "E111", (-20, -22)), (axes[0, 1], "E113", (0, -22))]:
    for annotation in ax.texts:
        if annotation.get_text() == label:
            annotation.set_position(offset)
            annotation.set_ha("right" if label == "E111" else "center")

finish(
    fig,
    "amplitude-heldout",
    names,
    extra=[Line2D([], [], color=fv.NEUTRALS["vink"], ls="--", label="Final RKL: log-OLS fit")],
)
fig, axes = plt.subplots(1, 2, figsize=(8.2, 6.7))
fig.subplots_adjust(left=0.08, right=0.97, bottom=0.32, top=0.88, wspace=0.28)
for ax, metric, label in zip(
    axes,
    ["overlap64_16_final", "student_stable_rank"],
    ["Early frame overlap (k64)", "Final update stable rank"],
    strict=True,
):
    points = [(n, panel[n][metric], pairs[n]["outcomes"]["teacher/math"]["relative"]) for n in pairs]
    identity.scatter(ax, points, fv)
    ax.set_xlabel(label)
    ax.set_ylabel("Relative RKL reduction")
    ax.set_ylim(0, 1.17)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.margins(x=0.16, y=0.18)
    ax.set_title(f"All 22 pairs: rho={spearmanr([p[1] for p in points], [p[2] for p in points]).statistic:.3f}")
    identity.place_labels(fig, ax, [p for p in points if p[0] in {"E105", "E106", "E108", "E110", "E117"}])
# Place E105 outside the aggregate bounding box of the cross-size ring collection.
for annotation in axes[0].texts:
    if annotation.get_text() == "E105":
        annotation.set_position((-75, 8))
        annotation.set_ha("right")

finish(fig, "frame-spectrum-heldout", list(pairs))
(DATA / "plot-fit-results.json").write_text(json.dumps(fits, indent=2) + "\n")
for p in [
    DATA / "pair_identity.py",
    DATA / "plot_section4.py",
    DATA / "plot-fit-results.json",
    *OUT.glob("*.png"),
    *OUT.glob("*.pdf"),
]:
    files.append(dict(snapshot=str(p.relative_to(REPO)), sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(DATA / "sources.json").write_text(
    json.dumps(
        dict(updated="2026-11-16", scope="Weekly section 4 only; existing measurements", files=files),
        indent=2,
        ensure_ascii=False,
    )
    + "\n"
)
print(json.dumps(fits, indent=2))
