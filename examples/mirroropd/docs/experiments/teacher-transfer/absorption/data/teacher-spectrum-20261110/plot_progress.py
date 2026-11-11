"""Plot measured spectra and OPD trajectories from published result artifacts.

Semantic ledger: arm colors are stable across figures; within-arm SVD plots use
neutral base, teal teacher, and orange student. Dashed lines denote references.
Geometric ledger: fixed grids, shared fraction limits, external figure legends,
reserved footer; no annotations over data. Every figure is audited after draw.
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = next(p for p in Path(__file__).resolve().parents if (p / "examples/mirroropd").is_dir())
ROOT = REPO / "runs/teacher-spectrum-followup-20261110"
STYLE = REPO.parent / "dotfiles/.agents/skills/figure-viz/assets/mpl"
sys.path.insert(0, str(STYLE))
import fvstyle as fv  # noqa: E402

OUT = REPO / "examples/mirroropd/docs/experiments/teacher-transfer/figures/teacher-spectrum-progress-20261110"
OUT.mkdir(parents=True, exist_ok=True)
fv.apply_rc("paper", fontsize=10)
ARM = ["qwen-grpo", "qwen-sft", "r1-justrl", "r1-polaris"]
NAMES = ["Qwen GRPO", "Qwen SFT", "R1 JustRL", "R1 / Polaris"]
COLORS = [fv.HUES[k] for k in ["vteal", "vorange", "vcoral", "vviolet"]]
TEAL, ORANGE, GRAY = COLORS[0], COLORS[1], fv.NEUTRALS["baseline"]
inputs = {}


def read(path):
    raw = path.read_bytes()
    inputs[str(path.relative_to(REPO))] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


report = read(ROOT / "progress-analysis.json")
temporal = {a: [read(ROOT / f"temporal-svd/{a}-l{layer:02d}.json") for layer in [0, 13, 27]] for a in ARM[:3]}


def save(fig, name, note):
    fig.text(0.08, 0.015, note, color=GRAY, fontsize=9)
    fig.canvas.draw()
    # Digital document figures are displayed at native width; no 5.5in print shrink.
    scoped = [("facet-ylim-mismatch", i) for i in range(6)] if name == "weight-spectra" else []
    problems = fv.audit(fig, print_slot=float(fig.get_figwidth()), scoped_ignore=scoped)
    (OUT / f"{name}.audit.json").write_text(json.dumps(problems, indent=2, default=str))
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(OUT / f"{name}.{ext}", dpi=170)
    plt.close(fig)
    print(name, "audit", len(problems), flush=True)


fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for i, (arm, name, color) in enumerate(zip(ARM, NAMES, COLORS, strict=True)):
    ax = axes[i // 2]
    risk = report["pairs"][arm]["native_risk"]
    for split, ls in [("train", "--"), ("heldout", "-")]:
        rows = [
            (0 if k.endswith("/base") else int(k.rsplit("/s", 1)[1]), v)
            for k, v in risk.items()
            if k.startswith(f"{split}/all/")
        ]
        rows.sort()
        ax.plot(
            [s for s, _ in rows],
            [v["remaining"] for _, v in rows],
            ls,
            marker="o",
            color=color,
            label=f"{name}: {split}",
        )
    ax.set(xlabel="Training step", ylabel="Native KL / initial native KL", ylim=(0, 1.06))
    ax.grid(axis="y", alpha=0.2)
for ax, title in zip(axes, ["Qwen bank: 80 train / 80 heldout", "R1 bank: 160 train / 160 heldout"], strict=True):
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.14), ncol=2)
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.19, top=0.72, wspace=0.3)
save(
    fig,
    "native-risk",
    "Fixed H=1024; full vocabulary. Banks differ across panels; this is distribution matching, not accuracy.",
)

fig, axes = plt.subplots(2, 3, figsize=(12, 7))
for j, arm in enumerate(ARM[:3]):
    d = temporal[arm][1]  # same predeclared layer 13
    b = np.array(d["base_singular_values"])
    t = np.array(d["teacher_singular_values"])
    s = np.array(d["stages"][-1]["weight_singular_values"])
    rank = np.arange(1, len(b) + 1)
    ax = axes[0, j]
    for y, c, label, ls in [(b, GRAY, "Base", "-"), (t, TEAL, "Teacher", "--"), (s, ORANGE, "Student step 320", ":")]:
        ax.plot(rank, y, color=c, label=label, ls=ls)
    ax.set(yscale="log", xlabel="Singular value index", ylabel="Weight singular value", title=NAMES[j])
    ax = axes[1, j]
    for y, c, label, ls in [
        (t / b - 1, TEAL, "Teacher / base - 1", "--"),
        (s / b - 1, ORANGE, "Student / base - 1", "-"),
    ]:
        ax.plot(rank, y, color=c, label=label, ls=ls)
    ax.set(xlabel="Singular value index", ylabel="Relative singular-value change")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    for ax in axes[:, j]:
        ax.grid(axis="y", alpha=0.2)
fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 1), ncol=3, fontsize=9)
fig.legend(*axes[1, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.51), ncol=2, fontsize=9)
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.13, top=0.88, hspace=0.8, wspace=0.4)
save(
    fig,
    "weight-spectra",
    "Exact FP64 SVD of stored k_proj, layer 13; full index range. Relative-change axes use separate scales.",
)

fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
for j, arm in enumerate(ARM[:3]):
    d = temporal[arm][1]
    series = [
        ("Teacher update", d["teacher_update_singular_values"], TEAL, "--"),
        ("Earliest student", d["stages"][0]["update_singular_values"], GRAY, ":"),
        ("Student step 320", d["stages"][-1]["update_singular_values"], ORANGE, "-"),
    ]
    for label, spectrum, color, ls in series:
        energy = np.square(spectrum)
        cumulative = np.cumsum(energy) / energy.sum()
        assert abs(cumulative[-1] - 1) < 1e-8
        axes[j].plot(np.arange(1, len(energy) + 1), cumulative, label=label, color=color, ls=ls)
    axes[j].set(
        title=NAMES[j],
        xscale="log",
        ylim=(0, 1.04),
        xlabel="Retained singular components",
        ylabel="Cumulative update energy",
    )
    axes[j].grid(axis="y", alpha=0.2)
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3)
fig.subplots_adjust(left=0.07, right=0.98, bottom=0.2, top=0.79, wspace=0.3)
save(fig, "update-spectra", "Exact SVD of k_proj updates, layer 13; each curve uses its own energy. Not task gain.")

fig, axes = plt.subplots(2, 3, figsize=(12, 7))
summary = {}
for j, arm in enumerate(ARM[:3]):
    matrices = temporal[arm]
    steps = [s["step"] for s in matrices[0]["stages"]]
    assert all([s["step"] for s in m["stages"]] == steps for m in matrices)
    fields = [
        "early_right_energy",
        "own_top_energy",
        "random_right_energy",
        "teacher_right_energy",
        "early_right_overlap",
        "teacher_right_overlap",
    ]
    values = {key: np.array([[s["subspaces"]["16"][key] for s in m["stages"]] for m in matrices]) for key in fields}
    summary[arm] = {
        "steps": steps,
        **{k: np.median(v, axis=0).tolist() for k, v in values.items()},
        "stable_rank": np.median([[s["stable_rank"] for s in m["stages"]] for m in matrices], axis=0).tolist(),
    }
    for key, label, c, ls in [
        ("early_right_energy", "Earliest student space", TEAL, "-"),
        ("own_top_energy", "Current student top-16", ORANGE, "-"),
        ("teacher_right_energy", "Teacher update top-16", fv.HUES["vcoral"], "-"),
        ("random_right_energy", "Random rank-16", GRAY, "--"),
    ]:
        v = values[key]
        axes[0, j].plot(steps, np.median(v, axis=0), color=c, ls=ls, marker="o", label=label)
        axes[0, j].fill_between(steps, v.min(0), v.max(0), color=c, alpha=0.1)
    for key, label, c in [
        ("early_right_overlap", "Earliest / current", TEAL),
        ("teacher_right_overlap", "Teacher / current", fv.HUES["vcoral"]),
    ]:
        v = values[key]
        axes[1, j].plot(steps, np.median(v, axis=0), color=c, marker="o", label=label)
    for i in [0, 1]:
        axes[i, j].set(
            xlabel="Training step",
            ylim=(-0.025, 1.05),
            ylabel=["Update energy captured", "Normalized projector overlap"][i],
        )
        axes[i, j].grid(axis="y", alpha=0.2)
    axes[0, j].set_title(NAMES[j])
axes[0, 0].legend(fontsize=7, loc="upper right")
axes[1, 0].legend(fontsize=8, loc="upper right")
fig.subplots_adjust(left=0.07, right=0.98, bottom=0.13, top=0.94, wspace=0.3, hspace=0.4)
save(
    fig,
    "subspace-trajectory",
    "k_proj layers 0/13/27: median; shading = layer range. Earliest saved step: Qwen 64, R1 16; not formation time.",
)

band_data = {}
fig, axes = plt.subplots(1, 4, figsize=(12.8, 4.5))
for j, arm in enumerate(ARM):
    data = read(ROOT / f"results/{arm}.output-s320-fisher-r128-seed0/output/scores.json")
    bands = ["0:1", "1:4", "4:16", "16:128", "complement"]
    # Per-domain energy-weighted residual; rows retain their measured domain.
    config = next(c for c in read(ROOT / "manifest.json")["configs"] if c["label"] == arm)
    records = read(Path(config["records"]))
    domains = {str(r.get("row_id", r.get("id", i))): r["domain"] for i, r in enumerate(records)}
    matrix = []
    for domain in ["math", "code", "if"]:
        rows = [r for r in data["evaluation"] if domains[str(r["row_id"])] == domain]
        matrix.append(
            [
                sum(r["bands"][b]["teacher_energy"] * r["bands"][b]["missing"] for r in rows)
                / sum(r["bands"][b]["teacher_energy"] for r in rows)
                for b in bands
            ]
        )
    band_data[arm] = dict(domains=["math", "code", "if"], bands=bands, missing=matrix)
    im = axes[j].imshow(matrix, vmin=0, vmax=1.5, cmap=fv.ramp_cmap("seq"), aspect="auto")
    axes[j].set(
        title=NAMES[j],
        xticks=range(5),
        xticklabels=["1", "2–4", "5–16", "17–128", "Rest"],
        yticks=range(3),
        yticklabels=["Math", "Code", "IF"],
    )
    axes[j].set_xlabel("Teacher residual SVD band")
    for tick in axes[j].get_xticklabels():
        tick.set_rotation(40)
        tick.set_ha("right")
fig.subplots_adjust(left=0.055, right=0.88, bottom=0.25, top=0.88, wspace=0.45)
cax = fig.add_axes([0.91, 0.25, 0.015, 0.6])
fig.colorbar(im, cax=cax, label="Residual energy / teacher energy")
save(
    fig,
    "output-bands",
    "Step 320, heldout, base-Fisher metric; rank 128 fit on calibration only. Rest: truncation/state variation.",
)

fig, ax = plt.subplots(figsize=(9, 4.8))
keys = [
    "top0.01-restore0",
    "top0.01-restore1",
    "top0.05-restore0",
    "top0.05-restore1",
    "opd-delete-diag",
    "opd-delete-off",
    "opd-delete-null",
]
labels = ["Top 1%", "Top 1%\nrescaled", "Top 5%", "Top 5%\nrescaled", "Delete\ndiag", "Delete\noff", "Delete\nnull"]
values = [report["interventions"][f"qwen-grpo.intervention-{k}"]["heldout"]["gain_retained"] for k in keys]
ax.bar(
    range(len(keys)), values, color=[TEAL, ORANGE, TEAL, ORANGE, fv.HUES["vblue"], fv.HUES["vblue"], fv.HUES["vblue"]]
)
ax.axhline(1, color=GRAY, ls="--", label="Original student")
ax.set(
    xticks=range(len(keys)),
    xticklabels=labels,
    ylabel="Fraction of original native-KL improvement",
    ylim=(0, 1.12),
    title="Qwen GRPO: spectral intervention on student updates",
)
ax.legend(loc="upper right", fontsize=9)
ax.grid(axis="y", alpha=0.2)
fig.subplots_adjust(left=0.1, right=0.98, bottom=0.22, top=0.9)
save(
    fig,
    "interventions",
    "All attention/MLP matrices; other tensors reset to base. Full-scope control: 99.74%; task accuracy unmeasured.",
)

(OUT / "band-data.json").write_text(json.dumps(band_data, indent=2))
(OUT / "temporal-summary.json").write_text(json.dumps(summary, indent=2))
(OUT / "plot-data.json").write_text(json.dumps(report, indent=2))
(OUT / "provenance.json").write_text(
    json.dumps(
        {
            "inputs_sha256": inputs,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "style_sha256": hashlib.sha256((STYLE / "fvstyle.py").read_bytes()).hexdigest(),
            "snapshot_counts": report["counts"],
        },
        indent=2,
    )
)
if Path(__file__).resolve() != (OUT / "plot_progress.py").resolve():
    shutil.copyfile(__file__, OUT / "plot_progress.py")
