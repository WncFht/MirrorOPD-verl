"""All-layer shared-ancestor geometry and independent output diagnostics.

Heatmap cells are exact projected energy fractions, one teacher/module cell.
Cosine heatmap uses six-matrix medians in identical parameter coordinates.
Entropy bars compare actual-token-normalized output entropies on fixed states;
this denominator is distinct from the primary H1024 RKL curve denominator.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "curve-spectrum-v1"
COMMON = ROOT / "common-ancestor-v1"
sys.path.insert(0, str(ROOT.parents[2] / "dotfiles/.agents/skills/figure-viz/assets/mpl"))
import fvstyle as fv  # noqa: E402


def save(fig, name, ignore=None):
    name += "-v2"
    fig.canvas.draw()
    audit = [str(x) for x in fv.audit(fig, print_slot=12, scoped_ignore=ignore or [])]
    (OUT / (name + ".audit.json")).write_text(json.dumps(audit))
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT / (name + "." + ext), dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(name, audit, flush=True)


def main():
    assert (COMMON / "complete.json").exists()
    records = []
    summary = []
    layers = []
    for file in COMMON.glob("*.json"):
        if file.name == "complete.json":
            continue
        obj = json.loads(file.read_text())
        data = obj["matrices"]
        teacher = obj["teacher"]
        for r in data:
            records.append(dict(teacher=teacher, **r))
        scopes = [("all", data)] + [
            (module, [r for r in data if r["module"] == module]) for module in sorted({r["module"] for r in data})
        ]
        for name, rows in scopes:
            ebase = sum(r["base_energy"] for r in rows)
            ed = sum(r["teacher_delta_energy"] for r in rows)
            eg = sum(r["gap_energy"] for r in rows)
            eo = sum(r["teacher_ancestor_energy"] for r in rows)
            dot = sum(r["dot_gap_delta"] for r in rows)
            summary.append(
                dict(
                    teacher=teacher,
                    module=name,
                    matrices=len(rows),
                    update_norm_over_initial=np.sqrt(ed / ebase),
                    own_norm_over_initial=np.sqrt(eo / ebase),
                    own_norm_over_teacher_update=np.sqrt(eo / ed),
                    shared_axis_energy_fraction=dot * dot / (eg * ed),
                    shared_gap_gain=dot / eg,
                )
            )
        for layer in range(28):
            rows = [r for r in data if r["layer"] == layer]
            ed = sum(r["teacher_delta_energy"] for r in rows)
            eg = sum(r["gap_energy"] for r in rows)
            dot = sum(r["dot_gap_delta"] for r in rows)
            layers.append(dict(teacher=teacher, layer=layer, shared_axis_energy_fraction=dot * dot / (eg * ed)))
    pd.DataFrame(records).to_csv(OUT / "common-ancestor-matrices.csv", index=False)
    sd = pd.DataFrame(summary)
    sd.to_csv(OUT / "common-ancestor-summary.csv", index=False)
    pd.DataFrame(layers).to_csv(OUT / "common-ancestor-layers.csv", index=False)
    fv.apply_rc("paper", fontsize=10)
    teachers = [
        "q3-base",
        "base-opd",
        "coreward-i",
        "coreward-ii",
        "base-gt-math",
        "sft-deepmath",
        "sft-gsm8k",
        "grpo-kl",
        "e3",
    ]
    modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    neighbors = json.loads((OUT / "teacher-neighbors.json").read_text())["comparisons"]
    cos = np.eye(len(teachers))
    for i, a in enumerate(teachers):
        for j, b in enumerate(teachers):
            if i == j:
                continue
            r = next(r for r in neighbors if {r["teacher_a"], r["teacher_b"]} == {a, b})
            cos[i, j] = r["delta_cosine"]
    arr = np.array(
        [
            [sd[(sd.teacher == t) & (sd.module == m)].iloc[0].shared_axis_energy_fraction for m in modules]
            for t in teachers
        ]
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    fig.subplots_adjust(left=0.13, right=0.91, bottom=0.23, top=0.9, wspace=0.52)
    im = axes[0].imshow(cos, vmin=-1, vmax=1, cmap="BrBG")
    axes[0].set_xticks(range(9), teachers, rotation=55, ha="right")
    axes[0].set_yticks(range(9), teachers)
    axes[0].set(xlabel="Teacher update", ylabel="Teacher update", title="Six-matrix signed delta cosine")
    im2 = axes[1].imshow(arr, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    axes[1].set_xticks(range(7), modules, rotation=55, ha="right")
    axes[1].set_yticks(range(9), teachers)
    axes[1].set(xlabel="Module (all 28 layers)", ylabel="Teacher", title="Energy projected onto shared Base gap")
    for i in range(9):
        for j in range(7):
            if arr[i, j] > 0.65:
                continue
            axes[1].text(
                j,
                i,
                f"{arr[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if arr[i, j] > 0.65 else fv.NEUTRALS["baseline"],
            )
    fig.colorbar(im, cax=fig.add_axes([0.445, 0.29, 0.012, 0.49]))
    fig.colorbar(im2, cax=fig.add_axes([0.94, 0.29, 0.012, 0.49]))
    fig.text(
        0.5,
        0.025,
        "Shared gap = Qwen3-Base minus initial Qwen3; all-layer panel: 196 attention/MLP matrices per teacher",
        ha="center",
        fontsize=8,
    )
    save(fig, "common-ancestor-geometry", [("missing-axis-label", ax) for ax in fig.axes[2:]])
    a = pd.read_csv(OUT / "band-analysis-v3/output-diagnostics.csv")
    a = a[(a.student == "q3-1.7") & (a.domain == "all") & (a.window == "all")]
    names = ["grpo-kl", "e3", "sft-gsm8k", "sft-deepmath", "q3-base", "base-opd"]
    colors = fv.series_colors("paper", 2)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.25, top=0.9, wspace=0.32)
    for j, step in enumerate([0, 320]):
        p = a[a.step == step].pivot(index="teacher", columns="metric", values="value").loc[names]
        axes[0].bar(np.arange(6) + (j - 0.5) * 0.36, p.hp - p.hq, width=0.34, color=colors[j], label=f"Step{step}")
        axes[1].bar(np.arange(6) + (j - 0.5) * 0.36, p.js, width=0.34, color=colors[j], label=f"Step{step}")
    for ax in axes:
        ax.set_xticks(range(6), names, rotation=45, ha="right")
        ax.set_xlabel("Teacher")
        ax.legend(frameon=False)
    axes[0].set_ylabel("Student entropy − teacher entropy (nats/token)")
    axes[1].set_ylabel("Jensen–Shannon divergence (nats/token)")
    fig.text(
        0.5,
        0.025,
        "Qwen thinking, fixed held-out states; per-question actual-token means; hp=student entropy, hq=teacher entropy",
        ha="center",
        fontsize=8,
    )
    save(fig, "output-entropy-and-js")
    print(sd[sd.module == "all"].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
