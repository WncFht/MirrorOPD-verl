# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Ledger: two input banks; each curve is one fixed prompt panel, not another teacher.

All 21 measured snapshots are shown. Absolute RKL exposes the small initial
scale while A/T curves distinguish current-batch fitting from old-batch retention.
No math-only teacher-copy reference is plotted as a pooled-risk lower bound.
"""

import argparse
from pathlib import Path

from examples.mirroropd.research.core.artifacts import file_hash
from examples.mirroropd.research.core.io import dump, read, write_csv
from examples.mirroropd.research.core.paths import workspace
from examples.mirroropd.research.core.plotting import load_style

LINES = {
    "thinking": ("qwen3-thinking-rlvr-control320-20261030", "T-C-RLVR"),
    "nonthinking": ("qwen3-nonthinking-origin-controls320-20261030", "NT-C-rlvr-gsm8k"),
}
PANELS = {
    "train_all": ("Training pool A-T (320)", list("ABCDEFGHIJKLMNOPQRST"), "vblue"),
    "A": ("First batch A (16)", ["A"], "vorange"),
    "T": ("Last batch T (16)", ["T"], "vviolet"),
    "held_all": ("All held-out (128)", ["M01", "M02", "M03", "M04", "Code01", "Code02", "IF01", "IF02"], "vgray"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    records, sources = [], []
    for mode, (name, arm) in LINES.items():
        root = workspace() / "wandb-logs/MirrorOPD-verl/analysis" / name / "arms" / arm
        assert read(root / "complete.json")["updates"] == 320
        identity = None
        for step in range(0, 321, 16):
            path = root / f"fixed-{step:03d}.json"
            rows = read(path)
            current = [(r["id"], r["source"], r["group"], r["length"]) for r in rows]
            identity = identity or current
            assert current == identity and len({r["id"] for r in rows}) == len(rows)
            assert all(r["step"] == step for r in rows)
            for panel, (_, groups, _) in PANELS.items():
                chosen = [r for r in rows if r["group"] in groups]
                assert len(chosen) == 16 * len(groups)
                records.append(
                    dict(
                        mode=mode,
                        step=step,
                        panel=panel,
                        prompts=len(chosen),
                        rkl=sum(r["stats"]["all"]["rkl"] for r in chosen) / (len(chosen) * 1024),
                    )
                )
            sources.append(dict(mode=mode, step=step, path=str(path), sha256=file_hash(path)))
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "curves.csv", records)
    dump(output / "sources.json", sources)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fv = load_style()
    fv.apply_rc("paper", fontsize=10)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.subplots_adjust(left=0.085, right=0.97, bottom=0.26, top=0.78, wspace=0.26)
    for ax, mode in zip(axes, LINES, strict=True):
        reductions = {}
        for panel, (label, _, color) in PANELS.items():
            selected = [r for r in records if r["mode"] == mode and r["panel"] == panel]
            ax.plot(
                [r["step"] for r in selected],
                [r["rkl"] for r in selected],
                label=label,
                color=fv.HUES[color],
                linestyle="--" if panel == "held_all" else "-",
                linewidth=1.7,
            )
            reductions[panel] = 100 * (1 - selected[-1]["rkl"] / selected[0]["rkl"])
        mode_label = "Thinking" if mode == "thinking" else "Non-thinking"
        ax.set(
            title=f"{mode_label} input\nTraining reduction {reductions['train_all']:+.1f}%; "
            f"held-out {reductions['held_all']:+.1f}%",
            xlabel="Optimizer updates",
            ylabel="Absolute RKL (fixed-H normalization)",
            xlim=(0, 320),
        )
        ax.set_ylim(bottom=0)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Teacher: tocsa/qwen3-1.7b-rlvr-gsm8k | Student: Qwen3-1.7B\n"
        "Same-size, near-identical initial weights; sequential A-T updates",
        y=0.96,
    )
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.12), ncol=4, fontsize=9)
    fig.text(
        0.5,
        0.025,
        "Frozen S0; H1024; batch16/reuse16; 320 updates; one training seed.\n"
        "A and T are subsets of the training pool. Modes use different state banks; compare within each panel.",
        ha="center",
        fontsize=9,
        color=fv.NEUTRALS["baseline"],
    )
    fig.canvas.draw()
    problems = fv.audit(fig, print_slot=fig.get_figwidth())
    dump(output / "audit.json", [str(p) for p in problems])
    if any(fv.problem_severity(p) == "error" for p in problems):
        raise RuntimeError(f"Figure audit failed: {problems}")
    for extension in ["png", "pdf"]:
        fig.savefig(output / f"rlvr-training-retention.{extension}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
