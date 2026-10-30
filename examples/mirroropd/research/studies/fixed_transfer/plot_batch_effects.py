# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Ledger: columns are trained batches; rows are scored batches/panels.

Each cell is RKL_before minus RKL_after across one 16-update block.
Outlined diagonal cells are the optimized batch itself; held-out rows never
participate in updates. No unobserved values or updates beyond the view limit.
"""

import argparse
from pathlib import Path

import numpy as np

from examples.mirroropd.research.core.artifacts import file_hash
from examples.mirroropd.research.core.io import dump, read, write_csv
from examples.mirroropd.research.core.plotting import load_style
from examples.mirroropd.research.studies.fixed_transfer.display_policy import MAX_DISPLAY_UPDATES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--updates", type=int, default=320)
    args = parser.parse_args()
    if not 0 < args.updates <= MAX_DISPLAY_UPDATES or args.updates % 16:
        raise ValueError("Use complete blocks within the current presentation range")
    snapshots, sources = {}, []
    identity = None
    for step in range(0, args.updates + 1, 16):
        path = args.root / f"fixed-{step:03d}.json"
        rows = read(path)
        current = [(r["id"], r["source"], r["group"], r["length"]) for r in rows]
        identity = identity or current
        assert current == identity and all(r["step"] == step for r in rows)
        snapshots[step] = rows
        sources.append(dict(step=step, path=str(path.resolve()), sha256=file_hash(path)))
    available = {r["group"] for r in snapshots[0]}
    if not set("ABCDEFGHIJKLMNOPQRST"[: args.updates // 16]) <= available:
        raise ValueError("Snapshots omit batches that were trained")
    groups = [(g, [g]) for g in "ABCDEFGHIJKLMNOPQRST" if g in available]
    panels = groups + [
        ("Math (64)", ["M01", "M02", "M03", "M04"]),
        ("Code (32)", ["Code01", "Code02"]),
        ("IF (32)", ["IF01", "IF02"]),
        ("All held-out (128)", ["M01", "M02", "M03", "M04", "Code01", "Code02", "IF01", "IF02"]),
    ]
    matrix = np.empty((len(panels), args.updates // 16))
    measured = []
    for col, stop in enumerate(range(16, args.updates + 1, 16)):
        trained = chr(65 + col)
        for row, (label, selected) in enumerate(panels):
            values = []
            for step in [stop - 16, stop]:
                chosen = [r for r in snapshots[step] if r["group"] in selected]
                assert len(chosen) == len(selected) * 16
                values.append(sum(r["stats"]["all"]["rkl"] for r in chosen) / (len(chosen) * 1024))
            gain = values[0] - values[1]
            matrix[row, col] = gain
            relation = (
                "held_out"
                if row >= len(groups)
                else "current"
                if label == trained
                else "seen"
                if label < trained
                else "not_yet_seen"
            )
            measured.append(
                dict(
                    trained_batch=trained,
                    evaluated_panel=label,
                    before_step=stop - 16,
                    after_step=stop,
                    relation=relation,
                    before=values[0],
                    after=values[1],
                    rkl_decrease=gain,
                )
            )
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "batch-effects.csv", measured)
    dump(args.output / "sources.json", sources)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from matplotlib.patches import Rectangle

    fv = load_style()
    fv.apply_rc("paper", fontsize=10)
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.subplots_adjust(left=0.20, right=0.89, bottom=0.18, top=0.87)
    bound = max(float(np.abs(matrix).max()), 1e-12)
    colors = LinearSegmentedColormap.from_list(
        "risk_change", [fv.HUES["vorange"], fv.NEUTRALS["paper"], fv.HUES["vblue"]]
    )
    im = ax.imshow(matrix, aspect="auto", cmap=colors, norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound))
    for i in range(args.updates // 16):
        ax.add_patch(
            Rectangle(
                (i - 0.5, i - 0.5),
                1,
                1,
                fill=False,
                facecolor=fv.NEUTRALS["paper"],
                edgecolor=fv.NEUTRALS["vink"],
                linewidth=0.65,
            )
        )
    ax.axhline(len(groups) - 0.5, color=fv.NEUTRALS["vink"], linewidth=1)
    ax.set_xticks(range(args.updates // 16), list("ABCDEFGHIJKLMNOPQRST")[: args.updates // 16])
    ax.set_yticks(range(len(panels)), [label for label, _ in panels])
    ax.set(
        xlabel="Batch optimized for the next 16 updates (A to T)", ylabel="Fixed batch / held-out panel being evaluated"
    )
    bar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    bar.set_label("RKL decrease: before - after (blue improves; orange worsens)")
    fig.suptitle(args.label + f"\nEffect of each training block, 0-{args.updates} updates", y=0.96)
    fig.text(
        0.5,
        0.055,
        "Outlined diagonal: current training batch. Earlier rows: already-seen batches; "
        "later training rows: not-yet-seen batches.\n"
        "Bottom four rows: fixed held-out domains and their pooled summary. "
        "H1024; frozen states; one training seed.\n"
        "Cells are descriptive absolute risk changes, not significance tests. "
        "All held-out overlaps the three domain rows.",
        ha="center",
        fontsize=8.5,
        color=fv.NEUTRALS["baseline"],
    )
    fig.canvas.draw()
    issues = fv.audit(fig, print_slot=fig.get_figwidth())
    dump(args.output / "audit.json", [str(x) for x in issues])
    if any(fv.problem_severity(x) == "error" for x in issues):
        raise RuntimeError(str(issues))
    for ext in ["png", "pdf"]:
        fig.savefig(args.output / f"batch-effects.{ext}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
