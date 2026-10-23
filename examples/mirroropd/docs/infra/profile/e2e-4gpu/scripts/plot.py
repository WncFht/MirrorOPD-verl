# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Export measured baseline phase costs. Empty data produces no placeholder figure."""

import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--style-dir", type=Path, default=os.environ.get("FIGURE_VIZ_MPL"))
    args = parser.parse_args()
    data = json.loads(args.data.read_text())
    rows = data["jobs"].get("baseline", {}).get("steps", [])
    if not rows:
        print("No completed measured baseline steps; no figure generated.")
        return
    if any(r["phase_overlap_seconds"] > 1e-6 for r in rows):
        raise ValueError("Overlapping phases cannot be rendered as exclusive stacked costs")
    if args.style_dir:
        sys.path.insert(0, str(args.style_dir))
    import matplotlib

    matplotlib.use("Agg")
    import fvstyle as fv
    import matplotlib.pyplot as plt

    fv.apply_rc("paper")
    colors = fv.series_colors("paper", 4)
    # Four broad, disjoint groups keep the static overview legible; HTML has all phases.
    values = []
    for r in rows:
        p = r["phases"]
        groups = [p["gen"], p["teacher_transfer"] + p["teacher_forward"], p["update_actor"]]
        values.append(groups + [r["seconds"] - sum(groups)])
    fig, ax = plt.subplots(figsize=(8.5, max(4, 0.32 * len(rows) + 2)))
    bottom = [0.0] * len(rows)
    for i, label in enumerate(("Rollout pipeline", "Teacher + transfer", "Actor update", "Other step work")):
        vv = [r[i] for r in values]
        ax.barh([r["step"] for r in rows], vv, left=bottom, color=colors[i], label=label)
        bottom = [a + b for a, b in zip(bottom, vv, strict=True)]
    ax.set_yticks([r["step"] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Measured in-step wall time (s)")
    ax.set_ylabel("Training step")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
    fig.text(
        0.5,
        0.025,
        (
            "SYNTHETIC TEST ONLY: rendering check, not performance evidence"
            if data.get("synthetic_test")
            else "Graph rollout; work counts vary; validation and post-step cleanup excluded"
        ),
        ha="center",
        fontsize=9,
        color=fv.NEUTRALS["baseline"],
    )
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.canvas.draw()
    issues = fv.audit(fig)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(
        json.dumps({"data": str(args.data), "steps": rows, "audit": issues}, indent=2) + "\n"
    )
    for extension in ("png", "pdf", "svg"):
        fig.savefig(args.output.with_suffix("." + extension), dpi=180, bbox_inches="tight")
    svg = args.output.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)
    print(json.dumps({"audit": issues, "requires_visual_review": True}))


if __name__ == "__main__":
    main()
