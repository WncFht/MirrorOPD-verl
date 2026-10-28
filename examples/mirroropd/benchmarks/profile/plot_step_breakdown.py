# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Plot reported non-overlapping outer-stage timers, including explicit residual.

Rows have different generated token counts; this plot describes cost distribution,
not the causal throughput benefit of asynchronous IO.
"""

import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--style-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "FIGURE_VIZ_MPL",
                "/mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/fht_data/dotfiles/.agents/skills/figure-viz/assets/mpl",
            )
        ),
    )
    args = parser.parse_args()
    data = json.loads(args.data.read_text())
    rows = []
    for name in ("e2e-io0", "e2e-io1"):
        for row in data[name]["metrics"]:
            step = int(row["step/outer"])
            if step not in (2, 4):
                continue
            key = "perf/timing/"
            generation = row[key + "rollout_pipeline_seconds"]
            teacher = row[key + "teacher_transfer_seconds"] + row[key + "teacher_forward_seconds"]
            actor = row[key + "update_actor_seconds"]
            other = row[key + "outer_step_seconds"] - generation - teacher - actor
            assert other >= 0
            rows.append(
                {
                    "label": f"IO {name[-1]} / step {step}",
                    "tokens": row["rollout/response_tokens"],
                    "seconds": [generation, teacher, actor, other],
                }
            )
    sys.path.insert(0, str(args.style_dir))
    import matplotlib

    matplotlib.use("Agg")
    import fvstyle as fv
    import matplotlib.pyplot as plt

    fv.apply_rc("paper")
    colors = fv.series_colors("paper", 4)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    left = [0.0] * len(rows)
    for i, label in enumerate(("Rollout pipeline", "Teacher + transfer", "Actor update", "Weight sync + other")):
        values = [r["seconds"][i] for r in rows]
        ax.barh(range(len(rows)), values, left=left, color=colors[i], label=label)
        left = [a + b for a, b in zip(left, values, strict=True)]
    ax.set_yticks(range(len(rows)), [f"{r['label']}\n{r['tokens'] / 1000:.1f}k response tokens" for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Outer step wall time (s)")
    ax.set_ylabel("Measured run / step")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
    fig.text(
        0.5,
        0.015,
        "Eager rollout; work counts differ; unprofiled steps; descriptive stage costs",
        ha="center",
        fontsize=9,
        color=fv.NEUTRALS["baseline"],
    )
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.canvas.draw()
    audit = str(fv.audit(fig))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(args.output.with_suffix("." + ext), dpi=160, bbox_inches="tight")
    args.output.with_suffix(".json").write_text(
        json.dumps({"source": str(args.data), "rows": rows, "audit": audit}, indent=2) + "\n"
    )
    print(audit)


if __name__ == "__main__":
    main()
