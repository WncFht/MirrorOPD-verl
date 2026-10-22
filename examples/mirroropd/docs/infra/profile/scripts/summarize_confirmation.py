# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Summarize matched ABBA runs; display all steps without treating them as IID trials."""

import argparse
import json
import os
import statistics
import sys
from pathlib import Path


def summarize(root):
    runs = []
    work = None
    pooled = {}
    for path in sorted(root.glob("*/pipeline.json")):
        data = json.loads(path.read_text())
        signature = (
            data["input_tokens"],
            data["loss_tokens"],
            data["arguments"]["chunk"],
            data["arguments"]["budget"],
            data["world_size"],
        )
        if work is None:
            work = signature
        assert signature == work, "Workloads must match"
        assert len(data["records"]) == 12 and data["arguments"]["warmup_steps"] == 3
        rows = [r for r in data["records"] if not r["warmup"] and not r["profiled"]]
        assert len(rows) == 9
        backend = data["arguments"]["backend"]
        pooled.setdefault(backend, []).extend(rows)
        runs.append(
            {
                "name": path.parent.name,
                "backend": backend,
                "rows": rows,
                "median_seconds": {
                    k: statistics.median(r[k] for r in rows) for k in ("teacher_s", "actor_s", "wall_s")
                },
                "peak_allocated_gib": max(r["peak_allocated_gib"] for r in rows),
                "source": str(path),
            }
        )
    assert [r["backend"] for r in runs] == ["precompute", "fused_candidate", "fused_candidate", "precompute"]
    medians = {
        name: {k: statistics.median(r[k] for r in rows) for k in ("teacher_s", "actor_s", "wall_s")}
        for name, rows in pooled.items()
    }
    changes = {k: 1 - medians["fused_candidate"][k] / medians["precompute"][k] for k in ("actor_s", "wall_s")}
    return {
        "scope": (
            "H800 x4; fixed prefixes; lr=0; matched chunk=512/budget=16384; "
            "2 independent runs and 18 measured steps per backend. No IID confidence intervals."
        ),
        "work": work,
        "runs": runs,
        "pooled_medians_seconds": medians,
        "time_reduction_fraction": changes,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--style-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "FIGURE_VIZ_MPL",
                "/mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/fht_data/dotfiles/.agents/skills/figure-viz/assets/mpl",
            )
        ),
    )
    args = p.parse_args()
    result = summarize(args.root)
    sys.path.insert(0, str(args.style_dir))
    import matplotlib

    matplotlib.use("Agg")
    import fvstyle as fv
    import matplotlib.pyplot as plt

    fv.apply_rc("paper")
    colors = dict(zip(("precompute", "fused_candidate"), fv.series_colors("paper", 2), strict=True))
    audits = {}
    for metric, label in [("actor_s", "Actor update incl. offload (s)"), ("wall_s", "Teacher + actor pipeline (s)")]:
        fig, ax = plt.subplots(figsize=(7, 4.4))
        seen = set()
        for i, run in enumerate(result["runs"]):
            backend = run["backend"]
            values = [r[metric] for r in run["rows"]]
            x = [i + (j - 4) * 0.04 for j in range(9)]
            ax.scatter(x, values, color=colors[backend], label=backend if backend not in seen else "_nolegend_")
            ax.hlines(statistics.median(values), i - 0.28, i + 0.28, color=colors[backend], linewidth=2)
            seen.add(backend)
        ax.set_xticks(range(4), ["A1: baseline", "B1: fused", "B2: fused", "A2: baseline"])
        ax.set_xlabel("Execution order")
        ax.set_ylabel(label)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
        fig.text(
            0.5,
            0.015,
            "H800 x4; 3 warmup + 9 measured steps/run; horizontal lines: medians",
            ha="center",
            fontsize=9,
            color=fv.NEUTRALS["baseline"],
        )
        fig.tight_layout(rect=(0, 0.09, 1, 1))
        fig.canvas.draw()
        audits[metric] = str(fv.audit(fig))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(
                args.output.with_name(args.output.name + "-" + metric).with_suffix("." + ext),
                dpi=160,
                bbox_inches="tight",
            )
        plt.close(fig)
    result["figure_audit"] = audits
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(result["pooled_medians_seconds"], result["time_reduction_fraction"], audits)


if __name__ == "__main__":
    main()
