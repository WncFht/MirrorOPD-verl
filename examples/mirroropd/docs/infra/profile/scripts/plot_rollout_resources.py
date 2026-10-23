# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Show sampled GPU activity and residency over three training-only steps.

Stage intervals are inferred from application timers; points are actual NVML
samples. Connecting lines guide the eye and do not reconstruct transitions.
"""

import argparse
import gzip
import json
import os
import statistics
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--first-step", type=int, default=51)
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
    analysis = json.loads((args.data / "phase-analysis.json").read_text())
    windows = [w for w in analysis["windows"] if args.first_step <= w["step"] <= args.first_step + 2]
    assert windows and all(w["phase"] != "validation" for w in windows)
    start = min(w["start"] for w in windows)
    end = max(w["end"] for w in windows) + 4
    with gzip.open(args.data / "system-sampled.jsonl.gz", "rt") as f:
        samples = [json.loads(x) for x in f]
    samples = [r for r in samples if start <= r["_timestamp"] <= end and r.get("system.gpu.0.gpu") is not None]
    rows = [
        {
            "seconds": r["_timestamp"] - start,
            "gpu_mean": statistics.mean(r[f"system.gpu.{i}.gpu"] for i in range(8)),
            "gpu_min": min(r[f"system.gpu.{i}.gpu"] for i in range(8)),
            "memory_gib_mean": statistics.mean(r[f"system.gpu.{i}.memoryAllocatedBytes"] / 2**30 for i in range(8)),
        }
        for r in samples
    ]
    sys.path.insert(0, str(args.style_dir))
    import matplotlib

    matplotlib.use("Agg")
    import fvstyle as fv
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fv.apply_rc("paper")
    colors = dict(zip(("train_rollout", "actor_update"), fv.series_colors("paper", 2), strict=True))
    fig, axes = plt.subplots(2, 1, figsize=(7, 5.6), sharex=True)
    for ax in axes:
        for w in windows:
            ax.axvspan(w["start"] - start, w["end"] - start, color=colors[w["phase"]], alpha=0.15)
        ax.set_xlim(0, end - start)
    axes[0].plot([r["seconds"] for r in rows], [r["gpu_mean"] for r in rows], marker="o", color=fv.NEUTRALS["vink"])
    axes[0].set_ylabel("GPU busy: 8-card mean (%)")
    axes[0].set_ylim(0, 105)
    axes[1].plot(
        [r["seconds"] for r in rows], [r["memory_gib_mean"] for r in rows], marker="o", color=fv.NEUTRALS["vink"]
    )
    axes[1].set_ylabel("Device memory: mean (GiB)")
    axes[1].set_ylim(0, 80)
    axes[1].set_xlabel(f"Seconds from training step {args.first_step} rollout start")
    fig.legend(
        handles=[
            Patch(facecolor=colors[k], alpha=0.3, label=label)
            for k, label in [("train_rollout", "Training rollout"), ("actor_update", "Actor update incl. transfers")]
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1),
        ncol=2,
        frameon=False,
    )
    fig.text(
        0.5,
        0.015,
        "A100 x8; ~15 s samples; inferred stages; lines only guide the eye",
        ha="center",
        fontsize=9,
        color=fv.NEUTRALS["baseline"],
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.canvas.draw()
    audit = str(fv.audit(fig))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(args.output.with_suffix("." + ext), dpi=160, bbox_inches="tight")
    args.output.with_suffix(".json").write_text(
        json.dumps({"rows": rows, "windows": windows, "audit": audit}, indent=2) + "\n"
    )
    print(audit)


if __name__ == "__main__":
    main()
