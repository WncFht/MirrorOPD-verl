# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Static, audited figures from the reviewed baseline and real Torch traces."""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--style-dir", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.data.read_text())
    review = json.loads(args.review.read_text())
    sys.path.insert(0, str(args.style_dir))
    import matplotlib

    matplotlib.use("Agg")
    import fvstyle as fv
    import matplotlib.pyplot as plt

    fv.apply_rc("paper")
    colors = fv.series_colors("paper", 4)
    args.output.mkdir(parents=True, exist_ok=True)
    audits = []

    def save(fig, name, caption):
        fig.text(0.5, 0.015, caption, ha="center", fontsize=9, color=fv.NEUTRALS["baseline"])
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        fig.canvas.draw()
        issues = fv.audit(fig)
        audits.append(dict(figure=name, issues=issues))
        for ext in ("png", "pdf", "svg"):
            fig.savefig(args.output / f"{name}.{ext}", dpi=180, bbox_inches="tight")
        plt.close(fig)

    rows = review["rollout_boundaries"]
    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = [0.0] * len(rows)
    for i, (key, label) in enumerate(
        [
            ("before_request_s", "Before first request"),
            ("request_envelope_s", "vLLM request envelope"),
            ("after_requests_to_samples_s", "After requests: scoring / publish"),
            ("after_samples_s", "After samples: collection / sleep"),
        ]
    ):
        y = [r[key] for r in rows]
        ax.bar([r["step"] for r in rows], y, bottom=bottom, label=label, color=colors[i])
        bottom = [a + b for a, b in zip(bottom, y, strict=True)]
    ax.set_xlabel("Training step")
    ax.set_ylabel("Rollout pipeline wall time (s)")
    ax.set_xticks([r["step"] for r in rows])
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
    save(fig, "rollout-boundaries", "Measured boundaries; last two regions are not pure computation or polling costs")

    samples = data["jobs"]["baseline"]["resources"]
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.8), sharex=True)
    for i in range(4):
        axes[0].plot(
            [r["seconds"] for r in samples],
            [r["gpus"][i]["memory_gib"] for r in samples],
            color=colors[i],
            label=f"GPU {i}",
        )
        axes[1].plot(
            [r["seconds"] for r in samples],
            [r["gpus"][i]["busy_percent"] for r in samples],
            color=colors[i],
            label=f"GPU {i}",
        )
    axes[0].set_ylabel("NVML used memory (GiB)")
    axes[0].set_xlabel("Time since launch (s)")
    axes[1].set_ylabel("NVML GPU busy (%)")
    axes[1].set_xlabel("Time since launch (s)")
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False)
    save(fig, "four-gpu-lifecycle", "Two-second sampling; includes cold start and validation; GPU busy is not MFU")

    traces = [r for r in review["traces"] if r["calibration"]]
    rows = [next(p for p in r["phases"] if p["name"] == "timer/update_actor" and p["covered"]) for r in traces]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bottom = [0.0] * len(rows)
    for i, (label, values) in enumerate(
        [
            ("Compute (includes NCCL overlap)", [r["compute_s"] for r in rows]),
            ("NCCL without compute", [r["nccl_without_compute_s"] for r in rows]),
            ("Copy/memset only", [r["gpu_active_s"] - r["compute_s"] - r["nccl_without_compute_s"] for r in rows]),
            ("No recorded GPU activity", [r["no_gpu_event_s"] for r in rows]),
        ]
    ):
        ax.barh(range(len(rows)), values, left=bottom, color=colors[i], label=label)
        bottom = [a + b for a, b in zip(bottom, values, strict=True)]
    ax.set_yticks(range(len(rows)), [f"Rank {i}" for i in range(len(rows))])
    ax.invert_yaxis()
    ax.set_xlabel("Actor update wall time (s)")
    ax.set_ylabel("Training rank")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
    save(
        fig, "actor-gpu-union", "Profiled step 4; interval unions, not kernel sums; gaps are not proven removable waste"
    )
    (args.output / "review-audit.json").write_text(json.dumps(audits, indent=2) + "\n")
    print(json.dumps(audits, indent=2))


if __name__ == "__main__":
    main()
