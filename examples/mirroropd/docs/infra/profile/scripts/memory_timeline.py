# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Replay a trusted local allocator snapshot into requested-live and reserved bytes.

Requested bytes exclude allocator rounding and external/NCCL allocations. The
plot is a peak envelope per 5 ms bin, not a sampled exact per-operation timeline.
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path


def reconstruct(snapshot, device=0):
    events = snapshot["device_traces"][device]
    segments = [s for s in snapshot["segments"] if s["device"] == device]
    live_end = sum(
        b.get("requested_size", b["size"]) for s in segments for b in s["blocks"] if b["state"] == "active_allocated"
    )
    reserved_end = sum(s["total_size"] for s in segments)
    signs = {"alloc": 1, "free_requested": -1, "segment_alloc": 1, "segment_free": -1}
    net_live = sum(e["size"] * signs[e["action"]] for e in events if e["action"] in ("alloc", "free_requested"))
    net_reserved = sum(
        e["size"] * signs[e["action"]] for e in events if e["action"] in ("segment_alloc", "segment_free")
    )
    live, reserved = live_end - net_live, reserved_end - net_reserved
    begin = events[0]["time_us"]
    bins = {}
    for e in events:
        action = e["action"]
        if action in ("alloc", "free_requested"):
            live += e["size"] * signs[action]
        elif action in ("segment_alloc", "segment_free"):
            reserved += e["size"] * signs[action]
        if min(live, reserved) < 0:
            raise ValueError("Snapshot history is incomplete or requested-byte reconstruction failed")
        bucket = int((e["time_us"] - begin) // 5000)
        row = bins.setdefault(bucket, [0, 0])
        row[0], row[1] = max(row[0], live), max(row[1], reserved)
    assert (live, reserved) == (live_end, reserved_end)
    return [
        {"seconds": b * 0.005, "requested_live_gib": v[0] / 2**30, "reserved_gib": v[1] / 2**30}
        for b, v in sorted(bins.items())
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", type=Path, required=True)
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
    with args.snapshot.open("rb") as f:
        snapshot = pickle.load(f)
    rows = reconstruct(snapshot)
    sys.path.insert(0, str(args.style_dir))
    import matplotlib

    matplotlib.use("Agg")
    import fvstyle as fv
    import matplotlib.pyplot as plt

    fv.apply_rc("paper")
    colors = fv.series_colors("paper", 2)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        [r["seconds"] for r in rows],
        [r["requested_live_gib"] for r in rows],
        label="Requested live bytes",
        color=colors[0],
    )
    ax.plot([r["seconds"] for r in rows], [r["reserved_gib"] for r in rows], label="Reserved segments", color=colors[1])
    ax.set_xlabel("Time from first recorded allocation (s)")
    ax.set_ylabel("Memory (GiB)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
    fig.text(
        0.5,
        0.015,
        "Rank 0 allocator; 5 ms peak bins; excludes external allocations",
        ha="center",
        fontsize=9,
        color=fv.NEUTRALS["baseline"],
    )
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.canvas.draw()
    audit = fv.audit(fig)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(args.output.with_suffix("." + ext), dpi=160, bbox_inches="tight")
    result = {
        "source": str(args.snapshot),
        "scope": __doc__,
        "rows": rows,
        "audit": str(audit),
        "peak_requested_gib": max(r["requested_live_gib"] for r in rows),
        "peak_reserved_gib": max(r["reserved_gib"] for r in rows),
    }
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print({k: v for k, v in result.items() if k != "rows"})


if __name__ == "__main__":
    main()
