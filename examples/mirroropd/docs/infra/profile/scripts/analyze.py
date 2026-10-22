# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Trace interval statistics and reproducible figures; never sum overlap as wall time."""

import argparse
import gzip
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path


def union_length(intervals):
    end = None
    total = 0.0
    for left, right in sorted(intervals):
        if right < left:
            raise ValueError("negative interval")
        total += right - left if end is None or left >= end else max(0.0, right - end)
        end = right if end is None else max(end, right)
    return total


def category(name):
    text = name.lower()
    if "nccl" in text:
        return "NCCL"
    if any(x in text for x in ("gemm", "cublas", "cutlass", "addmm")):
        return "GEMM"
    if any(x in text for x in ("flash", "attention", "fmha")):
        return "Attention"
    if any(x in text for x in ("_partials", "_rows", "_gradient")):
        return "Fused KL"
    if any(x in text for x in ("softmax", "reduce", "elementwise", "vectorized", "cast", "copy")):
        return "Pointwise/reduce"
    return "Other"


def trace_stats(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        events = json.load(f)["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel" and e.get("dur", 0) > 0]
    totals = defaultdict(float)
    spans = defaultdict(list)
    for e in kernels:
        kind = category(e["name"])
        totals[kind] += e["dur"] / 1000
        device = e.get("args", {}).get("device", e.get("pid"))
        spans[device].append((e["ts"], e["ts"] + e["dur"], kind))
    devices = {}
    for device, entries in spans.items():
        compute = [(a, b) for a, b, k in entries if k != "NCCL"]
        comm = [(a, b) for a, b, k in entries if k == "NCCL"]
        all_intervals = compute + comm
        c, n, u = union_length(compute), union_length(comm), union_length(all_intervals)
        devices[str(device)] = {
            "compute_union_ms": c / 1000,
            "nccl_union_ms": n / 1000,
            "kernel_union_ms": u / 1000,
            "overlap_ms": (c + n - u) / 1000,
            "nccl_without_compute_ms": (u - c) / 1000,
            "first_to_last_kernel_ms": (max(b for a, b in all_intervals) - min(a for a, b in all_intervals)) / 1000,
        }
    return {
        "source": str(path),
        "kernel_sum_ms_by_category": dict(totals),
        "devices": devices,
        "note": (
            "Kernel sums are GPU work, not additive wall time. "
            "NCCL without compute is not necessarily critical-path wait."
        ),
    }, kernels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--figures", type=Path, required=True)
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
    sys.path.insert(0, str(args.style_dir))
    import matplotlib

    matplotlib.use("Agg")
    import fvstyle as fv
    import matplotlib.pyplot as plt

    fv.apply_rc("paper")
    args.figures.mkdir(parents=True, exist_ok=True)
    audits = []

    def save(fig, name, caption):
        fig.text(0.5, 0.015, caption, ha="center", fontsize=8, color=fv.NEUTRALS["baseline"])
        fig.canvas.draw()
        audits.append({"figure": name, "audit": str(fv.audit(fig))})
        for ext in ("png", "pdf", "svg"):
            fig.savefig(args.figures / f"{name}.{ext}", dpi=160, bbox_inches="tight")
        plt.close(fig)

    benchmark = args.root / "kernel" / "benchmark.json"
    if benchmark.exists():
        rows = [r for r in json.loads(benchmark.read_text())["rows"] if "wall_ms_median" in r]
        names = sorted({r["backend"] for r in rows})
        colors = dict(zip(names, fv.series_colors("paper", len(names)), strict=True))
        for n in sorted({r["tokens"] for r in rows}):
            fig, ax = plt.subplots(figsize=(8, 4.5))
            for name in names:
                points = [r for r in rows if r["backend"] == name and r["tokens"] == n]
                ax.scatter(
                    [r["peak_allocated_gib"] for r in points],
                    [r["wall_ms_median"] for r in points],
                    label=name,
                    color=colors[name],
                )
            ax.set_xlabel("Peak allocated memory (GiB)")
            ax.set_ylabel("Forward + backward (ms)")
            ax.legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=False)
            fig.tight_layout(rect=(0, 0.08, 1, 1))
            save(
                fig, f"head-pareto-n{n}", f"Synthetic BF16 head only; N={n}; each point is a token-chunk configuration"
            )
            fig, ax = plt.subplots(figsize=(8, 4.5))
            for name in names:
                points = sorted(
                    (r for r in rows if r["backend"] == name and r["tokens"] == n), key=lambda r: r["chunk"]
                )
                ax.plot(
                    [r["chunk"] for r in points],
                    [r["wall_ms_median"] for r in points],
                    marker="o",
                    label=name,
                    color=colors[name],
                )
            ax.set_xlabel("Token chunk size")
            ax.set_ylabel("Forward + backward (ms)")
            ax.set_xticks(sorted({r["chunk"] for r in rows}))
            ax.legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=False)
            fig.tight_layout(rect=(0, 0.08, 1, 1))
            save(fig, f"head-chunk-n{n}", f"Synthetic BF16 head only; N={n}; median unprofiled timing")
    stats = []
    paths = sorted(args.root.rglob("*.json.gz"))
    # Limit diagrams to rank 0; retain separate raw traces on every rank.
    for path in paths:
        rank_match = re.search(r"rank[_-]?(\d+)", path.name)
        if rank_match and int(rank_match.group(1)) != 0:
            continue
        record, kernels = trace_stats(path)
        stats.append(record)
        if not kernels:
            continue
        origin = min(e["ts"] for e in kernels)
        # First 80 ms of GPU activity; this is a window, never a whole-step claim.
        selected = [e for e in kernels if e["ts"] < origin + 80000]
        lanes = sorted({str(e.get("args", {}).get("stream", e["tid"])) for e in selected})
        kinds = sorted({category(e["name"]) for e in selected})
        # Four active color families; secondary compute kinds share a family.
        base = fv.series_colors("paper", 4)
        color_map = {
            "GEMM": base[0],
            "NCCL": base[1],
            "Attention": base[2],
            "Fused KL": base[3],
            "Pointwise/reduce": base[3],
            "Other": fv.NEUTRALS["baseline"],
        }
        fig, ax = plt.subplots(figsize=(11, max(2.5, len(lanes) * 0.45)))
        for e in selected:
            lane = lanes.index(str(e.get("args", {}).get("stream", e["tid"])))
            ax.broken_barh(
                [((e["ts"] - origin) / 1000, min(e["dur"], origin + 80000 - e["ts"]) / 1000)],
                (lane - 0.35, 0.7),
                facecolors=color_map[category(e["name"])],
            )
        from matplotlib.patches import Patch

        ax.legend(
            handles=[Patch(color=color_map[k], label=k) for k in kinds],
            loc="upper left",
            bbox_to_anchor=(1, 1),
            frameon=False,
        )
        ax.set_yticks(range(len(lanes)), ["stream " + x for x in lanes])
        ax.set_xlabel("Time from first kernel (ms)")
        ax.set_ylabel("CUDA stream")
        ax.set_xlim(0, 80)
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        save(
            fig,
            path.parent.name + "-" + path.name.split(".")[0],
            "Measured trace window; category colors describe kernel names, not SM utilization",
        )
    (args.root / "trace-summary.json").write_text(json.dumps(stats, indent=2) + "\n")
    (args.root / "figure-audit.json").write_text(json.dumps(audits, indent=2) + "\n")


if __name__ == "__main__":
    main()
