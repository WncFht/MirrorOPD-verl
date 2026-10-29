# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Read completed local W&B runs, audit logger parity, and render diagnostics.

Semantic ledger: E41/E43 identify distinct model pairs, not matched loss arms.
Color identifies the run, line style the within-batch observation or window.
Position heatmaps encode raw KL with one shared log color scale; bins are relative
to each response. Time budgets use disjoint recorded training/validation totals.
GPU phase attribution uses approximate logger-derived windows with 30s margins.
Geometry: equal-size panels; outer/update/elapsed-time axes are never interchanged.
Only actual observations are connected; missing values are not replaced with zero.
"""

import argparse
import json
import statistics as st
from datetime import datetime
from pathlib import Path

import numpy as np

from examples.mirroropd.research.core.io import dump, read_rows, write_csv
from examples.mirroropd.research.core.plotting import load_style

RUNS = {"E41": "53dd553d", "E43": "eccc09a1"}
BENCHES = ["math500", "amc23", "aime24", "aime25", "aime26"]
TITLES = ["MATH-500", "AMC23", "AIME24", "AIME25", "AIME26"]
BINS = [f"p{i:02d}_{i + 10:02d}" for i in range(0, 100, 10)]


def audit_parity(history, file_rows):
    if len(history) != len(file_rows):
        raise ValueError("W&B and file logger row counts differ")
    checked = 0
    for wb, file_row in zip(history, file_rows, strict=True):
        for key, value in file_row["data"].items():
            if key not in wb or wb[key] != value:
                raise ValueError(f"Logger mismatch: row {checked}, key {key}")
        checked += 1
    return checked


def interval_counter_delta(samples, start, end):
    """Conservative in-window delta, per engine; missing/reset counters fail closed."""
    counters = {}
    for row in samples:
        if not start <= row["time"] <= end:
            continue
        for engine in row["engines"]:
            value = engine.get("metrics", {}).get("vllm:num_preemptions_total")
            if value is not None:
                counters.setdefault(engine["address"], []).append(value)
    if len(counters) != 8 or any(len(v) < 2 for v in counters.values()):
        return None
    if any(any(b < a for a, b in zip(v, v[1:], strict=False)) for v in counters.values()):
        raise ValueError("vLLM counter reset inside a validation interval")
    return sum(v[-1] - v[0] for v in counters.values())


def summarize_run(tag, decoded, data):
    meta = json.loads((decoded / "run.json").read_text())
    if meta["status"] != "finished" or meta["exit_code"] != 0:
        raise ValueError(f"{tag} is not a completed successful run")
    rows = read_rows(decoded / "history.jsonl")
    parity = audit_parity(rows, read_rows(data / tag / "metrics.jsonl"))
    updates = [r for r in rows if "distill/full_kl" in r]
    train = [r for r in rows if "rollout/response_length/mean" in r]
    valid = [r for r in rows if "eval/math500/score/mean" in r]
    assert [r["step/optimizer"] for r in updates] == list(range(1, 1601))
    assert [r["step/outer"] for r in train] == list(range(1, 101))
    assert all(r["optim/actor/update_applied"] == 1 for r in updates)
    position = []
    for r in updates:
        count = sum(r[f"distill/full_kl_position/{b}/token_count"] for b in BINS)
        total = sum(r[f"distill/full_kl_position/{b}/sum"] for b in BINS)
        assert abs(total / count - r["distill/full_kl"]) < 1e-5
        if r["rollout/data_reuse/update_index"] in (1, 16):
            for b in BINS:
                stem = f"distill/full_kl_position/{b}/"
                position.append(
                    dict(
                        experiment=tag,
                        outer=r["step/outer"],
                        reuse=r["rollout/data_reuse/update_index"],
                        bin=b,
                        mean=r[stem + "mean"],
                        total=r[stem + "sum"],
                        count=r[stem + "token_count"],
                    )
                )
    system = read_rows(decoded / "system.jsonl.gz")
    gpu = []
    for row in system:
        metrics = row["metrics"]
        if not all(f"gpu.{i}.gpu" in metrics for i in range(8)):
            continue
        gpu.append(
            dict(
                time=datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00")).timestamp(),
                util=st.mean(metrics[f"gpu.{i}.gpu"] for i in range(8)),
                memory=max(metrics[f"gpu.{i}.memoryAllocatedBytes"] for i in range(8)) / 2**30,
            )
        )
    vllm = read_rows(data / tag / "vllm-metrics.jsonl")
    phases = []
    for row in valid:
        seconds = row.get("perf/timing/validation_seconds")
        if seconds is None:
            continue
        end = row["_timestamp"]
        start = end - seconds
        gs = [r for r in gpu if start + 30 <= r["time"] <= end - 30]
        ks = [
            e["metrics"]["vllm:kv_cache_usage_perc"] * 100
            for r in vllm
            if start + 30 <= r["time"] <= end - 30
            for e in r["engines"]
            if "vllm:kv_cache_usage_perc" in e.get("metrics", {})
        ]
        phases.append(
            dict(
                experiment=tag,
                outer=row["step/outer"],
                seconds=seconds,
                start=start,
                end=end,
                gpu_samples=len(gs),
                gpu_util_mean=st.mean(r["util"] for r in gs) if gs else None,
                gpu_memory_peak_gib=max(r["memory"] for r in gs) if gs else None,
                kv_p95_pct=float(np.percentile(ks, 95)) if ks else None,
                preemptions_observed=interval_counter_delta(vllm, start, end),
            )
        )
    exit_info = json.loads((data / tag / "exit.json").read_text())
    train_s = sum(r["perf/timing/outer_step_seconds"] for r in train)
    valid_s = sum(r.get("perf/timing/validation_seconds", 0) for r in rows)
    overhead = exit_info["elapsed_seconds"] - train_s - valid_s
    assert overhead >= 0
    last = updates[-160:]
    reductions = []
    for step in range(91, 101):
        batch = [r for r in last if r["step/outer"] == step]
        reductions.append(1 - batch[-1]["optim/actor/loss"] / batch[0]["optim/actor/loss"])
    summary = dict(
        wandb_status=meta["status"],
        wandb_history_rows=len(rows),
        logger_parity_rows=parity,
        wandb_system_rows=len(system),
        complete_gpu_samples=len(gpu),
        gpu_peak_gib=max(r["memory"] for r in gpu),
        train_hours=train_s / 3600,
        validation_hours=valid_s / 3600,
        other_hours=overhead / 3600,
        late_batch_loss_reduction_mean=st.mean(reductions),
        last_outer_first_loss=last[-16]["optim/actor/loss"],
        last_outer_last_loss=last[-1]["optim/actor/loss"],
        max_grad_norm=max(r["optim/actor/grad_norm_preclip"] for r in updates),
        clip_threshold_exceeded_fraction=st.mean(r["optim/actor/grad_norm_preclip"] > 1 for r in updates),
        nonfinite_loss_count=sum(r.get("optim/actor/loss_nonfinite", 0) for r in updates),
        nonfinite_grad_count=sum(r.get("optim/actor/grad_norm_preclip_nonfinite", 0) for r in updates),
        skipped_updates=max(r.get("optim/actor/skipped_updates", 0) for r in updates),
    )
    return dict(rows=rows, updates=updates, train=train, valid=valid, position=position, phases=phases, summary=summary)


def render(runs, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D

    fv = load_style()
    fv.apply_rc("paper", fontsize=8)
    colors = {"E41": fv.HUES["vteal"], "E43": fv.HUES["vcoral"]}
    out.mkdir(parents=True, exist_ok=True)
    audits = {}

    def finish(fig, name, caption, ignore=()):
        fig.text(0.5, 0.018, caption, ha="center", fontsize=7, color=fv.NEUTRALS["baseline"])
        fig.canvas.draw()
        audits[name] = fv.audit(fig, ignore=ignore)
        for ext in ("png", "pdf"):
            fig.savefig(out / f"{name}.{ext}", dpi=260)
        plt.close(fig)

    def clean(ax, xlabel="Outer step"):
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", alpha=0.45)
        ax.spines[["top", "right"]].set_visible(False)

    handles = [Line2D([], [], color=c, label=tag) for tag, c in colors.items()]
    fig, axes = plt.subplots(3, 2, figsize=(5.5, 6.0))
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.12, top=0.96, hspace=0.65, wspace=0.35)
    for ax, bench, title in zip(axes.flat, BENCHES, TITLES, strict=False):
        for tag, run in runs.items():
            key = f"eval/{bench}/score/mean"
            points = [r for r in run["valid"] if key in r]
            if points:
                ax.plot([r["step/outer"] for r in points], [100 * r[key] for r in points], "o-", color=colors[tag])
        ax.set(title=title, ylabel="Accuracy (%)", ylim=(0, 100 if bench in BENCHES[:2] else 45))
        clean(ax)
    axes[-1, -1].axis("off")
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.05), ncol=2, frameon=False)
    # Benchmark groups intentionally use different accuracy ranges.
    finish(fig, "validation", "avg@16; E41 has no step 0 or AIME24/26; no smoothing.", ignore=("facet-ylim-mismatch",))

    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.8))
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.17, top=0.94, hspace=0.55, wspace=0.4)
    for tag, run in runs.items():
        for row_idx, points in enumerate([run["train"], run["valid"]]):
            if row_idx == 0:
                lengths = [r["rollout/response_length/mean"] / 1000 for r in points]
                caps = [r["rollout/finish_reason/length_count"] / 64 * 100 for r in points]
            else:
                common = ["math500", "amc23", "aime25"]
                counts = [8000, 640, 480]
                lengths = [
                    sum(r[f"eval/{b}/generation/response_length/mean"] * n for b, n in zip(common, counts, strict=True))
                    / 9120
                    / 1000
                    for r in points
                ]
                caps = [
                    sum(r[f"eval/{b}/generation/finish_reason/length_count"] for b in common) / 9120 * 100
                    for r in points
                ]
            for col, values in enumerate([lengths, caps]):
                axes[row_idx, col].plot([r["step/outer"] for r in points], values, color=colors[tag])
    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            ax.set(
                title=["Train: cap 7,168", "Validation: common 3 sets"][i],
                ylabel=["Mean length (k tokens)", "Length-capped (%)"][j],
                ylim=(0, None),
            )
            clean(ax)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.055), ncol=2, frameon=False)
    # Training and validation have different caps and therefore separate ranges.
    finish(
        fig,
        "length-and-cap",
        "Validation uses MATH/AMC/AIME25 only; cap rate is not repetition rate.",
        ignore=("facet-ylim-mismatch",),
    )

    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.8))
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.24, top=0.94, hspace=0.55, wspace=0.4)
    for i, (tag, run) in enumerate(runs.items()):
        for idx, style in [(1, "--"), (16, "-")]:
            points = [r for r in run["updates"] if r["rollout/data_reuse/update_index"] == idx]
            for j, key in enumerate(["distill/full_kl", "optim/actor/loss"]):
                axes[i, j].plot([r["step/outer"] for r in points], [r[key] for r in points], style, color=colors[tag])
        for j, ax in enumerate(axes[i]):
            ax.set(
                title=f"{tag}: " + ["Raw vocabulary KL", "Optimized objective"][j],
                ylabel="Nats / token",
                ylim=(0, 0.52),
            )
            clean(ax)
    styles = [
        Line2D([], [], color=fv.NEUTRALS["vink"], linestyle=s, label=t)
        for s, t in [("--", "Before update 1"), ("-", "Before update 16")]
    ]
    fig.legend(handles=handles + styles, loc="lower center", bbox_to_anchor=(0.5, 0.05), ncol=2, frameon=False)
    finish(fig, "kl-learning", "Each point is pre-update; new outer steps use new trajectories.")

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.1))
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.37, top=0.88, wspace=0.35)
    for ax, (tag, run) in zip(axes, runs.items(), strict=True):
        for start, end, style in [(1, 5, "--"), (96, 100, "-")]:
            means = [
                st.mean(
                    r["optim/actor/loss"]
                    for r in run["updates"]
                    if start <= r["step/outer"] <= end and r["rollout/data_reuse/update_index"] == idx
                )
                for idx in range(1, 17)
            ]
            ax.plot(range(1, 17), means, style, color=colors[tag])
        ax.set(title=tag, ylabel="Mean objective (nats/token)", xticks=[1, 4, 8, 12, 16], ylim=(0, 0.36))
        clean(ax, "Reuse observation index")
    windows = [
        Line2D([], [], color=fv.NEUTRALS["vink"], linestyle=s, label=t)
        for s, t in [("--", "Outer 1–5"), ("-", "Outer 96–100")]
    ]
    fig.legend(handles=handles + windows, loc="lower center", bbox_to_anchor=(0.5, 0.055), ncol=2, frameon=False)
    finish(fig, "reuse", "Five batches equally weighted; 16 observations span 15 applied updates.")

    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.5))
    fig.subplots_adjust(left=0.12, right=0.83, bottom=0.14, top=0.93, hspace=0.5, wspace=0.25)
    norm = LogNorm(vmin=0.03, vmax=1.5)
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("kl", [fv.NEUTRALS["paper"], fv.HUES["vteal"]])
    for i, (tag, run) in enumerate(runs.items()):
        for j, idx in enumerate([1, 16]):
            points = [r for r in run["updates"] if r["rollout/data_reuse/update_index"] == idx]
            values = np.asarray([[r[f"distill/full_kl_position/{b}/mean"] for r in points] for b in BINS])
            assert values.min() >= norm.vmin and values.max() <= norm.vmax
            im = axes[i, j].imshow(
                values,
                origin="lower",
                aspect="auto",
                extent=[0.5, 100.5, 0, 100],
                norm=norm,
                cmap=cmap,
                interpolation="nearest",
            )
            axes[i, j].set(
                title=f"{tag}: before update {idx}",
                xlabel="Outer step",
                ylabel="Relative position (%)",
                yticks=[0, 50, 100],
            )
    cax = fig.add_axes([0.87, 0.23, 0.025, 0.54])
    fig.colorbar(im, cax=cax, label="Raw KL (log scale)")
    finish(fig, "position-kl", "Same color scale; token-weighted bins use each response's actual length.")

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.4))
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.34, top=0.94, wspace=0.4)
    phases = [
        ("train_hours", "Training", "vteal"),
        ("validation_hours", "Validation", "vorange"),
        ("other_hours", "Setup / other", "vgray"),
    ]
    bottom = np.zeros(2)
    for key, label, color in phases:
        values = [run["summary"][key] for run in runs.values()]
        axes[0].bar(list(runs), values, bottom=bottom, label=label, color=fv.HUES[color])
        bottom += values
    axes[0].set(ylabel="Total wall time (hours)")
    clean(axes[0], "Run")
    for tag, run in runs.items():
        rows = run["phases"]
        axes[1].plot([r["outer"] for r in rows], [r["seconds"] / 60 for r in rows], "o-", color=colors[tag], label=tag)
    axes[1].set(ylabel="Validation time (minutes)", ylim=(0, None))
    clean(axes[1])
    fig.legend(
        *axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.13), ncol=3, frameon=False
    )
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.06), ncol=2, frameon=False)
    finish(fig, "time-budget", "Totals include recorded validation; phase colors apply only to the bars.")

    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.8))
    fig.subplots_adjust(left=0.14, right=0.97, bottom=0.17, top=0.94, hspace=0.55, wspace=0.4)
    fields = [
        ("gpu_util_mean", "Mean GPU utilization (%)"),
        ("gpu_memory_peak_gib", "Peak GPU memory (GiB)"),
        ("kv_p95_pct", "Per-engine KV p95 (%)"),
        ("preemptions_observed", "Observed preemptions"),
    ]
    for ax, (key, title) in zip(axes.flat, fields, strict=True):
        for tag, run in runs.items():
            rows = [r for r in run["phases"] if r[key] is not None]
            ax.plot([r["outer"] for r in rows], [r[key] for r in rows], "o-", color=colors[tag])
        limits = {
            "gpu_util_mean": (-5, 105),
            "gpu_memory_peak_gib": (0, 80),
            "kv_p95_pct": (-5, 105),
            "preemptions_observed": (-350, 8500),
        }
        ax.set(ylabel=title, ylim=limits[key])
        clean(ax)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.055), ncol=2, frameon=False)
    finish(fig, "validation-resources", "W&B GPU samples + vLLM counters; approximate windows, no missing-value fill.")
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.8))
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.29, top=0.88, wspace=0.35)
    for ax, (tag, run) in zip(axes, runs.items(), strict=True):
        ax.plot(
            [r["step/optimizer"] for r in run["updates"]],
            [r["optim/actor/grad_norm_preclip"] for r in run["updates"]],
            color=colors[tag],
            linewidth=0.6,
        )
        ax.axhline(1, linestyle="--", color=fv.NEUTRALS["baseline"])
        ax.set(title=tag, ylabel="Preclip gradient norm", yscale="log", ylim=(0.1, 10))
        clean(ax, "Optimizer update")
    threshold = Line2D([], [], linestyle="--", color=fv.NEUTRALS["baseline"], label="Clip threshold = 1")
    fig.legend(handles=handles + [threshold], loc="lower center", bbox_to_anchor=(0.5, 0.065), ncol=3, frameon=False)
    finish(fig, "gradient-health", "All 1,600 updates per run; clipping exceedance is not a skipped update.")
    dump(out / "figure-audit.json", audits)
    return audits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    runs = {}
    for tag, rid in RUNS.items():
        matches = list((args.logs_root / "derived/MirrorOPD-verl").glob(f"*-{rid}"))
        if len(matches) != 1:
            raise ValueError(f"Expected one decoded W&B directory for {rid}")
        runs[tag] = summarize_run(tag, matches[0], args.data)
    write_csv(args.data / "position-kl-detail.csv", [r for run in runs.values() for r in run["position"]])
    write_csv(args.data / "validation-resources.csv", [r for run in runs.values() for r in run["phases"]])
    dump(args.data / "wandb-diagnostics.json", {tag: run["summary"] for tag, run in runs.items()})
    audits = render(runs, args.out)
    print(json.dumps({tag: run["summary"] for tag, run in runs.items()}, indent=2))
    print("Figure audit:", json.dumps(audits))


if __name__ == "__main__":
    main()
