# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Read completed local W&B runs, audit logger parity, and render diagnostics.

Semantic ledger: E41/E43 identify distinct model pairs, not matched loss arms.
Each figure contains one run; validation color identifies the benchmark, and
line style identifies the within-batch observation or averaging window.
Position heatmaps encode raw KL with one shared log color scale; bins are relative
to each response. Time budgets use disjoint recorded training/validation totals.
GPU phase attribution uses approximate logger-derived windows with 30s margins.
Geometry: equal-size panels; outer/update/elapsed-time axes are never interchanged.
Only actual observations are connected; missing values are not replaced with zero.
"""

import argparse
import hashlib
import json
import statistics as st
from datetime import datetime
from pathlib import Path

import numpy as np

from examples.mirroropd.research.core.io import dump, read_rows, write_csv
from examples.mirroropd.research.core.plotting import load_style
from examples.mirroropd.research.studies.qwen_teachers.figure_inputs import (
    DEFAULT_MANIFEST,
    check_decoded_inputs,
    check_plot_inputs,
    check_source,
    decoding_label,
    load_inputs,
    metric_label,
    percent_label,
    setting_label,
)

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
    if meta["run"]["run_id"] != RUNS[tag]:
        raise ValueError(f"Decoded run identity mismatch for {tag}")
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


def attach_base_reference(runs, logs_root):
    """Reuse a measured initial Base evaluation after checking the frozen protocol."""
    import hashlib

    root = logs_root / "derived/MirrorOPD-verl"
    source = root / "run-20261026_223012-f0e44ec7"
    target = root / "run-20261027_133032-53dd553d"
    configs = [json.loads((d / "config.json").read_text()) for d in (source, target)]
    checked = ["data", "actor_rollout_ref.model", "actor_rollout_ref.rollout"]
    for field in checked:
        values = []
        for config in configs:
            value = config
            for key in field.split("."):
                value = value[key]
            if field.endswith("rollout"):
                value = {k: v for k, v in value.items() if k not in ("trace", "disable_log_stats")}
            values.append(value)
        if values[0] != values[1]:
            raise ValueError(f"Shared Base evaluation protocol mismatch: {field}")
    check_source(source / "history.jsonl", runs["E41"]["figure_inputs"]["base_reference_history"])
    initial = [
        r for r in read_rows(source / "history.jsonl") if r.get("step/outer") == 0 and "eval/math500/score/mean" in r
    ]
    if len(initial) != 1 or any(r["step/outer"] == 0 for r in runs["E41"]["valid"]):
        raise ValueError("Expected one external Base reference and no E41 in-run initial validation")
    # Keep the measured row separate: its timestamp/resources belong to E39.
    runs["E41"]["base_reference"] = initial[0]
    return {
        "source_run": "f0e44ec7",
        "target_run": RUNS["E41"],
        "meaning": "Shared untrained Base evaluation, not an E41 in-run validation event",
        "checked_config_sections": checked,
        "ignored_rollout_fields": ["trace", "disable_log_stats"],
        "files": {
            str(p.relative_to(logs_root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (source / "config.json", source / "history.jsonl", target / "config.json")
        },
    }


def attach_teacher_references(runs):
    """Attach full-set teacher evidence validated against each frozen run config."""
    provenance = {}
    for tag, run in runs.items():
        inputs = run["figure_inputs"]
        reference = inputs["teacher_reference"]
        run["teacher_reference"] = reference["rows"] if reference else []
        provenance[tag] = {
            "config": inputs["config"],
            "manifest_sha256": inputs["manifest_sha256"],
            "teacher_reference": reference,
        }
    return provenance


def render(runs, out):
    """One experiment per figure; validation colors identify benchmarks.

    Semantic ledger: the E41 step-zero point is the independently measured shared
    Base reference; only accuracy/length/cap use that reference. Resource and time
    plots retain actual run events. KL line styles encode pre-update observations.
    Geometry: validation metrics use separate axes, relative-position heatmaps use
    the same log color scale, and optimizer steps never stand for outer steps.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    from matplotlib.lines import Line2D

    fv = load_style()
    fv.apply_rc("paper", fontsize=8)
    palette = [fv.HUES[k] for k in ("vteal", "vorange", "vteal", "vviolet", "vorange")]
    neutral = fv.NEUTRALS["baseline"]
    audits = {}
    plotted = {}
    for tag, run in runs.items():
        directory = out / tag.lower()
        directory.mkdir(parents=True, exist_ok=True)
        benches = ["math500", "amc23", "aime25"] if tag == "E41" else BENCHES
        valid = ([run["base_reference"]] if "base_reference" in run else []) + run["valid"]
        color = fv.HUES["vteal"]
        inputs = run["figure_inputs"]
        evaluation = inputs["evaluation"]
        training = inputs["training"]
        student, teacher = inputs["student"], inputs["teacher"]
        setting = setting_label(inputs)
        student_metric = metric_label(evaluation)
        plotted[tag] = {
            "run_id": RUNS[tag],
            "student": student,
            "teacher": teacher,
            "enable_thinking": inputs["enable_thinking"],
            "student_accuracy_metric": (
                f"{student_metric}: mean correctness over {evaluation['samples']} responses per question"
            ),
            "protocol": inputs,
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "validation_response_cap": evaluation["response_cap"],
            "training_response_cap": training["response_cap"],
            "teacher_reference": run["teacher_reference"],
            "validation": [
                {
                    "outer": r["step/outer"],
                    "benchmark": b,
                    "accuracy": r[f"eval/{b}/score/mean"],
                    "ci_low": r[f"eval/{b}/score/mean_ci_low"],
                    "ci_high": r[f"eval/{b}/score/mean_ci_high"],
                    "mean_length": r[f"eval/{b}/generation/response_length/mean"],
                    "cap_count": r[f"eval/{b}/generation/finish_reason/length_count"],
                    "source_run": "f0e44ec7" if tag == "E41" and r["step/outer"] == 0 else RUNS[tag],
                }
                for r in valid
                for b in benches
            ],
        }

        def panels(n=2, rows=1, width=6.5):
            fig, axes = plt.subplots(rows, n, figsize=(width, 2.85 * rows + 1.10), squeeze=False)
            fig.subplots_adjust(
                left=0.11,
                right=0.97,
                bottom=0.32 if rows == 1 else 0.22,
                top=0.71 if rows == 1 else 0.82,
                wspace=0.50,
                hspace=0.60,
            )
            return fig, axes

        def clean(ax, label="Outer step"):
            ax.set_xlabel(label)
            ax.grid(axis="y", alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
            if label == "Outer step":
                ax.set(xlim=(-3, 103), xticks=[0, 50, 100])

        def finish(fig, name, caption, handles=None, ignore=(), tag=tag, directory=directory, setting=setting):
            fig.text(0.5, 0.98, setting, ha="center", va="top", fontsize="medium")
            if handles:
                fig.legend(
                    handles=handles,
                    loc="upper center",
                    bbox_to_anchor=(0.53, 0.86 if fig.get_figheight() < 5 else 0.91),
                    ncol=min(3, len(handles)),
                    frameon=False,
                )
            fig.text(0.5, 0.025, caption, ha="center", fontsize="medium", color=neutral)
            fig.canvas.draw()
            issues = fv.audit(fig, ignore=ignore)
            audits[f"{tag}/{name}"] = issues
            if any(fv.problem_severity(p) == "error" for p in issues):
                raise RuntimeError(f"{tag}/{name}: {issues}")
            for ext in ("png", "pdf", "svg"):
                fig.savefig(directory / f"{name}.{ext}", dpi=240)
            plt.close(fig)

        # E39-style validation: dataset-colored accuracy, length, and cap panels.
        # Five datasets use two rows so that each panel retains at most 3 series.
        groups = [benches] if tag == "E41" else [BENCHES[:2], BENCHES[2:]]
        fig, axes = panels(3, rows=len(groups), width=7.3)
        handles = []
        for row_idx, group in enumerate(groups):
            for b in group:
                i = BENCHES.index(b)
                points = [r for r in plotted[tag]["validation"] if r["benchmark"] == b]
                x = [r["outer"] for r in points]
                count = (500 if b == "math500" else 40 if b == "amc23" else 30) * evaluation["samples"]
                for ax, field, factor in zip(
                    axes[row_idx], ["accuracy", "mean_length", "cap_count"], [100, 0.001, 100 / count], strict=True
                ):
                    ax.plot(
                        x,
                        [r[field] * factor for r in points],
                        marker=["o", "s", "^", "D", "v"][i],
                        color=palette[i],
                        markersize=3,
                        label=TITLES[i],
                    )
                axes[row_idx, 0].fill_between(
                    x,
                    [100 * r["ci_low"] for r in points],
                    [100 * r["ci_high"] for r in points],
                    color=palette[i],
                    alpha=0.12,
                )
                reference = next((r for r in run["teacher_reference"] if r["task"] == b), None)
                if reference:
                    axes[row_idx, 0].axhline(100 * reference["accuracy"], color=palette[i], linestyle="--", linewidth=1)
                if tag == "E41":
                    for ax, field, factor in zip(
                        axes[row_idx], ["accuracy", "mean_length", "cap_count"], [100, 0.001, 100 / count], strict=True
                    ):
                        ax.plot([0], [points[0][field] * factor], marker="D", color=palette[i], markersize=4)
            for ax, label in zip(
                axes[row_idx],
                [
                    f"Accuracy: {student_metric} (%)",
                    "Mean length (k tokens)",
                    f"Length-capped (%)\ncap = {evaluation['response_cap']:,} tokens",
                ],
                strict=True,
            ):
                ax.set_ylabel(label)
                ax.margins(y=0.12)
                ax.set_ylim(bottom=-3 if label != "Mean length (k tokens)" else -0.8)
                clean(ax)
            handles.extend(axes[row_idx, 0].get_legend_handles_labels()[0])
        caption = (
            f"Student {student_metric} = mean correctness of {evaluation['samples']} responses/question; "
            "bands: 95% question-bootstrap CI."
        )
        if tag == "E41":
            caption += (
                "\nTeacher full-set reference unavailable; step 0 diamonds: shared Base evaluation from E39."
                f"\nEvaluation: {decoding_label(evaluation)}"
            )
        else:
            teacher_eval = inputs["teacher_reference"]["evaluation"]
            teacher_metric = metric_label(teacher_eval)
            handles.append(Line2D([], [], color=neutral, linestyle="--", label=f"Teacher {teacher_metric} reference"))
            caption += (
                f"\nTeacher {teacher_metric}: "
                + "; ".join(
                    f"{TITLES[BENCHES.index(r['task'])]} {percent_label(r['accuracy'])}%"
                    for r in run["teacher_reference"]
                )
                + f".\nIndependent teacher: {teacher_eval['samples']} responses/question; "
                + decoding_label(teacher_eval)
            )
        finish(fig, "validation", caption, handles, ignore=("facet-ylim-mismatch",))

        fig, axes = panels()
        for ax, key, label, scale in zip(
            axes.flat,
            ["rollout/response_length/mean", "rollout/finish_reason/length_count"],
            ["Train length (k tokens)", f"Train length-capped (%)\ncap = {training['response_cap']:,} tokens"],
            [0.001, 100 / (training["batch"] * training["samples"])],
            strict=True,
        ):
            ax.plot([r["step/outer"] for r in run["train"]], [r[key] * scale for r in run["train"]], color=color)
            ax.set(ylabel=label, ylim=(0, None))
            clean(ax)
        finish(
            fig,
            "training-behavior",
            f"Training cap {training['response_cap']:,}; changing DAPO batches; cap rate is not repetition rate.",
        )

        styles = [
            Line2D([], [], color=color, linestyle=s, label=t)
            for s, t in [("--", "Before update 1"), ("-", "Before update 16")]
        ]
        fig, axes = panels()
        for idx, style in [(1, "--"), (16, "-")]:
            points = [r for r in run["updates"] if r["rollout/data_reuse/update_index"] == idx]
            for ax, key, title in zip(
                axes.flat,
                ["distill/full_kl", "optim/actor/loss"],
                ["Raw vocabulary KL", "Optimized objective"],
                strict=True,
            ):
                ax.plot([r["step/outer"] for r in points], [r[key] for r in points], style, color=color)
                ax.set(title=title, ylabel="Nats / token", ylim=(0, 0.52))
                clean(ax)
        finish(
            fig,
            "kl-learning",
            "New outer steps use new trajectories; raw KL and actual loss are recorded separately.",
            styles,
        )

        fig, axes = panels(1, width=5.5)
        ax = axes[0, 0]
        windows = []
        for start, end, style in [(1, 5, "--"), (96, 100, "-")]:
            means = [
                st.mean(
                    r["optim/actor/loss"]
                    for r in run["updates"]
                    if start <= r["step/outer"] <= end and r["rollout/data_reuse/update_index"] == idx
                )
                for idx in range(1, 17)
            ]
            (line,) = ax.plot(range(1, 17), means, style, color=color, label=f"Outer {start}–{end}")
            windows.append(line)
        ax.set(ylabel="Mean objective (nats/token)", xticks=[1, 4, 8, 12, 16], ylim=(0, 0.36))
        clean(ax, "Reuse observation index")
        finish(fig, "reuse", "Five batches equally weighted; 16 observations span 15 applied updates.", windows)

        fig, axes = panels()
        fig.subplots_adjust(right=0.82)
        norm = LogNorm(vmin=0.03, vmax=1.5)
        cmap = LinearSegmentedColormap.from_list("kl", [fv.NEUTRALS["paper"], color])
        for ax, idx in zip(axes.flat, [1, 16], strict=True):
            points = [r for r in run["updates"] if r["rollout/data_reuse/update_index"] == idx]
            values = np.asarray([[r[f"distill/full_kl_position/{b}/mean"] for r in points] for b in BINS])
            assert values.min() >= norm.vmin and values.max() <= norm.vmax
            im = ax.imshow(
                values,
                origin="lower",
                aspect="auto",
                extent=[0.5, 100.5, 0, 100],
                norm=norm,
                cmap=cmap,
                interpolation="nearest",
            )
            ax.set(
                title=f"Before update {idx}", xlabel="Outer step", ylabel="Relative position (%)", yticks=[0, 50, 100]
            )
        cax = fig.add_axes([0.86, 0.30, 0.025, 0.39])
        fig.colorbar(im, cax=cax, label="Raw KL (log scale)")
        finish(fig, "position-kl", "Token-weighted raw KL bins use each response's actual length.")

        fig, axes = panels()
        phases = [
            ("train_hours", "Training", "vteal"),
            ("validation_hours", "Validation", "vorange"),
            ("other_hours", "Setup / other", "vgray"),
        ]
        bottom = 0
        for key, label, hue in phases:
            value = run["summary"][key]
            axes[0, 0].bar([tag], [value], bottom=bottom, label=label, color=fv.HUES[hue])
            bottom += value
        axes[0, 0].set_ylabel("Total wall time (hours)")
        clean(axes[0, 0], "Run")
        points = run["phases"]
        axes[0, 1].plot([r["outer"] for r in points], [r["seconds"] / 60 for r in points], "o-", color=color)
        axes[0, 1].set(
            ylabel="Validation time (minutes)",
            ylim=(-2, max(r["seconds"] / 60 for r in points) * 1.12),
        )
        clean(axes[0, 1])
        finish(
            fig,
            "time-budget",
            (
                "Actual E41 events only; shared Base reference adds no wall time."
                if tag == "E41"
                else "Actual E43 events, including its own initial validation."
            ),
            axes[0, 0].get_legend_handles_labels()[0],
        )

        fig, axes = panels(2, rows=2)
        fields = [
            ("gpu_util_mean", "Mean GPU utilization (%)", (-5, 105)),
            ("gpu_memory_peak_gib", "Peak GPU memory (GiB)", (0, 80)),
            ("kv_p95_pct", "Per-engine KV p95 (%)", (-5, 105)),
            ("preemptions_observed", "Observed preemptions", (-350, 8500)),
        ]
        for ax, (key, label, limits) in zip(axes.flat, fields, strict=True):
            points = [r for r in run["phases"] if r[key] is not None]
            ax.plot([r["outer"] for r in points], [r[key] for r in points], "o-", color=color)
            ax.set(ylabel=label, ylim=limits)
            clean(ax)
        finish(
            fig,
            "validation-resources",
            "W&B GPU samples + vLLM counters; approximate windows; missing values stay missing.",
        )

        fig, axes = panels(1, width=5.5)
        ax = axes[0, 0]
        ax.plot(
            [r["step/optimizer"] for r in run["updates"]],
            [r["optim/actor/grad_norm_preclip"] for r in run["updates"]],
            color=color,
            linewidth=0.6,
        )
        ax.axhline(1, linestyle="--", color=neutral)
        ax.set(ylabel="Preclip gradient norm", yscale="log", ylim=(0.1, 10))
        clean(ax, "Optimizer update")
        finish(
            fig,
            "gradient-health",
            "All 1,600 updates applied; clipping exceedance is not a skipped update.",
            [Line2D([], [], color=neutral, linestyle="--", label="Clip threshold = 1")],
        )
    dump(out / "figure-audit.json", audits)
    dump(out / "validation-inputs.json", plotted)
    return audits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--figure-inputs",
        type=Path,
        default=DEFAULT_MANIFEST,
    )
    args = parser.parse_args()
    runs = {}
    for tag, rid in RUNS.items():
        matches = list((args.logs_root / "derived/MirrorOPD-verl").glob(f"*-{rid}"))
        if len(matches) != 1:
            raise ValueError(f"Expected one decoded W&B directory for {rid}")
        inputs = load_inputs(tag, matches[0] / "config.json", args.figure_inputs)
        if inputs["run_id"] != rid:
            raise ValueError(f"Run identity mismatch for {tag}")
        check_decoded_inputs(inputs, matches[0])
        check_plot_inputs(inputs, args.data)
        runs[tag] = summarize_run(tag, matches[0], args.data)
        runs[tag]["figure_inputs"] = inputs
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "position-kl-detail.csv", [r for run in runs.values() for r in run["position"]])
    write_csv(args.out / "validation-resources.csv", [r for run in runs.values() for r in run["phases"]])
    dump(args.out / "wandb-diagnostics.json", {tag: run["summary"] for tag, run in runs.items()})
    provenance = attach_base_reference(runs, args.logs_root)
    args.out.mkdir(parents=True, exist_ok=True)
    dump(args.out / "base-reference-provenance.json", provenance)
    teacher_provenance = attach_teacher_references(runs)
    dump(args.out / "teacher-reference-provenance.json", teacher_provenance)
    audits = render(runs, args.out)
    print(json.dumps({tag: run["summary"] for tag, run in runs.items()}, indent=2))
    print("Figure audit:", json.dumps(audits))


if __name__ == "__main__":
    main()
