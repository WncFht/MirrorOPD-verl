#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Compare frozen standalone/shared snapshots without changing training.

Matched global steps exclude step 1 and use ratio-of-sums token throughput.
System telemetry is sampled, never integrated into GPU busy time. Inferred
phase cores exclude the unallocated timer residual and both boundary guards;
shared teacher transfer/forward form one block because transfer is an aggregate
of onload and offload. Null sparse history never becomes an observation.
"""

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev

from examples.mirroropd.research.analysis.legacy.analyze_run import is_number, read_rows, write_json, write_rows
from examples.mirroropd.research.analysis.legacy.audit_pipeline import extract_status

STAGES = (
    "gen",
    "teacher_forward",
    "teacher_transfer",
    "old_log_prob",
    "adv",
    "update_actor",
    "save_checkpoint",
    "update_weights",
)
METRICS = tuple("timing_s/" + x for x in (*STAGES, "step", "rollout_batch")) + (
    "perf/total_num_tokens",
    "perf/throughput",
    "perf/mfu/actor",
    "prompt_length/mean",
    "response_length/mean",
    "response_length/max",
    "response_length/clip_ratio",
    "critic/score/mean",
    "actor/distillation/loss",
    "actor/distillation/abs_loss",
    "actor/entropy",
    "actor/grad_norm",
    "rollout_corr/kl",
    "rollout_corr/k3_kl",
    "actor/perf/max_memory_allocated_gb",
    "actor/perf/max_memory_reserved_gb",
    "actor/perf/cpu_memory_used_gb",
    "memory/fsdp_worker_peak_allocated_gib_max",
    "rollout/preemptions",
    "teacher/request_latency_s/mean",
)


def describe(values):
    """Summarize finite observations, retaining missingness and sample count."""
    values = sorted(x for x in values if is_number(x))
    if not values:
        return None
    return {
        "n": len(values),
        "mean": mean(values),
        "median": median(values),
        "min": values[0],
        "max": values[-1],
        "stdev": stdev(values) if len(values) > 1 else None,
    }


def training_rows(history):
    """Return uniquely keyed, completed training rows; refuse duplicate global steps."""
    rows = [r for r in history if is_number(r.get("training/global_step")) and is_number(r.get("timing_s/step"))]
    steps = [r["training/global_step"] for r in rows]
    if len(steps) != len(set(steps)):
        raise ValueError("Duplicate global steps need explicit reconciliation")
    return sorted(rows, key=lambda r: r["training/global_step"])


def group_stats(rows, gpus=8, batch=64):
    """Use global work divided by global elapsed time, not a mean of per-step rates."""
    if not rows:
        return None
    times = sum(r["timing_s/step"] for r in rows)
    tokens = sum(r["perf/total_num_tokens"] for r in rows)
    output_tokens = sum(r["response_length/mean"] * batch for r in rows)
    means = {key: describe(r.get(key) for r in rows) for key in METRICS}
    stages = {
        name: sum(r["timing_s/" + name] for r in rows if is_number(r.get("timing_s/" + name))) / len(rows)
        for name in STAGES
        if any(is_number(r.get("timing_s/" + name)) for r in rows)
    }
    residuals = [
        r["timing_s/step"] - sum(r["timing_s/" + k] for k in STAGES if is_number(r.get("timing_s/" + k))) for r in rows
    ]
    if min(residuals) < -0.01:
        raise ValueError("Disjoint stage timers exceed step: inspect timer semantics")
    return {
        "steps": [r["training/global_step"] for r in rows],
        "count": len(rows),
        "metrics": means,
        "stage_mean_seconds": stages,
        "residual_seconds": describe(residuals),
        "timed_seconds": times,
        "valid_tokens": tokens,
        "output_tokens": output_tokens,
        "valid_tokens_per_second": tokens / times,
        "output_tokens_per_second": output_tokens / times,
        "valid_tokens_per_gpu_second": tokens / times / gpus,
        "prompts_per_minute": batch * len(rows) / times * 60,
        "timed_gpu_hours": times * gpus / 3600,
    }


def phase_windows(rows, guard=2.0):
    """Infer conservative cores; combine teacher moves because their split is unrecorded."""
    windows = []
    for r in rows:
        testing = r.get("timing_s/testing")
        testing = testing if is_number(testing) else 0.0
        end = r["_timestamp"] - testing
        start = end - r["timing_s/step"]
        stages = [(k, r["timing_s/" + k]) for k in STAGES if is_number(r.get("timing_s/" + k))]
        teacher = sum(t for k, t in stages if k in ("teacher_forward", "teacher_transfer"))
        ordered = [("gen", r["timing_s/gen"])]
        if teacher:
            ordered.append(("teacher_block", teacher))
        ordered += [(k, t) for k, t in stages if k not in ("gen", "teacher_forward", "teacher_transfer")]
        residual = r["timing_s/step"] - sum(t for _, t in ordered)
        if residual < -0.01:
            raise ValueError("Negative timer residual")
        cursor = start
        for phase, seconds in ordered:
            windows.append(
                {
                    "step": r["training/global_step"],
                    "phase": phase,
                    "approx_start": cursor,
                    "approx_end": cursor + seconds,
                    "core_start": cursor + max(0, residual) + guard,
                    "core_end": cursor + seconds - guard,
                    "residual_seconds": residual,
                }
            )
            cursor += seconds
        if testing:
            windows.append(
                {
                    "step": r["training/global_step"],
                    "phase": "validation",
                    "approx_start": end,
                    "approx_end": r["_timestamp"],
                    "core_start": end + guard,
                    "core_end": r["_timestamp"] - guard,
                    "residual_seconds": 0,
                }
            )
    return windows


def system_stats(rows):
    """Report full-group sample means and device maxima, never zero-fill absent GPUs."""
    fields = {
        "busy_percent": ("gpu", 1),
        "memory_gib": ("memoryAllocatedBytes", 1 / 1024**3),
        "power_watts": ("powerWatts", 1),
        "sm_clock_mhz": ("smClock", 1),
    }
    result = {"groups": {}, "per_gpu": {}}
    for label, ids in [("all_eight", range(8)), ("physical_0_3", range(4)), ("physical_4_7", range(4, 8))]:
        group = {}
        for name, (suffix, scale) in fields.items():
            values = [[r.get(f"system.gpu.{i}.{suffix}") for i in ids] for r in rows]
            values = [[v * scale for v in point] for point in values if all(is_number(v) for v in point)]
            group[name] = describe(mean(v) for v in values)
            if name == "memory_gib":
                group["max_device_memory_gib"] = max((max(v) for v in values), default=None)
                group["sum_device_memory_gib"] = describe(sum(v) for v in values)
        result["groups"][label] = group
    for i in range(8):
        result["per_gpu"][str(i)] = {
            name: describe(
                r[f"system.gpu.{i}.{suffix}"] * scale for r in rows if is_number(r.get(f"system.gpu.{i}.{suffix}"))
            )
            for name, (suffix, scale) in fields.items()
        }
    result["driver_rss_mib"] = describe(r.get("system.proc.memory.rssMB") for r in rows)
    return result


def select_windows(rows, windows, phase=None):
    """Take each telemetry row at most once, and keep short/empty cores empty."""
    spans = [w for w in windows if phase is None or w["phase"] == phase]
    return [r for r in rows if any(w["core_start"] <= r["_timestamp"] <= w["core_end"] for w in spans)]


def flatten(config, prefix=""):
    """Flatten the actual uploaded config for a complete audit of changed fields."""
    out = {}
    for key, value in config.items():
        key = f"{prefix}.{key}" if prefix else key
        out.update(flatten(value, key) if isinstance(value, dict) else {key: value})
    return out


def load_run(directory):
    """Load a frozen export, preserving system/history bounds independently."""
    history = read_rows(directory / "history.jsonl")
    return {
        "directory": str(directory),
        "source": json.loads((directory / "source.json").read_text()),
        "config": json.loads((directory / "config.json").read_text()),
        "hardware": json.loads((directory / "hardware.json").read_text()),
        "history": history,
        "train": training_rows(history),
        "system": read_rows(directory / "system-sampled.jsonl.gz"),
    }


def compare(baseline, shared, guard=2.0):
    """Build paired-step and all-observed analyses with explicit observation limits."""
    ids = sorted(
        {r["training/global_step"] for r in baseline["train"] if r["training/global_step"] > 1}
        & {r["training/global_step"] for r in shared["train"] if r["training/global_step"] > 1}
    )
    if not ids:
        raise ValueError("No common post-warmup steps")
    result = {"matched_steps": ids, "guard_seconds": guard, "runs": {}, "paired_steps": [], "config_diff": {}}
    for label, data in [("standalone", baseline), ("shared", shared)]:
        selected = [r for r in data["train"] if r["training/global_step"] in ids]
        steady = [r for r in data["train"] if r["training/global_step"] > 1]
        windows = phase_windows(selected, guard)
        val = [
            {
                "step": r["training/global_step"],
                "seconds": r["timing_s/testing"],
                "metrics": {
                    k: v
                    for k, v in r.items()
                    if k.startswith(("val-core/", "val-aux/", "validation/")) and is_number(v)
                },
            }
            for r in data["train"]
            if is_number(r.get("timing_s/testing"))
        ]
        ts = sorted(r["_timestamp"] for r in data["system"] if is_number(r.get("system.gpu.0.gpu")))
        step_spans = [
            {
                "core_start": r["_timestamp"]
                - r["timing_s/step"]
                - (r["timing_s/testing"] if is_number(r.get("timing_s/testing")) else 0)
                + guard,
                "core_end": r["_timestamp"]
                - (r["timing_s/testing"] if is_number(r.get("timing_s/testing")) else 0)
                - guard,
            }
            for r in selected
        ]
        result["runs"][label] = {
            "source": data["source"],
            "hardware": data["hardware"],
            "matched": group_stats(selected),
            "all_training": group_stats(data["train"]),
            "steady_all": group_stats(steady),
            "steady_without_checkpoint": group_stats(
                [r for r in steady if not is_number(r.get("timing_s/save_checkpoint"))]
            ),
            "validations": val,
            "validation_seconds": describe(v["seconds"] for v in val),
            "history_timestamp_range": [data["train"][0]["_timestamp"], data["train"][-1]["_timestamp"]],
            "gpu_timestamp_range": [ts[0], ts[-1]],
            "raw_system_rows": len(data["system"]),
            "numeric_gpu_rows": len(ts),
            "gpu_interval_seconds": describe(b - a for a, b in zip(ts, ts[1:], strict=False)),
            "matched_system": system_stats(select_windows(data["system"], step_spans)),
            "matched_phases": {
                phase: system_stats(select_windows(data["system"], windows, phase))
                for phase in dict.fromkeys(w["phase"] for w in windows)
            },
            "matched_windows": windows,
            "validation_system": system_stats(
                select_windows(data["system"], phase_windows(data["train"], guard), "validation")
            ),
        }
    a = {r["training/global_step"]: r for r in baseline["train"]}
    b = {r["training/global_step"]: r for r in shared["train"]}
    for step in ids:
        result["paired_steps"].append(
            {
                "step": step,
                "baseline_seconds": a[step]["timing_s/step"],
                "shared_seconds": b[step]["timing_s/step"],
                "delta_seconds": b[step]["timing_s/step"] - a[step]["timing_s/step"],
                "baseline_tokens": a[step]["perf/total_num_tokens"],
                "shared_tokens": b[step]["perf/total_num_tokens"],
                "prompt_mean_equal": a[step]["prompt_length/mean"] == b[step]["prompt_length/mean"],
            }
        )
    ra, rb = (result["runs"][label]["matched"] for label in ("standalone", "shared"))
    result["comparison"] = {
        "shared_over_baseline_step_time": rb["timed_seconds"] / ra["timed_seconds"],
        "shared_over_baseline_token_throughput": rb["valid_tokens_per_second"] / ra["valid_tokens_per_second"],
        "shared_over_baseline_tokens": rb["valid_tokens"] / ra["valid_tokens"],
        "paired_step_delta": describe(r["delta_seconds"] for r in result["paired_steps"]),
        "shared_slower_steps": sum(r["delta_seconds"] > 0 for r in result["paired_steps"]),
        "logged_throughput_inferred_gpu_divisor": {},
    }
    for label, data in [("standalone", baseline), ("shared", shared)]:
        result["comparison"]["logged_throughput_inferred_gpu_divisor"][label] = describe(
            r["perf/total_num_tokens"] / r["timing_s/step"] / r["perf/throughput"] for r in data["train"]
        )
    ca, cb = flatten(baseline["config"]), flatten(shared["config"])
    result["config_diff"] = {
        k: {"standalone": ca.get(k), "shared": cb.get(k)}
        for k in sorted(ca.keys() | cb.keys())
        if ca.get(k) != cb.get(k)
    }
    return result


def console_observations(path):
    """Persist numerical progress and warning counts, not raw private console content."""
    text = path.read_text()
    steps = [int(s) for s in re.findall(r"\bstep:(\d+) - ", text)]
    status = extract_status(text)
    return {
        "file": "output.log",
        "local_copy_saved_at_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
        "printed_steps": sorted(set(steps)),
        "status_rows": len(status),
        "last_status": status[-1] if status else None,
        "preemption_unavailable_warnings": text.count("backend did not report preemptions"),
        "traceback_occurrences": text.count("Traceback (most recent call last)"),
        "cuda_oom_occurrences": text.count("CUDA out of memory"),
    }, status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline-console", type=Path)
    parser.add_argument("--shared-console", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    a, b = load_run(args.baseline), load_run(args.shared)
    result = compare(a, b)
    console_path = args.out / "console-observations.json"
    sources = {label: data["source"] for label, data in [("standalone", a), ("shared", b)]}
    saved = json.loads(console_path.read_text()) if console_path.exists() else {}
    # Reproduce without private raw logs, but never carry console evidence into a new snapshot.
    result["console"] = saved.get("records", {}) if saved.get("sources") == sources else {}
    for label, path in [("standalone", args.baseline_console), ("shared", args.shared_console)]:
        if path:
            observations, statuses = console_observations(path)
            result["console"][label] = observations
            write_rows(args.out / f"{label}-replay-status.jsonl.gz", statuses)
    if result["console"]:
        write_json(console_path, {"sources": sources, "records": result["console"]})
    write_json(args.out / "comparison.json", result)
    for guard in (0, 1, 3):
        alternate = compare(a, b, guard)
        write_json(
            args.out / f"phase-sensitivity-guard{guard}.json",
            {k: v["matched_phases"] for k, v in alternate["runs"].items()},
        )
    with (args.out / "matched-steps.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result["paired_steps"][0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(result["paired_steps"])
    print(json.dumps(result["comparison"], indent=2))


if __name__ == "__main__":
    main()
