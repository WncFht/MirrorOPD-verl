#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Audit recorded timers, sparse GPU probes and replay-buffer progress.

Stage windows are reconstructed from history timestamps and ordered timers.
The unaccounted timer residual and a configurable guard are removed from both
stage boundaries. These are inferred windows, NOT profiler spans. GPU values
remain sampled observations; no continuous busy-time integral is inferred.
Replay tail spans assume monotone prompt completion within each fixed window.
"""

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from zoneinfo import ZoneInfo

from examples.mirroropd.research.core.io import dump as write_json
from examples.mirroropd.research.core.io import read_rows
from examples.mirroropd.research.studies.pipeline.analyze_run import write_rows

PHASES = ("gen", "old_log_prob", "adv", "update_actor", "update_weights")
STATUS = re.compile(
    r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+):pending: (\d+), "
    r"running: (\d+), finished: (\d+), failure: (\d+)"
)


def describe(values):
    if not values:
        return None
    v = sorted(values)
    return {
        "count": len(v),
        "mean": mean(v),
        "median": median(v),
        "min": v[0],
        "max": v[-1],
        "p90": v[math.ceil(0.9 * len(v)) - 1],
    }


def intervals(history, guard):
    out = []
    for row in history:
        if row.get("training/global_step", 0) <= 1 or "timing_s/step" not in row:
            continue
        end = row["_timestamp"] - row.get("timing_s/testing", 0)
        start = end - row["timing_s/step"]
        residual = row["timing_s/step"] - sum(row["timing_s/" + name] for name in PHASES)
        if residual < -0.001:
            raise ValueError("Ordered timers exceed step duration")
        cursor = start
        for name in PHASES:
            duration = row["timing_s/" + name]
            out.append(
                {
                    "step": row["training/global_step"],
                    "phase": name,
                    "approx_start": cursor,
                    "approx_end": cursor + duration,
                    "core_start": cursor + max(0, residual) + guard,
                    "core_end": cursor + duration - guard,
                    "residual_seconds": residual,
                }
            )
            cursor += duration
        if "timing_s/testing" in row:
            out.append(
                {
                    "step": row["training/global_step"],
                    "phase": "validation",
                    "approx_start": end,
                    "approx_end": row["_timestamp"],
                    "core_start": end + guard,
                    "core_end": row["_timestamp"] - guard,
                    "residual_seconds": 0,
                }
            )
    return out


def probes_by_phase(system, spans):
    result = {}
    fields = {
        "busy_percent": ("gpu", 1),
        "memory_gib": ("memoryAllocatedBytes", 1 / 1024**3),
        "power_watts": ("powerWatts", 1),
        "sm_clock_mhz": ("smClock", 1),
        "temperature_c": ("temp", 1),
    }
    for phase in (*PHASES, "validation"):
        rr = [
            r
            for r in system
            if "system.gpu.0.gpu" in r
            and any(w["phase"] == phase and w["core_start"] <= r["_timestamp"] <= w["core_end"] for w in spans)
        ]
        entry = {"gpu_probe_rows": len(rr), "groups": {}, "per_gpu": {}}
        for label, ids in [("physical_0_3", range(4)), ("physical_4_7", range(4, 8))]:
            entry["groups"][label] = {
                key: describe([mean(r[f"system.gpu.{i}.{metric}"] * scale for i in ids) for r in rr])
                for key, (metric, scale) in fields.items()
            }
            entry["groups"][label]["any_busy_probe_rows"] = sum(
                any(r[f"system.gpu.{i}.gpu"] > 0 for i in ids) for r in rr
            )
        for i in range(8):
            entry["per_gpu"][str(i)] = {
                key: describe([r[f"system.gpu.{i}.{metric}"] * scale for r in rr])
                for key, (metric, scale) in fields.items()
            }
        # CPU and GPU streams have distinct timestamps. Do not zip their rows.
        cpu_rows = [
            r
            for r in system
            if "system.proc.memory.rssMB" in r
            and any(w["phase"] == phase and w["core_start"] <= r["_timestamp"] <= w["core_end"] for w in spans)
        ]
        entry["driver_rss_mib"] = describe([r["system.proc.memory.rssMB"] for r in cpu_rows])
        entry["cpu_metric_raw"] = describe([r["system.cpu"] for r in cpu_rows if "system.cpu" in r])
        result[phase] = entry
    return result


def extract_status(text):
    rows = []
    for m in STATUS.finditer(text):
        stamp = datetime.strptime(m[1], "%Y-%m-%d %H:%M:%S,%f").replace(tzinfo=ZoneInfo("Asia/Shanghai"))
        row = {"timestamp": stamp.timestamp(), "time_asia_shanghai": stamp.isoformat()}
        row.update(dict(zip(("pending", "running", "finished", "failure"), map(int, m.groups()[1:]), strict=True)))
        row["prompt_groups"] = sum(row[k] for k in ("pending", "running", "finished", "failure"))
        rows.append(row)
    return sorted(rows, key=lambda r: r["timestamp"])


def tail_analysis(status, history):
    completed = []
    for row in history:
        if "timing_s/testing" not in row:
            continue
        end, duration = row["_timestamp"], row["timing_s/testing"]
        selected = [r for r in status if end - duration <= r["timestamp"] <= end]
        windows = []
        for r in selected:
            if (
                not windows
                or r["finished"] < windows[-1][-1]["finished"]
                or r["prompt_groups"] != windows[-1][-1]["prompt_groups"]
            ):
                windows.append([])
            windows[-1].append(r)
        bounds = {}
        for limit in (1, 2, 4, 7):
            pairs = [
                (a, b)
                for window in windows
                for a, b in zip(window, window[1:], strict=False)
                if a["pending"] == b["pending"] == a["failure"] == b["failure"] == 0
                and max(a["running"], b["running"]) <= limit
            ]
            seconds = sum(b["timestamp"] - a["timestamp"] for a, b in pairs)
            bounds[str(limit)] = {
                "consecutive_observed_span_seconds": seconds,
                "fraction_of_testing_timer": seconds / duration,
                "intervals": len(pairs),
            }
        completed.append(
            {
                "step": row["training/global_step"],
                "start": end - duration,
                "end": end,
                "testing_seconds": duration,
                "status_rows": len(selected),
                "window_count": len(windows),
                "window_sizes": [w[0]["prompt_groups"] for w in windows],
                "nested_tail_spans": bounds,
                "windows": windows,
            }
        )
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-data", type=Path, required=True)
    parser.add_argument("--console-log", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--guard-seconds", type=float, default=2)
    args = parser.parse_args()
    if args.guard_seconds < 0:
        parser.error("guard must be nonnegative")
    config = json.loads((args.api_data / "config.json").read_text())
    if (
        config["trainer"]["n_gpus_per_node"],
        config["distillation"]["n_gpus_per_node"],
        config["data"]["train_batch_size"],
        config["actor_rollout_ref"]["rollout"]["n"],
    ) != (4, 4, 64, 1):
        parser.error(
            "This audit's physical groups and idle reservation calculation require the recorded 4+4 / batch64 / n1 run"
        )
    history = read_rows(args.api_data / "history.jsonl")
    system = read_rows(args.api_data / "system-sampled.jsonl.gz")
    console = args.console_log.read_text()
    status = extract_status(console)
    spans = intervals(history, args.guard_seconds)
    steady = [r for r in history if r.get("training/global_step", 0) > 1]
    timing = {name: describe([r["timing_s/" + name] for r in steady]) for name in (*PHASES, "step")}
    total = {name: sum(r["timing_s/" + name] for r in history) for name in PHASES}
    console_steps = sorted(set(map(int, re.findall(r"\bstep:(\d+) - ", console))))
    last_console_line = next(line for line in console.splitlines() if f"step:{max(console_steps)} - " in line)
    last_console_fields = {"training/global_step": max(console_steps)}
    for field in last_console_line.split(" - ")[1:]:
        key, _, value = field.partition(":")
        try:
            last_console_fields[key] = float(re.sub(r"^np\.(?:float\d*|int\d*)\((.*)\)$", r"\1", value))
        except ValueError:
            continue
    # _runtime can be finalized on the following explicit-step log call.
    runtime_origins = [b["_timestamp"] - a["_runtime"] for a, b in zip(history, history[1:], strict=False)]
    validation = tail_analysis(status, history)
    val_end = validation[-1]["end"]
    end_probes = [r for r in system if "system.gpu.0.gpu" in r and val_end - 180 <= r["_timestamp"] <= val_end]
    suffix = []
    for r in reversed(end_probes):
        if any(r[f"system.gpu.{i}.gpu"] > 0 for i in range(4)):
            break
        suffix.append(r)
    suffix.reverse()
    args.out.mkdir(parents=True, exist_ok=True)
    write_rows(args.out / "replay-status.jsonl", status)
    write_rows(args.out / "inferred-stage-windows.jsonl", spans)
    write_json(args.out / "last-console-step.json", last_console_fields)
    write_json(
        args.out / "pipeline-analysis.json",
        {
            "method": "Ordered timer reconstruction; sparse probe means, NOT kernel busy-time integrals.",
            "guard_seconds": args.guard_seconds,
            "timing_steady": timing,
            "timing_total_recorded_steps": total,
            "gpu_phase_probes": probes_by_phase(system, spans),
            "guard_sensitivity": {str(g): probes_by_phase(system, intervals(history, g)) for g in (0, 1, 3)},
            "teacher_reserved_idle_gpu_hours": {
                name: seconds * 4 / 3600 for name, seconds in total.items() if name != "gen"
            },
            "validation_teacher_reserved_gpu_hours": sum(v["testing_seconds"] for v in validation) * 4 / 3600,
            "completed_validation": validation,
            "validation_end_zero_busy_suffix": {
                "gpu_probe_rows": len(suffix),
                "first_timestamp": suffix[0]["_timestamp"] if suffix else None,
                "last_timestamp": suffix[-1]["_timestamp"] if suffix else None,
                "observed_span_seconds": suffix[-1]["_timestamp"] - suffix[0]["_timestamp"] if suffix else 0,
                "interpretation": "No student GPU busy in these observations; not proof of zero work between probes.",
            },
            "runtime_vs_next_timestamp_origin": describe(runtime_origins),
            "api_metric_steps": [r["training/global_step"] for r in history],
            "console_metric_steps": console_steps,
            "status_after_last_api_timestamp": [r for r in status if r["timestamp"] > history[-1]["_timestamp"]],
        },
    )
    write_json(
        args.out / "source.json",
        {
            "run": "wncfht/MirrorOPD-verl/b38hw8v0",
            "api_snapshot": args.api_data.name,
            "console_source": "W&B run file output.log, fetched read-only with the authorized netrc",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status_timezone": "Asia/Shanghai",
            "console_metric_step_max": max(console_steps),
            "api_metric_step_max": history[-1]["training/global_step"],
            "limitations": (
                "GPU probes ~15 s apart; status logs ~60 s apart; no existing kernel trace or teacher request timer."
            ),
        },
    )
    print(
        {
            "api_rows": len(history),
            "console_step_max": max(console_steps),
            "status_rows": len(status),
            "validation_windows": validation[0]["window_count"],
            "phase_probe_rows": {k: v["gpu_probe_rows"] for k, v in probes_by_phase(system, spans).items()},
        }
    )


if __name__ == "__main__":
    main()
