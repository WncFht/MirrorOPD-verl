# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Infer rollout/actor/validation windows from sparse canonical metric rows.

Requires one optimizer update per outer step and a pre-update metric row. System
telemetry is sampled; trimming reduces boundary ambiguity but is not a CUDA trace.
"""

import argparse
import bisect
import collections
import gzip
import json
import statistics as s
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Phase-aligned sampled GPU metrics for a synchronous single-update run"
    )
    parser.add_argument("--data", type=Path, required=True)
    r = parser.parse_args().data
    h = [json.loads(x) for x in (r / "history.jsonl").read_text().splitlines()]
    with gzip.open(r / "system-sampled.jsonl.gz", "rt") as f:
        sys = [json.loads(x) for x in f]
    g = [x for x in sys if isinstance(x.get("system.gpu.0.gpu"), (int, float))]
    by = collections.defaultdict(list)
    for x in h:
        by[x["step/outer"]].append(x)
    windows = []
    checks = []
    for step, rows in sorted(by.items()):
        outer = next(x for x in rows if x.get("perf/timing/outer_step_seconds") is not None)
        pre = next(x for x in rows if x.get("step/update") == step - 1)
        post = next(x for x in rows if x.get("step/update") == step and x.get("perf/timing/outer_step_seconds") is None)

        def t(name, outer=outer):
            return outer.get("perf/timing/" + name + "_seconds") or 0.0

        start_actor = pre["_timestamp"]
        end_actor = post["_timestamp"]
        end_gen = start_actor - t("adv") - t("old_log_prob") - t("teacher_forward") - t("teacher_transfer")
        begin_gen = end_gen - t("rollout_pipeline")
        if step > 1:
            windows.extend(
                [
                    {"phase": "train_rollout", "start": begin_gen, "end": end_gen, "step": step},
                    {"phase": "actor_update", "start": start_actor, "end": end_actor, "step": step},
                ]
            )
        if t("validation"):
            windows.append(
                {
                    "phase": "validation",
                    "start": outer["_timestamp"] - t("validation"),
                    "end": outer["_timestamp"],
                    "step": step,
                }
            )
        checks.append(
            {
                "step": step,
                "actor_boundary_error": end_actor - start_actor - t("update_actor"),
                "step_begin": outer["_timestamp"] - t("validation") - t("outer_step"),
                "rollout_begin": begin_gen,
            }
        )
    windows.sort(key=lambda x: x["start"])
    starts = [x["start"] for x in windows]

    def select(row, phase, trim):
        ts = row["_timestamp"]
        i = bisect.bisect_right(starts, ts) - 1
        if i < 0:
            return False
        w = windows[i]
        return w["phase"] == phase and w["start"] + trim <= ts <= w["end"] - trim

    def describe(a):
        a = sorted(a)
        return (
            {
                "count": len(a),
                "mean": s.mean(a),
                "median": s.median(a),
                "p10": a[int(0.1 * (len(a) - 1))],
                "p90": a[int(0.9 * (len(a) - 1))],
                "min": a[0],
                "max": a[-1],
            }
            if a
            else None
        )

    out = {
        "scope": (
            "W&B sampled system telemetry; stage boundaries inferred from pre/post update timestamps "
            "and timers; not a CUDA trace"
        ),
        "gpu_sample_count": len(g),
        "interval_seconds": describe([b["_timestamp"] - a["_timestamp"] for a, b in zip(g, g[1:], strict=False)]),
        "windows": windows,
        "boundary_checks": checks,
        "groups": {},
    }
    for phase in ("train_rollout", "actor_update", "validation"):
        for trim in (0, 3, 5, 8):
            selected = [x for x in g if select(x, phase, trim)]
            result = {
                "sample_times": len(selected),
                "sample_steps": len(
                    {
                        next(
                            w["step"]
                            for w in windows
                            if w["phase"] == phase and w["start"] <= x["_timestamp"] <= w["end"]
                        )
                        for x in selected
                    }
                ),
                "per_gpu": {},
            }
            for metric in [
                "gpu",
                "memory",
                "memoryAllocatedBytes",
                "memoryAllocated",
                "powerWatts",
                "powerPercent",
                "smClock",
                "memoryClock",
                "temp",
            ]:
                a = [
                    x[f"system.gpu.{i}.{metric}"] / (2**30 if metric == "memoryAllocatedBytes" else 1)
                    for x in selected
                    for i in range(8)
                    if x.get(f"system.gpu.{i}.{metric}") is not None
                ]
                result[metric] = describe(a)
            a = [x[f"system.gpu.{i}.gpu"] for x in selected for i in range(8)]
            result["gpu_at_least_95_fraction"] = sum(v >= 95 for v in a) / len(a) if a else None
            result["gpu_at_most_10_fraction"] = sum(v <= 10 for v in a) / len(a) if a else None
            result["all_eight_at_least_90_fraction"] = (
                s.mean(all(x[f"system.gpu.{i}.gpu"] >= 90 for i in range(8)) for x in selected) if selected else None
            )
            for i in range(8):
                result["per_gpu"][str(i)] = {
                    m: describe(
                        [x[f"system.gpu.{i}.{m}"] / (2**30 if m == "memoryAllocatedBytes" else 1) for x in selected]
                    )
                    for m in ["gpu", "memory", "memoryAllocatedBytes", "powerWatts"]
                }
            out["groups"][phase + f"_trim{trim}s"] = result
            print(
                phase,
                trim,
                "samples",
                len(selected),
                "GPUs avg",
                [round(result["per_gpu"][str(i)]["gpu"]["mean"], 1) for i in range(8)] if selected else [],
                "gpu avg/median",
                (round(result["gpu"]["mean"], 2), result["gpu"]["median"]) if selected else None,
                "memory avg",
                round(result["memory"]["mean"], 2) if selected else None,
                "gib",
                round(result["memoryAllocatedBytes"]["mean"], 2) if selected else None,
                "power",
                round(result["powerWatts"]["mean"], 1) if selected else None,
                "low<=10",
                round(result["gpu_at_most_10_fraction"], 3) if selected else None,
            )
    print("actor boundary error", describe([abs(x["actor_boundary_error"]) for x in checks]))
    print("start gen difference", describe([x["rollout_begin"] - x["step_begin"] for x in checks if x["step"] > 1]))
    (r / "phase-analysis.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
