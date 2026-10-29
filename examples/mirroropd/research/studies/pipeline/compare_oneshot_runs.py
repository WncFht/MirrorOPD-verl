#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Recompute selected historical runs from frozen API data; no network or GPU calls."""

import argparse
import gzip
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from examples.mirroropd.research.core.paths import REPO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    root = REPO
    base = root / "examples/mirroropd/docs/archive/evidence/root-data/oneshot-comparison-20261021"

    def read(p):
        op = gzip.open if p.suffix == ".gz" else open
        with op(p, "rt") as f:
            return [json.loads(s) for s in f if s.strip()]

    def num(v):
        return isinstance(v, int | float) and math.isfinite(v)

    def stat(v):
        v = [x for x in v if num(x)]
        return (
            {"n": len(v), "mean": statistics.mean(v), "median": statistics.median(v), "min": min(v), "max": max(v)}
            if v
            else {"n": 0}
        )

    def value(r, k):
        return r.get(k) if num(r.get(k)) else 0

    stage_keys = [
        "step",
        "gen",
        "old_log_prob",
        "opd_teacher",
        "teacher_forward",
        "teacher_transfer",
        "adv",
        "update_actor",
        "update_weights",
    ]

    def summarize(rows):
        d = {k: stat(r.get("timing_s/" + k) for r in rows) for k in stage_keys}
        d["tokens"] = stat(r.get("perf/total_num_tokens") for r in rows)
        d["response_length"] = stat(r.get("response_length/mean") for r in rows)
        d["effective_tokens_per_second"] = sum(value(r, "perf/total_num_tokens") for r in rows) / sum(
            value(r, "timing_s/step") for r in rows
        )
        d["generation_tokens_per_second"] = sum(value(r, "response_length/mean") * 64 for r in rows) / sum(
            value(r, "timing_s/gen") for r in rows
        )
        d["rollout_span"] = stat(r.get("timing_s/rollout_batch") for r in rows)
        d["gen_minus_span"] = stat(
            r["timing_s/gen"] - r["timing_s/rollout_batch"] for r in rows if num(r.get("timing_s/rollout_batch"))
        )
        d["gen_minus_longest_request"] = stat(
            r["timing_s/gen"] - r["timing_s/agent_loop/generate_sequences/max"]
            for r in rows
            if num(r.get("timing_s/agent_loop/generate_sequences/max"))
        )
        return d

    out = {}
    for name in ["mdxt04lm", "mzwm9q04", "5aub1k4h", "rcxtujbi"]:
        p = base / name
        assert (p / "source.json").exists(), p
        h = read(p / "history.jsonl.gz")
        h = [r for r in h if num(r.get("timing_s/step"))]
        s = read(p / "system-sampled.jsonl.gz")
        s = [r for r in s if all(num(r.get(f"system.gpu.{i}.gpu")) for i in range(8))]
        cfg = json.loads((p / "config.json").read_text())
        assert (
            cfg["data"]["train_batch_size"] == 64
            and cfg["actor_rollout_ref"]["rollout"]["n"] == 1
            and cfg["trainer"]["n_gpus_per_node"] == 8
        )
        steady = [r for r in h if r["training/global_step"] > 1 and not value(r, "timing_s/save_checkpoint")]
        windows = defaultdict(list)
        residuals = []
        for r in h:
            end = r["_timestamp"]
            test = value(r, "timing_s/testing")
            if test:
                windows["validation"].append((end - test + 15, end - 15))
            if r not in steady:
                continue
            end -= test
            start = end - r["timing_s/step"]
            windows["training"].append((start + 1, end - 1))
            if "timing_s/opd_teacher" in r:
                stages = [
                    ("gen", value(r, "timing_s/gen")),
                    ("reward", value(r, "timing_s/reward")),
                    ("old_log_prob", value(r, "timing_s/old_log_prob")),
                    ("teacher_and_adv", value(r, "timing_s/adv")),
                    ("update_actor", value(r, "timing_s/update_actor")),
                    ("update_weights", value(r, "timing_s/update_weights")),
                ]
            else:
                stages = [
                    ("gen", value(r, "timing_s/gen")),
                    ("teacher_and_adv", value(r, "timing_s/teacher_transfer") + value(r, "timing_s/teacher_forward")),
                    ("old_log_prob", value(r, "timing_s/old_log_prob")),
                    ("adv", value(r, "timing_s/adv")),
                    ("update_actor", value(r, "timing_s/update_actor")),
                    ("update_weights", value(r, "timing_s/update_weights")),
                ]
            residual = r["timing_s/step"] - sum(v for k, v in stages)
            residuals.append(residual)
            assert residual >= -0.01, (name, r["training/global_step"], residual)
            cursor = start
            for k, v in stages:
                # GPU system rows average ~15 seconds; use conservative long-stage interior.
                if cursor + max(residual, 0) + 15 < cursor + v - 15:
                    windows[k].append((cursor + max(residual, 0) + 15, cursor + v - 15))
                cursor += v

        def gpu_stat(w, samples_by_time=s):
            samples = [r for r in samples_by_time if any(a <= r["_timestamp"] <= b for a, b in w)]
            return {
                "rows": len(samples),
                "busy_percent": stat(stat(r[f"system.gpu.{i}.gpu"] for i in range(8))["mean"] for r in samples),
                "peak_device_memory_gib": max(
                    (r.get(f"system.gpu.{i}.memoryAllocatedBytes", 0) / 1024**3 for r in samples for i in range(8)),
                    default=None,
                ),
                "power_watts": stat(
                    stat(r.get(f"system.gpu.{i}.powerWatts") for i in range(8)).get("mean") for r in samples
                ),
            }

        out[name] = {
            "source": json.loads((p / "source.json").read_text()),
            "hardware": json.loads((p / "hardware.json").read_text()),
            "all_steady_exclude_checkpoint": summarize(steady),
            "steps_2_12": summarize([r for r in steady if 2 <= r["training/global_step"] <= 12]),
            "gpu": {k: gpu_stat(w) for k, w in windows.items()},
            "system_sampling_interval_s": stat(
                b["_timestamp"] - a["_timestamp"] for a, b in zip(s, s[1:], strict=False)
            ),
            "timer_residual_s": stat(residuals),
            "validation_seconds": stat(r.get("timing_s/testing") for r in h),
            "validation_fraction": sum(value(r, "timing_s/testing") for r in h)
            / sum(value(r, "timing_s/testing") + value(r, "timing_s/step") for r in h),
        }
        print(
            name,
            json.dumps(
                {
                    "stages": {
                        k: round(v["mean"], 3)
                        for k, v in out[name]["all_steady_exclude_checkpoint"].items()
                        if isinstance(v, dict) and v.get("n")
                    },
                    "q": out[name]["all_steady_exclude_checkpoint"]["effective_tokens_per_second"],
                    "val": out[name]["validation_seconds"],
                    "gpu": {
                        k: {"n": v["rows"], "busy": v["busy_percent"].get("mean"), "mem": v["peak_device_memory_gib"]}
                        for k, v in out[name]["gpu"].items()
                    },
                    "sample_interval": out[name]["system_sampling_interval_s"],
                },
                ensure_ascii=False,
            ),
        )
    for kind in ["eager", "graph"]:
        h = read(
            root / f"examples/mirroropd/docs/archive/evidence/root-data/local-shared4-20261021/{kind}/history.jsonl"
        )
        out["local_" + kind] = summarize([r for r in h if r["training/global_step"] > 1])
        print("local", kind, json.dumps(out["local_" + kind]))
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
