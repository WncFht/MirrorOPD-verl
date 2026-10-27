# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Read-only audit of the three full-length mirror observation runs.

Outer 1 is warmup, 2/3 are unprofiled observations, and 4 is a rank-0
sampled trace. Host intervals, GPU kernel work and device interval unions
are reported separately; none is silently substituted for another.
"""

import argparse
import csv
import gzip
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from analyze import category, union_length
from semantic_trace import attribute


def read_json(path):
    return json.loads(path.read_text())


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def mean(values):
    return statistics.mean(values) if values else None


def clipped_intervals(events, window):
    return [
        (max(e["start_ns"], window["start_ns"]), min(e["end_ns"], window["end_ns"]))
        for e in events
        if e["start_ns"] < window["end_ns"] and e["end_ns"] > window["start_ns"]
    ]


def wall_interval(event):
    offset = event["wall_ns"] - event["mono_ns"]
    return (event["start_ns"] + offset) / 1e9, (event["end_ns"] + offset) / 1e9


def device_stats(events):
    kernels = [e for e in events if e.get("cat") == "kernel" and e.get("dur", 0) > 0]
    if not kernels:
        return {}
    devices = {e.get("args", {}).get("device") for e in kernels}
    assert len(devices) == 1, devices
    compute, comm = [], []
    totals = defaultdict(float)
    for e in kernels:
        kind = category(e["name"])
        totals[kind] += e["dur"] / 1000
        (comm if kind == "NCCL" else compute).append((e["ts"], e["ts"] + e["dur"]))
    c, n, u = union_length(compute), union_length(comm), union_length(compute + comm)
    return {
        "kernel_sum_ms_by_category": dict(totals),
        "compute_union_ms": c / 1000,
        "nccl_union_ms": n / 1000,
        "kernel_union_ms": u / 1000,
        "overlap_ms": (c + n - u) / 1000,
        "nccl_without_compute_ms": (u - c) / 1000,
        "first_to_last_kernel_ms": (max(b for a, b in compute + comm) - min(a for a, b in compute + comm)) / 1000,
    }


def trace_summary(path):
    with gzip.open(path, "rt") as stream:
        data = json.load(stream)
    events = data["traceEvents"]
    annotations = [e for e in events if e.get("ph") == "X" and e.get("cat") == "user_annotation"]
    mini = [e for e in annotations if e["name"].startswith("mini_batch")]
    assert sorted(e["name"] for e in mini) == ["mini_batch0", "mini_batch1"]
    launches = {
        e["args"]["correlation"]: e
        for e in events
        if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})
    }
    cpu = {
        e["args"]["External id"]: e for e in events if e.get("cat") == "cpu_op" and "External id" in e.get("args", {})
    }

    def training_launch(event):
        launch = launches.get(event.get("args", {}).get("correlation"))
        if launch is None:
            launch = cpu.get(event.get("args", {}).get("External id"))
        # Use the launching CPU process/time, including its asynchronous comm
        # threads. Never classify GPU work by a host span's elapsed duration.
        return launch is not None and any(
            m["pid"] == launch["pid"] and m["ts"] <= launch["ts"] < m["ts"] + m["dur"] for m in mini
        )

    gpu_categories = {"kernel", "gpu_memcpy", "gpu_memset"}
    training_gpu = [e for e in events if e.get("cat") in gpu_categories and training_launch(e)]
    relevant = [e for e in events if e.get("cat") not in gpu_categories] + training_gpu
    semantics = attribute(relevant)
    copies = defaultdict(lambda: {"count": 0, "sum_ms": 0.0, "bytes": 0})
    for e in training_gpu:
        if e.get("cat") == "gpu_memcpy":
            record = copies[e["name"]]
            record["count"] += 1
            record["sum_ms"] += e.get("dur", 0) / 1000
            record["bytes"] += e.get("args", {}).get("bytes", 0)
    memory = [e["args"] for e in events if e.get("name") == "[memory]" and e.get("args", {}).get("Device Type") == 1]
    return {
        "path": str(path),
        "scope": "Rank 0, outer 4, kernels launched during mini_batch0/1 only",
        "all_capture_annotations": {
            name: sum(e["name"] == name for e in annotations)
            for name in [
                "full_vocab/teacher_forward",
                "full_vocab/anchor_forward",
                "full_vocab/head_loss",
                "mini_batch0",
                "mini_batch1",
            ]
        },
        "mini_batch_cpu_ms": [e["dur"] / 1000 for e in sorted(mini, key=lambda e: e["ts"])],
        "training_gpu": device_stats(training_gpu),
        "whole_capture_gpu": device_stats(events),
        "training_semantics": semantics,
        "training_copies": dict(copies),
        "sampled_capture_allocated_peak_gib": max((e.get("Total Allocated", 0) for e in memory), default=0) / 2**30,
        "sampled_capture_reserved_peak_gib": max((e.get("Total Reserved", 0) for e in memory), default=0) / 2**30,
    }


def analyze_job(root, job, gpu_rows):
    directory = root / job["name"]
    config = yaml.safe_load((directory / "training/resolved_config.yaml").read_text())
    actor = config["actor_rollout_ref"]["actor"]
    rollout = config["actor_rollout_ref"]["rollout"]
    loss = config["distillation"]["distillation_loss"]
    assert config["trainer"]["total_training_steps"] == 4 and actor["ppo_epochs"] == 16
    assert config["data"]["train_batch_size"] == actor["ppo_mini_batch_size"] == 64
    assert config["data"]["max_prompt_length"] == 4096 and config["data"]["max_response_length"] == 7168
    assert actor["ppo_max_token_len_per_gpu"] == 32768 and not rollout["enforce_eager"]
    assert config["trainer"]["test_freq"] == config["trainer"]["save_freq"] == -1
    assert config["actor_rollout_ref"]["model"]["enable_gradient_checkpointing"]
    assert loss["full_vocab"]["backend"] == "precompute_fused" and loss["full_vocab"]["token_chunk_size"] == 128
    assert config["trainer"]["n_gpus_per_node"] == 4
    assert actor["strategy"] == "fsdp2"
    assert actor["fsdp_config"]["model_dtype"] == "fp32"
    assert actor["fsdp_config"]["param_offload"] and actor["fsdp_config"]["optimizer_offload"]
    assert not actor["fsdp_config"]["use_no_sync_for_gradient_accumulation"]
    expected = {"E44": ("mdpo", 0, 1), "E45": ("mdpo", 0, 0.3), "E46": ("bpo", 0, 0.3)}[job["name"]]
    assert tuple(loss["mirror"][key] for key in ("objective", "kappa", "eta")) == expected
    events = [r["data"] for r in read_rows(directory / "metrics.jsonl")]
    updates = [r for r in events if "optim/actor/update_applied" in r]
    outer = sorted([r for r in events if "perf/timing/outer_step_seconds" in r], key=lambda r: r["step/outer"])
    assert [r["step/update"] for r in updates] == list(range(1, 65))
    assert [r["step/optimizer"] for r in updates] == list(range(1, 65))
    assert [r["step/outer"] for r in outer] == [1, 2, 3, 4]
    assert all(r["optim/actor/update_applied"] == 1 and r["optim/actor/update_skipped"] == 0 for r in updates)
    assert all(r["optim/actor/grad_norm_preclip_nonfinite"] == 0 for r in updates)
    metric = "distill/full_kl" if job["name"] == "E44" else f"distill/{loss['mirror']['objective']}/objective"
    assert all(math.isfinite(r[metric]) and math.isfinite(r["optim/actor/grad_norm_preclip"]) for r in updates)
    position_keys = [key for key in updates[0] if "_position/" in key]
    assert len(position_keys) == 30
    reconstruction = []
    for r in updates:
        assert all(key in r and math.isfinite(r[key]) for key in position_keys)
        total = sum(r[k] for k in position_keys if k.endswith("/sum"))
        count = sum(r[k] for k in position_keys if k.endswith("/token_count"))
        reconstruction.append(abs(total / count - r[metric]))
        for key in ("distill/teacher_kl", "distill/target_kl", "distill/anchor_kl", "distill/target_log_z"):
            if job["name"] != "E44":
                assert math.isfinite(r[key])
    timeline = [event for p in (directory / "timeline").glob("*.jsonl") for event in read_rows(p)]
    assert len({(e["host"], e["clock_domain"]) for e in timeline}) == 1
    assert not [e for e in timeline if e.get("status") not in (None, "ok")]
    actor_pids = {e["pid"] for e in timeline if e["name"] == "actor/forward_backward"}
    assert len(actor_pids) == 4
    actor_windows = sorted([e for e in timeline if e["name"] == "timer/update_actor"], key=lambda e: e["start_ns"])
    step_windows = sorted([e for e in timeline if e["name"] == "timer/step"], key=lambda e: e["start_ns"])
    assert len(actor_windows) == len(step_windows) == 4
    host_names = [
        "actor/forward_backward",
        "actor/optimizer_step",
        "full_vocab/head_loss",
        "full_vocab/backbone",
        "full_vocab/hidden_read",
        "full_vocab/hidden_read_wait",
        "full_vocab/micro_batch_h2d",
        "engine/transfer",
    ]
    measurements = []
    for index, row in enumerate(outer):
        window = actor_windows[index]
        phase = [
            e
            for e in timeline
            if e["pid"] in actor_pids and e["start_ns"] >= window["start_ns"] and e["end_ns"] <= window["end_ns"]
        ]
        by_name = {}
        for name in host_names:
            ranks = defaultdict(list)
            for e in phase:
                if e["name"] == name:
                    ranks[str(e["rank"])].append(e)
            by_name[name] = {
                rank: {"count": len(es), "host_union_seconds": union_length(clipped_intervals(es, window)) / 1e9}
                for rank, es in ranks.items()
            }
        micros = [e for e in phase if e["name"] == "engine/micro_batch" and e["role"] == "actor"]
        microcounts = Counter(str(e["rank"]) for e in micros)
        assert len(microcounts) == 4
        training_input = sum(e["input_tokens"] for e in micros)
        assert training_input == row["perf/load_balance/mean"] * 4 * 16
        phase_gpu = {}
        # Use actual host timer intervals to align sampled NVML values. Drop
        # 0.25 s at each boundary, and report sample counts rather than zeros.
        for name in ["timer/gen", "timer/teacher_forward", "timer/update_actor", "timer/update_weights"]:
            ws = [
                e
                for e in timeline
                if e["name"] == name
                and e["start_ns"] >= step_windows[index]["start_ns"]
                and e["end_ns"] <= step_windows[index]["end_ns"]
            ]
            ranges = [wall_interval(e) for e in ws]
            samples = [
                g
                for g in gpu_rows
                if g["job"] == job["name"] and any(a + 0.25 <= g["timestamp"] <= b - 0.25 for a, b in ranges)
            ]
            phase_gpu[name] = {
                "sample_count": len(samples),
                "gpu_busy_mean": mean([g["gpu_busy_percent"] for g in samples]),
                "memory_gib_max_sample": max((g["memory_gib"] for g in samples), default=None),
                "power_w_mean": mean([g["power_w"] for g in samples]),
            }
        measurements.append(
            {
                "outer": index + 1,
                "measurement_kind": "warmup" if index == 0 else "profiled" if index == 3 else "unprofiled",
                "timers": {k.removeprefix("perf/timing/"): v for k, v in row.items() if k.startswith("perf/timing/")},
                "anchor_seconds_max": row.get("perf/anchor/prepare_seconds_max"),
                "response_tokens": row["rollout/response_tokens"],
                "input_tokens": training_input // 16,
                "training_input_tokens_including_reuse": training_input,
                "actor_input_tokens_per_second_including_reuse": training_input
                / row["perf/timing/update_actor_seconds"],
                "length_limit_fraction": row["rollout/response_length/padded_width_fraction"],
                "actor_allocated_lifetime_peak_gib": row["perf/actor/allocated_peak_gib"],
                "actor_reserved_lifetime_peak_gib": row["perf/actor/reserved_peak_gib"],
                "train_microbatches_per_rank": dict(microcounts),
                "actor_host_spans": by_name,
                "sampled_gpu_by_phase": phase_gpu,
            }
        )
    steady = measurements[1:3]
    avg = {k: mean([r["timers"][k] for r in steady]) for k in steady[0]["timers"]}
    avg["anchor_seconds_max"] = mean([r["anchor_seconds_max"] for r in steady if r["anchor_seconds_max"] is not None])
    avg["actor_fraction_of_outer"] = sum(r["timers"]["update_actor_seconds"] for r in steady) / sum(
        r["timers"]["outer_step_seconds"] for r in steady
    )
    avg["actor_input_tokens_per_second_including_reuse"] = sum(
        r["training_input_tokens_including_reuse"] for r in steady
    ) / sum(r["timers"]["update_actor_seconds"] for r in steady)
    traces = list((directory / "traces").glob("*.json.gz"))
    assert len(traces) == 1
    exit_record = read_json(directory / "exit.json")
    assert exit_record["returncode"] == 0
    return {
        "preset": job["name"],
        "algorithm": loss["mirror"],
        "execution": {
            "strategy": actor["strategy"],
            "model_dtype": actor["fsdp_config"]["model_dtype"],
            "gradient_checkpointing": True,
            "gradient_no_sync": False,
            "rollout_graph": True,
            "actor_token_budget": 32768,
            "head_token_chunk": 128,
        },
        "exit": exit_record,
        "checks": {
            "updates": len(updates),
            "outer_steps": len(outer),
            "finite_loss_and_gradients": True,
            "skipped_updates": 0,
            "position_reconstruction_max_abs_error": max(reconstruction),
            "four_ranks": True,
            "config_matches_plan": True,
        },
        "objective_first_last": [updates[0][metric], updates[-1][metric]],
        "unprofiled_means": avg,
        "outer_measurements": measurements,
        "trace": trace_summary(traces[0]),
    }


def compact_summary(result):
    """Publish small derived evidence; keep full per-rank intervals in outputs."""
    compact = {key: value for key, value in result.items() if key != "jobs"}
    compact["jobs"] = []
    for job in result["jobs"]:
        steady = job["outer_measurements"][1:3]
        host = {
            name: mean(
                [values["host_union_seconds"] for row in steady for values in row["actor_host_spans"][name].values()]
            )
            for name in steady[0]["actor_host_spans"]
        }
        trace = job["trace"]
        semantics = trace["training_semantics"]["semantic_kernel_sum_ms"]
        head = {
            key: value
            for key, value in semantics.items()
            if key.startswith("full_vocab/") and key not in ("full_vocab/backbone", "full_vocab/student_forward")
        }
        compact["jobs"].append(
            {
                "preset": job["preset"],
                "algorithm": job["algorithm"],
                "execution": job["execution"],
                "exit": job["exit"],
                "checks": job["checks"],
                "objective_first_last": job["objective_first_last"],
                "unprofiled_means": job["unprofiled_means"],
                "host_mean_seconds_per_rank_per_outer": host,
                "steady_sampled_gpu_by_phase": [row["sampled_gpu_by_phase"] for row in steady],
                "outer_measurements": [
                    {
                        key: value
                        for key, value in row.items()
                        if key not in ("actor_host_spans", "sampled_gpu_by_phase")
                    }
                    for row in job["outer_measurements"]
                ],
                "trace": {key: value for key, value in trace.items() if key != "training_semantics"}
                | {
                    "head_kernel_sum_ms": head,
                    "head_total_kernel_ms": sum(head.values()),
                    "head_fraction_of_all_training_kernel_work": sum(head.values()) / sum(semantics.values()),
                    "unattributed_kernel_ms": semantics.get("unattributed", 0),
                },
            }
        )
    return compact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, help="Optional compact JSON for documentation")
    args = parser.parse_args()
    root = args.root.resolve()
    plan = read_json(root / "plan.json")
    status = read_json(root / "status.json")
    assert status["state"] == "completed" and plan["source_commit"] == status["source_commit"]
    with (root / "gpu.csv").open() as stream:
        gpu_rows = [
            row | {key: float(row[key]) for key in ("timestamp", "gpu_busy_percent", "memory_gib", "power_w")}
            for row in csv.DictReader(stream)
        ]
    result = {
        "source_commit": plan["source_commit"],
        "evidence_root": str(root),
        "scope": "Three natural-generation observation runs; no matched-input speed or quality experiment.",
        "limits": [
            "Two unprofiled outer steps per method; no confidence interval.",
            "Trace includes teacher/anchor before first two updates; training attribution filters those out.",
            "Host spans overlap CUDA and each other; sums are not a wall-time partition.",
            "NVML is sampled at about 1 Hz; maxima can miss transients.",
            "Rollout preemptions were not reported for every request; missing is not zero.",
            "Idle gates and shared lock checked before jobs; no continuous per-process GPU audit was recorded.",
        ],
        "jobs": [],
    }
    for job in plan["jobs"]:
        result["jobs"].append(analyze_job(root, job, gpu_rows))
        print(job["name"], "verified and analyzed", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(
            json.dumps(compact_summary(result), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )


if __name__ == "__main__":
    main()
