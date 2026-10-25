"""Derive comparable report columns from native results and preserved telemetry."""

import gzip
import json
import math
import re
import statistics
from pathlib import Path

SUMMARY_FIELDS = [
    "model",
    "stage",
    "status",
    "max_num_seqs",
    "gpu_memory_utilization",
    "token_budget",
    "completed",
    "failed",
    "duration",
    "total_output_tokens",
    "output_throughput",
    "mean_output_tokens",
    "p50_output_tokens",
    "p95_output_tokens",
    "p99_output_tokens",
    "max_output_tokens",
    "cap_hit_count",
    "cap_hit_pct",
    "kv_peak_pct",
    "kv_capacity_tokens_min",
    "kv_capacity_tokens_max",
    "preemptions",
    "preemptions_per_million_tokens",
    "gpu_mem_peak_gib",
    "gpu_busy_mean_pct",
    "gpu0_busy_pct",
    "gpu1_busy_pct",
    "gpu2_busy_pct",
    "gpu3_busy_pct",
    "mean_running_requests",
    "running_limit_pct",
    "waiting_sample_pct",
    "mean_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "p99_tpot_ms",
    "mean_e2el_ms",
    "p99_e2el_ms",
    "gpu_window_samples",
    "shared_gpu_samples",
    "resource_isolated",
    "failure_reason",
    "log",
]


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * fraction
    lower, upper = math.floor(index), math.ceil(index)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def metric_total(lines, name):
    return sum(float(line.rsplit(" ", 1)[1]) for line in lines if line.startswith(name + "{"))


def resource_report(path, cap, gpus=4):
    if not path.exists():
        return {}
    with gzip.open(path, "rt") as stream:
        samples = [json.loads(line) for line in stream]
    if not samples:
        return {}
    peak = max(g["used_MiB"] for row in samples for g in row["gpu"]) / 1024
    result = {"gpu_mem_peak_gib": peak}
    queues = []
    for row in samples:
        lines = row.get("engine_metrics", [])
        queues.append(
            (metric_total(lines, "vllm:num_requests_running"), metric_total(lines, "vllm:num_requests_waiting"))
        )
    # Exclude the native client's initial single-request check. Once the measured
    # multi-request window starts, retain idle gaps and idle GPUs in its mean.
    begin = next((i for i, (running, waiting) in enumerate(queues) if running + waiting > 1), None)
    if begin is None:
        return result
    end = max(i for i, (running, waiting) in enumerate(queues) if running > 0 or waiting > 0)
    window = samples[begin : end + 1]
    queue_window = queues[begin : end + 1]
    result["gpu_window_samples"] = len(window)
    # Native DP4/TP1 uses one GPU worker per card. Multiple compute PIDs
    # during generation make this a shared-GPU measurement, even if all
    # requests succeed. Do not infer isolation from utilization alone.
    result["shared_gpu_samples"] = sum(any(len(gpu.get("pids", [])) > 1 for gpu in row["gpu"]) for row in window)
    result["resource_isolated"] = result["shared_gpu_samples"] == 0
    values = [gpu["utilization"] for row in window for gpu in row["gpu"]]
    result["gpu_busy_mean_pct"] = statistics.fmean(values)
    for index in range(gpus):
        values = [gpu["utilization"] for row in window for gpu in row["gpu"] if gpu["gpu"] == index]
        if values:
            result[f"gpu{index}_busy_pct"] = statistics.fmean(values)
    result["mean_running_requests"] = statistics.fmean(running for running, _ in queue_window)
    result["running_limit_pct"] = 100 * result["mean_running_requests"] / (gpus * cap)
    result["waiting_sample_pct"] = 100 * statistics.fmean(waiting > 0 for _, waiting in queue_window)
    return result


def build_report(row, result=None, response_cap=31744):
    cap, memory, tokens = row["candidate"]
    derived = {"max_num_seqs": cap, "gpu_memory_utilization": memory, "token_budget": tokens, "report_schema": 2}
    if result is not None:
        lengths = result.get("output_lens", [])
        errors = result.get("errors", [])
        if len(errors) == len(lengths):
            lengths = [length for length, error in zip(lengths, errors, strict=True) if not error]
        if lengths:
            derived.update(
                mean_output_tokens=statistics.fmean(lengths),
                p50_output_tokens=percentile(lengths, 0.5),
                p95_output_tokens=percentile(lengths, 0.95),
                p99_output_tokens=percentile(lengths, 0.99),
                max_output_tokens=max(lengths),
                cap_hit_count=sum(length >= response_cap for length in lengths),
                cap_hit_pct=100 * sum(length >= response_cap for length in lengths) / len(lengths),
            )
        for metric in ("ttft", "tpot", "e2el"):
            for stat in ("mean", "p99"):
                key = f"{stat}_{metric}_ms"
                value = result.get(key)
                if isinstance(value, (float, int)) and math.isfinite(value):
                    derived[key] = value
    if row.get("observed_peak_kv") is not None:
        derived["kv_peak_pct"] = 100 * row["observed_peak_kv"]
    if row.get("total_output_tokens", 0) > 0 and row.get("preemptions") is not None:
        derived["preemptions_per_million_tokens"] = row["preemptions"] * 1e6 / row["total_output_tokens"]
    directory = Path(row["log"]).parent
    derived.update(resource_report(directory / "telemetry.jsonl.gz", cap))
    log = Path(row["log"]).read_text(errors="replace") if Path(row["log"]).exists() else ""
    capacities = [int(value.replace(",", "")) for value in re.findall(r"GPU KV cache size: ([\d,]+) tokens", log)]
    if capacities:
        derived.update(kv_capacity_tokens_min=min(capacities), kv_capacity_tokens_max=max(capacities))
    if row.get("status") != "complete":
        if "out of memory" in log.lower() or "less than desired GPU memory" in log:
            derived["failure_reason"] = "memory_error"
        elif row.get("response_cap_verified") is False:
            derived["failure_reason"] = "response_budget_not_verified"
        else:
            derived["failure_reason"] = "see_run_log"
    return derived
