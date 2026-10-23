# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Run one training command while recording GPU/host and vLLM HTTP observations."""

import argparse
import ast
import importlib.metadata
import json
import os
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psutil
import pynvml as nv
import requests

NAMES = {
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
    "vllm:generation_tokens_total",
    "vllm:prompt_tokens_total",
    "vllm:num_preemptions_total",
    "vllm:request_success_total",
    "vllm:prefix_cache_queries",
    "vllm:prefix_cache_hits",
    "vllm:time_to_first_token_seconds_sum",
    "vllm:time_to_first_token_seconds_count",
    "vllm:request_prefill_time_seconds_sum",
    "vllm:request_prefill_time_seconds_count",
    "vllm:request_decode_time_seconds_sum",
    "vllm:request_decode_time_seconds_count",
}


def scrape(address):
    stamp = time.monotonic_ns()
    try:
        response = requests.get("http://" + address + "/metrics", timeout=1.5)
        response.raise_for_status()
        values = {}
        for line in response.text.splitlines():
            match = re.match(r"([\w:]+)(?:\{[^}]*\})?\s+([-+\d.eE]+)", line)
            if match and match[1] in NAMES:
                values[match[1]] = values.get(match[1], 0) + float(match[2])
        return {"mono_ns": stamp, "address": address, "metrics": values}
    except Exception as exc:
        return {"mono_ns": stamp, "address": address, "error": str(exc)[:200]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    assert command
    versions = {name: importlib.metadata.version(name) for name in ("torch", "vllm", "ray", "transformers")}
    nv.nvmlInit()
    devices = [nv.nvmlDeviceGetHandleByIndex(i) for i in range(4)]
    versions["gpu"] = [
        {
            "index": i,
            "name": nv.nvmlDeviceGetName(h),
            "uuid": nv.nvmlDeviceGetUUID(h),
            "memory_bytes": nv.nvmlDeviceGetMemoryInfo(h).total,
        }
        for i, h in enumerate(devices)
    ]
    (directory / "versions.json").write_text(json.dumps(versions, indent=2) + "\n")
    started = time.monotonic_ns()
    child = subprocess.Popen(command, env=os.environ | {"VERL_E2E_TIMELINE_DIR": str(directory / "events")})
    stop = threading.Event()

    def monitor():
        addresses = []
        offset = 0
        buffer = ""
        with (
            (directory / "resources.jsonl").open("a", buffering=1) as gpu_file,
            (directory / "engines.jsonl").open("a", buffering=1) as http_file,
            ThreadPoolExecutor(max_workers=4) as pool,
        ):
            while not stop.is_set():
                begin = time.monotonic()
                row = {"mono_ns": time.monotonic_ns(), "wall_ns": time.time_ns(), "gpus": []}
                for i, h in enumerate(devices):
                    mem = nv.nvmlDeviceGetMemoryInfo(h)
                    util = nv.nvmlDeviceGetUtilizationRates(h)
                    row["gpus"].append(
                        {
                            "index": i,
                            "memory_gib": mem.used / 2**30,
                            "busy_percent": util.gpu,
                            "memory_busy_percent": util.memory,
                            "power_w": nv.nvmlDeviceGetPowerUsage(h) / 1000,
                        }
                    )
                try:
                    procs = [psutil.Process(child.pid), *psutil.Process(child.pid).children(recursive=True)]
                    row["sum_process_rss_gib"] = sum(p.memory_info().rss for p in procs if p.is_running()) / 2**30
                    row["process_count"] = len(procs)
                except psutil.Error:
                    pass
                row["host_available_gib"] = psutil.virtual_memory().available / 2**30
                row["shm_used_gib"] = psutil.disk_usage("/dev/shm").used / 2**30
                gpu_file.write(json.dumps(row) + "\n")
                logfile = directory / "stdout.log"
                if logfile.exists():
                    with logfile.open(errors="replace") as stream:
                        stream.seek(offset)
                        buffer += stream.read()
                        offset = stream.tell()
                    matches = re.findall(r"LLMServerManager: (\[[^\n]*?\])", buffer)
                    if matches:
                        addresses = ast.literal_eval(matches[-1])
                    buffer = buffer[-8192:]
                if addresses:
                    for item in pool.map(scrape, addresses):
                        http_file.write(json.dumps(item) + "\n")
                stop.wait(max(0.1, 2 - (time.monotonic() - begin)))

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        code = child.wait()
    finally:
        stop.set()
        thread.join(timeout=8)
        nv.nvmlShutdown()
    (directory / "observation.json").write_text(
        json.dumps(
            {
                "returncode": code,
                "start_ns": started,
                "end_ns": time.monotonic_ns(),
                "note": "Observer RSS sums can include shared pages; all host metrics are descriptive, no CPU or GPU synchronization inserted.",
            },
            indent=2,
        )
        + "\n"
    )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
