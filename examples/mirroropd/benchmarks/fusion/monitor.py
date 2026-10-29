# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Low-overhead NVML and process RSS sampling for one explicitly allocated acceptance run."""

import argparse
import json
import os
import time
from pathlib import Path

import psutil
import pynvml


def optional_nvml(function, *args):
    """Keep unsupported counters optional without hiding device failures."""
    try:
        return function(*args)
    except pynvml.NVMLError as exc:
        if exc.value in {
            pynvml.NVML_ERROR_NOT_SUPPORTED,
            pynvml.NVML_ERROR_NO_PERMISSION,
            pynvml.NVML_ERROR_FUNCTION_NOT_FOUND,
        }:
            return None
        raise


def container_cpu_path():
    """Resolve this container's shared CPU counters, not the host root group."""
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        if line.startswith("0::"):
            path = Path("/sys/fs/cgroup") / line[3:].lstrip("/")
            if (path / "cpu.stat").is_file():
                return path
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    parent = psutil.Process(args.parent)
    output = Path(args.out)
    pynvml.nvmlInit()
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if not devices or any(not item or item == "-1" for item in devices):
        raise ValueError("Monitor requires explicit CUDA_VISIBLE_DEVICES")
    handles = [
        pynvml.nvmlDeviceGetHandleByUUID(item)
        if item.startswith("GPU-")
        else pynvml.nvmlDeviceGetHandleByIndex(int(item))
        for item in devices
    ]
    indices = [pynvml.nvmlDeviceGetIndex(handle) for handle in handles]
    background = []
    for index in range(pynvml.nvmlDeviceGetCount()):
        if index in indices:
            continue
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        background.append(
            {
                "index": index,
                "memory_used": pynvml.nvmlDeviceGetMemoryInfo(handle).used,
                "utilization": pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
            }
        )
    psutil.cpu_percent(interval=None)
    cpu_group = container_cpu_path()

    start = time.time()
    stream = None
    try:
        while parent.is_running() and parent.status() != psutil.STATUS_ZOMBIE:
            # The launcher owns directory creation and refuses existing run directories.
            if stream is None and output.parent.is_dir():
                stream = output.open("x")
                stream.write(
                    json.dumps(
                        {
                            "kind": "metadata",
                            "start_epoch": start,
                            "physical_gpus": indices,
                            "gpu_uuids": [pynvml.nvmlDeviceGetUUID(handle) for handle in handles],
                            "interval_seconds": 0.5,
                            "other_gpu_load_at_start": background,
                            "other_gpu_scope": "read-only context, not allocated to this run",
                            "container_cpu_scope": "Shared by all jobs in this container, not attributed to this run",
                            "container_cpu_quota": (
                                (cpu_group / "cpu.max").read_text().strip()
                                if cpu_group is not None and (cpu_group / "cpu.max").is_file()
                                else None
                            ),
                        }
                    )
                    + "\n"
                )
            if stream is not None:
                row = {
                    "kind": "sample",
                    "epoch": time.time(),
                    "gpus": [],
                    "host_cpu_percent": psutil.cpu_percent(interval=None),
                    "host_memory_available": psutil.virtual_memory().available,
                }
                for index, handle in zip(indices, handles, strict=True):
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    row["gpus"].append(
                        {
                            "index": index,
                            "memory_used": pynvml.nvmlDeviceGetMemoryInfo(handle).used,
                            "utilization": util.gpu,
                            "memory_utilization": util.memory,
                            "sm_clock_mhz": optional_nvml(pynvml.nvmlDeviceGetClockInfo, handle, pynvml.NVML_CLOCK_SM),
                            "memory_clock_mhz": optional_nvml(
                                pynvml.nvmlDeviceGetClockInfo, handle, pynvml.NVML_CLOCK_MEM
                            ),
                            "power_mw": optional_nvml(pynvml.nvmlDeviceGetPowerUsage, handle),
                            "pcie_tx_kb_per_second": optional_nvml(
                                pynvml.nvmlDeviceGetPcieThroughput, handle, pynvml.NVML_PCIE_UTIL_TX_BYTES
                            ),
                            "pcie_rx_kb_per_second": optional_nvml(
                                pynvml.nvmlDeviceGetPcieThroughput, handle, pynvml.NVML_PCIE_UTIL_RX_BYTES
                            ),
                        }
                    )
                if cpu_group is not None:
                    row["shared_container_cpu"] = {
                        key: int(value)
                        for key, value in (line.split() for line in (cpu_group / "cpu.stat").read_text().splitlines())
                    }
                processes = [parent] + parent.children(recursive=True)
                rss = 0
                for process in processes:
                    try:
                        rss += process.memory_info().rss
                    except psutil.Error:
                        pass
                row["process_tree_rss_sum"] = rss
                stream.write(json.dumps(row) + "\n")
                stream.flush()
            time.sleep(0.5)
    except psutil.NoSuchProcess:
        pass
    finally:
        if stream is not None:
            stream.write(json.dumps({"kind": "end", "epoch": time.time(), "wall_seconds": time.time() - start}) + "\n")
            stream.close()
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
