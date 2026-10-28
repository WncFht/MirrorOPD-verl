# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Low-overhead NVML and process RSS sampling for a single four-GPU acceptance run."""

import argparse
import json
import time
from pathlib import Path

import psutil
import pynvml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    parent = psutil.Process(args.parent)
    output = Path(args.out)
    pynvml.nvmlInit()
    handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(4)]
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
                            "physical_gpus": [0, 1, 2, 3],
                            "interval_seconds": 0.5,
                        }
                    )
                    + "\n"
                )
            if stream is not None:
                row = {"kind": "sample", "epoch": time.time(), "gpus": []}
                for index, handle in enumerate(handles):
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    row["gpus"].append(
                        {
                            "index": index,
                            "memory_used": pynvml.nvmlDeviceGetMemoryInfo(handle).used,
                            "utilization": util.gpu,
                            "memory_utilization": util.memory,
                        }
                    )
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
