# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Replay documented runtimes using adjacent archived measurements, without checkpoints."""

import argparse
import gzip
import json
import math
from pathlib import Path


def schedule(values, workers=8):
    loads = [0.0] * workers
    for value in sorted(values, reverse=True):
        loads[min(range(workers), key=loads.__getitem__)] += value
    return dict(work_seconds=sum(values), makespan_seconds=max(loads), lane_seconds=loads)


def replay(data):
    data = Path(data)
    with gzip.open(data / "scan-benchmark-20261111.json.gz", "rt") as stream:
        scan = json.load(stream)
    with gzip.open(data / "costs.json.gz", "rt") as stream:
        historical = json.load(stream)
    result = {}
    for name, workload in scan["workloads"].items():
        loads = [0.0] * 8
        for task in workload["tasks"]:
            loads[task["shard"]] += task["seconds"]
        result[name] = dict(serial_seconds=sum(loads), strided_seconds=max(loads), lane_seconds=loads)
        assert math.isclose(sum(loads), workload["serial_seconds"], rel_tol=1e-10)
    result["historical_weights"] = schedule([row["seconds"] for row in historical["weight_measurements"]])
    assert math.isclose(
        result["historical_weights"]["makespan_seconds"],
        historical["weights_history_schedule"]["makespan_seconds"],
        rel_tol=1e-10,
    )
    result["direction_whole_conditions"] = schedule(
        [row["measured_compute_seconds"] for row in historical["measured_direction_preparation"]]
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parent.parent)
    print(json.dumps(replay(parser.parse_args().data), indent=2))
