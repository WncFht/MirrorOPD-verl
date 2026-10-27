#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Replay recorded request lengths to estimate KV pressure, not throughput.

Input: JSON (optionally gzipped), mapping outer-step strings to request lists.
Each request has replica, input, length, start (ordering), and cap fields.
All decode steps have equal duration; no prefill cost, cache sharing, eviction,
or recomputation is modeled. Output ranges are candidate settings, not CIs.
"""

import argparse
import collections
import gzip
import heapq
import json
from pathlib import Path

CANDIDATES = [16, 24, 32, 40, 48, 56, 64, 80, 96, 128, 160, 192, 256]


def simulate(jobs, concurrency, budget):
    queue = iter(sorted(jobs, key=lambda item: item["start"]))
    heap = []
    offset = now = peak = area = above = ticks = active_ticks = 0

    def fill(t, current_offset):
        while len(heap) < concurrency:
            job = next(queue, None)
            if job is None:
                break
            length = max(1, job["length"])
            entry_offset = job["input"] + 8 - t
            heapq.heappush(heap, (t + length, entry_offset))
            current_offset += entry_offset
        return current_offset

    offset = fill(now, offset)
    while heap:
        end = heap[0][0]
        count = len(heap)
        duration = end - now
        low = count * now + offset
        high = count * end + offset
        peak = max(peak, high)
        area += (low + high) * duration / 2
        ticks += duration
        active_ticks += count * duration
        above += max(0, min(duration, end - (budget - offset) / count))
        now = end
        while heap and heap[0][0] == end:
            _, entry_offset = heapq.heappop(heap)
            offset -= entry_offset
        offset = fill(now, offset)
    return dict(peak=peak, above=above, ticks=ticks, area=area, n_weight=active_ticks)


def estimate(requests, capacity):
    summary, simulations = [], []
    for step, jobs in sorted(((int(k), v) for k, v in requests.items())):
        total = sum(job["length"] for job in jobs)
        mean = total / len(jobs)
        effective = (
            sum(job["input"] * job["length"] + job["length"] * (job["length"] - 1) / 2 for job in jobs) / total + 8
        )
        capped = [job for job in jobs if job["cap"]]
        summary.append(
            dict(
                step=step,
                responses=len(jobs),
                mean_length=mean,
                cap_fraction=len(capped) / len(jobs),
                cap_token_share=sum(j["length"] for j in capped) / total,
                cap_residency_share=sum(j["length"] ** 2 for j in capped) / sum(j["length"] ** 2 for j in jobs),
                effective_live_kv_tokens=effective,
                mean_capacity_at90pct=0.9 * capacity / effective,
                naive_capacity_at90pct=0.9 * capacity / mean,
            )
        )
        groups = collections.defaultdict(list)
        for job in jobs:
            groups[job["replica"]].append(job)
        for concurrency in CANDIDATES:
            runs = [simulate(group, concurrency, 0.9 * capacity) for group in groups.values()]
            ticks = sum(run["ticks"] for run in runs)
            simulations.append(
                dict(
                    step=step,
                    concurrency=concurrency,
                    pressure_fraction=sum(run["above"] for run in runs) / ticks,
                    peak_tokens=max(run["peak"] for run in runs),
                    mean_resident_tokens=sum(run["area"] for run in runs) / ticks,
                    mean_running=sum(run["n_weight"] for run in runs) / ticks,
                )
            )
    reference = next(s["pressure_fraction"] for s in simulations if s["step"] == 0 and s["concurrency"] == 24)
    for row in summary:
        options = [s for s in simulations if s["step"] == row["step"]]
        row["low_pressure_candidate"] = max(
            (s["concurrency"] for s in options if s["pressure_fraction"] <= 0.05), default=None
        )
        row["reference_pressure_candidate"] = max(
            (s["concurrency"] for s in options if s["pressure_fraction"] <= reference), default=None
        )
    return dict(
        capacity_reference_tokens=capacity,
        reserve_fraction=0.1,
        calibration_reference_pressure=reference,
        summary=summary,
        simulations=simulations,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-lengths", type=Path, required=True)
    parser.add_argument("--capacity-tokens", type=int, default=554208)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    opener = gzip.open if args.request_lengths.suffix == ".gz" else open
    with opener(args.request_lengths, "rt") as handle:
        requests = json.load(handle)
    result = estimate(requests, args.capacity_tokens)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    for row in result["summary"]:
        print(
            row["step"],
            round(row["mean_length"]),
            round(row["effective_live_kv_tokens"]),
            row["low_pressure_candidate"],
            row["reference_pressure_candidate"],
        )


if __name__ == "__main__":
    main()
