# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Review actual whole-cycle work, request boundaries and clock-aligned GPU traces."""

import argparse
import bisect
import gzip
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

from collect import jsonl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from analyze import category, union_length  # noqa: E402
from semantic_trace import attribute  # noqa: E402


def summary(values):
    return dict(mean=st.mean(values), median=st.median(values), min=min(values), max=max(values)) if values else {}


def request_boundaries(job):
    rows = []
    for step in job["steps"]:
        spans = [s for s in job["spans"] if s["start"] >= step["start"] and s["end"] <= step["end"]]
        grouped = defaultdict(list)
        for s in spans:
            grouped[s["name"]].append(s)
        if not all(grouped[n] for n in ("timer/gen", "rollout/request", "rollout/sample")):
            continue
        gen = grouped["timer/gen"][0]
        first = min(s["start"] for s in grouped["rollout/request"])
        last = max(s["end"] for s in grouped["rollout/request"])
        sample_end = max(s["end"] for s in grouped["rollout/sample"])
        rows.append(
            dict(
                step=step["step"],
                start=gen["start"],
                end=gen["end"],
                before_request_s=first - gen["start"],
                request_envelope_s=last - first,
                after_requests_to_samples_s=sample_end - last,
                after_samples_s=gen["end"] - sample_end,
                max_score_rpc_s=max((s["end"] - s["start"] for s in grouped["timer/compute_score"]), default=0),
                output_tokens=sum(s.get("output_tokens", 0) for s in grouped["rollout/request"]),
            )
        )
    return rows


def calibrate(trace, host, origin):
    """Validate the trace wall-clock reconstruction against same-PID semantic spans."""
    annotations = [
        e for e in trace["traceEvents"] if e.get("cat") == "user_annotation" and e["name"] == "full_vocab/backbone"
    ]
    if not annotations:
        return None  # Rollout has no matching semantic host anchor; keep its own clock.
    pid = annotations[0]["pid"]
    same = [s for s in host if s["pid"] == pid]
    offset = int(st.median(s["wall_ns"] - s["mono_ns"] for s in same))
    shift = (trace["baseTimeNanoseconds"] - offset - origin) / 1e9
    starts = sorted((s["start_ns"] - origin) / 1e9 for s in same if s["name"] == "full_vocab/backbone")
    errors = []
    for annotation in annotations:
        t = shift + annotation["ts"] / 1e6
        i = bisect.bisect_left(starts, t)
        errors.append(min(abs(t - starts[k]) for k in (i - 1, i) if 0 <= k < len(starts)))
    result = dict(
        pid=pid,
        shift_seconds=shift,
        anchors=len(errors),
        max_error_ms=max(errors) * 1000,
        median_error_ms=st.median(errors) * 1000,
    )
    if max(errors) > 0.005:
        raise ValueError(f"Trace-to-host calibration exceeds 5 ms: {result}")
    return result


def trace_review(file, job, raw):
    with gzip.open(file, "rt") as stream:
        trace = json.load(stream)
    events = trace["traceEvents"]
    gpu = [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and e.get("dur", 0) > 0]
    result = {"file": str(file), "calibration": calibrate(trace, raw, job["origin_ns"])}
    totals = defaultdict(float)
    for e in gpu:
        kind = category(e["name"]) if e["cat"] == "kernel" else e["cat"]
        if kind == "Fused KL":
            # This study uses precompute; generic *_gradient names are not proof of a fused KL kernel.
            kind = "Other"
        totals[kind] += e["dur"] / 1000
    result["gpu_work_ms"] = dict(totals)
    if "actor-rollout" in file.name:
        result["semantic"] = attribute(events)
    first = min(e["ts"] for e in gpu)
    last = max(e["ts"] + e["dur"] for e in gpu)
    result["gpu_window_seconds"] = (last - first) / 1e6
    intervals = [(e["ts"], e["ts"] + e["dur"]) for e in gpu]
    result["gpu_union_seconds"] = union_length(intervals) / 1e6
    result["phases"] = []
    if result["calibration"]:
        shift = result["calibration"]["shift_seconds"]
        capture = next(e for e in events if e.get("cat") == "Trace")
        capture_start = shift + capture["ts"] / 1e6
        capture_end = shift + (capture["ts"] + capture["dur"]) / 1e6
        result["capture_host_seconds"] = [capture_start, capture_end]
        # Clip GPU intervals to driver phase boundaries, then union; never add overlapping kernels.
        profiled = next(s for s in job["steps"] if s["step"] == 4)
        phases = [
            s
            for s in job["spans"]
            if s["name"]
            in {"timer/teacher_forward", "timer/update_actor", "timer/update_weights", "timer/teacher_transfer"}
            and s["start"] >= profiled["start"]
            and s["end"] <= profiled["end"]
        ]
        for phase in phases:
            if phase["start"] < capture_start or phase["end"] > capture_end:
                result["phases"].append(
                    {
                        "name": phase["name"],
                        "covered": False,
                        "reason": "Outside complete Torch capture window; not zero GPU work",
                    }
                )
                continue
            buckets = defaultdict(list)
            for e in gpu:
                a = max(shift + e["ts"] / 1e6, phase["start"])
                b = min(shift + (e["ts"] + e["dur"]) / 1e6, phase["end"])
                if b <= a:
                    continue
                kind = (
                    "NCCL"
                    if e["cat"] == "kernel" and "nccl" in e["name"].lower()
                    else ("Compute" if e["cat"] == "kernel" else "Memory")
                )
                buckets[kind].append((a, b))
            comp, comm, mem = [union_length(buckets[k]) for k in ("Compute", "NCCL", "Memory")]
            both = union_length(buckets["Compute"] + buckets["NCCL"])
            active = union_length(sum(buckets.values(), []))
            result["phases"].append(
                dict(
                    covered=True,
                    name=phase["name"],
                    start=phase["start"],
                    end=phase["end"],
                    host_s=phase["end"] - phase["start"],
                    compute_s=comp,
                    nccl_s=comm,
                    memory_s=mem,
                    compute_nccl_overlap_s=comp + comm - both,
                    nccl_without_compute_s=both - comp,
                    gpu_active_s=active,
                    no_gpu_event_s=phase["end"] - phase["start"] - active,
                )
            )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.data.read_text())
    baseline = data["jobs"]["baseline"]
    steady = [s for s in baseline["steps"] if 4 <= s["step"] <= 13]
    rows = request_boundaries(baseline)
    phases = {n: summary([s["phases"][n] for s in steady]) for n in steady[0]["phases"]}
    wall = sum(s["seconds"] for s in steady)
    for n, values in phases.items():
        values["fraction_of_total_step_time"] = sum(s["phases"][n] for s in steady) / wall
    updates = [r.get("data", r) for p in (args.root / "baseline").rglob("metrics.jsonl") for r in jsonl(p)]
    updates = [r for r in updates if "optim/actor/update_applied" in r]
    result = dict(
        source=str(args.root),
        steady_steps=[s["step"] for s in steady],
        in_step_seconds=summary([s["seconds"] for s in steady]),
        cycle_seconds=summary([s["start_to_next_start_seconds"] for s in steady]),
        phases=phases,
        startup_seconds=baseline["startup_to_first_step_seconds"],
        rollout_boundaries=rows,
        update_health={
            "count": len(updates),
            "applied": sum(r["optim/actor/update_applied"] for r in updates),
            "skipped": sum(r["optim/actor/update_skipped"] for r in updates),
            "loss_nonfinite": sum(r["optim/actor/loss_nonfinite"] for r in updates),
            "grad_nonfinite": sum(r["optim/actor/grad_norm_preclip_nonfinite"] for r in updates),
            "loss": summary([r["optim/actor/loss"] for r in updates]),
            "grad_norm": summary([r["optim/actor/grad_norm_preclip"] for r in updates]),
        },
        nvml_peak_gib={str(g): max(r["gpus"][g]["memory_gib"] for r in baseline["resources"]) for g in range(4)},
        checkpoint_seconds=next(s["phases"]["save_checkpoint"] for s in baseline["steps"] if s["step"] == 14),
        validation_seconds=sum(s["end"] - s["start"] for s in baseline["spans"] if s["name"] == "timer/testing"),
    )
    rr = [r for r in baseline["request_summary"] if 4 <= r["step"] <= 13]
    result["steady_requests"] = dict(
        tokens=sum(r["tokens"] for r in rr),
        count=sum(r["count"] for r in rr),
        length_finish_fraction=st.mean(r["length_finish_fraction"] for r in rr),
        request_envelope_s=summary([r["t100"] for r in rr]),
        t90_to_t100_ms=summary([(r["t100"] - r["t90"]) * 1000 for r in rr]),
    )
    result["micro_batches"] = [s for s in baseline["spans"] if s["name"] == "engine/micro_batch"]
    result["host_costs_by_step_rank"] = []
    for step in steady:
        grouped = defaultdict(float)
        for s in baseline["spans"]:
            if (
                s["start"] >= step["start"]
                and s["end"] <= step["end"]
                and (s["name"].startswith("full_vocab/") or s["name"].startswith("actor/"))
            ):
                grouped[(s["pid"], s["name"])] += s["end"] - s["start"]
        result["host_costs_by_step_rank"].extend(
            {"step": step["step"], "pid": pid, "name": name, "seconds": seconds}
            for (pid, name), seconds in grouped.items()
        )
    result["traces"] = []
    raw = [r for p in (args.root / "torch-profile/events").glob("*.jsonl") for r in jsonl(p)]
    for file in sorted((args.root / "torch-profile/traces").glob("*.json.gz")):
        result["traces"].append(trace_review(file, data["jobs"]["torch-profile"], raw))
        print(file.name, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
