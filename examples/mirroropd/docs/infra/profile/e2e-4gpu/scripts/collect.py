# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Build a measured, coverage-aware report dataset; never impute missing events."""

import argparse
import gzip
import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml


def jsonl(path):
    if not path.exists():
        return []
    rows = []
    content = path.read_text()
    lines = content.splitlines()
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except ValueError:
            if i == len(lines) - 1 and not content.endswith("\n"):
                continue  # A running writer may have an incomplete last line.
            raise ValueError(f"Invalid complete JSONL record: {path}:{i + 1}") from None
    return rows


def quantile(values, q):
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * q
    lo = int(index)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def union_seconds(rows):
    total = 0
    end = None
    for start, stop in sorted(rows):
        total += stop - start if end is None or start >= end else max(0, stop - end)
        end = stop if end is None else max(end, stop)
    return total / 1e9


def collect_job(path):
    result = {
        "name": path.name,
        "source": str(path),
        "status": (
            json.loads((path / "exit.json").read_text())
            if (path / "exit.json").exists()
            else {"state": "started" if (path / "launch.json").exists() else "not_started"}
        ),
    }
    result["config"] = (
        yaml.safe_load((path / "resolved_config.yaml").read_text()) if (path / "resolved_config.yaml").exists() else {}
    )
    observation = json.loads((path / "observation.json").read_text()) if (path / "observation.json").exists() else {}
    result["versions"] = json.loads((path / "versions.json").read_text()) if (path / "versions.json").exists() else None
    raw = [x for p in sorted((path / "events").glob("*.jsonl")) for x in jsonl(p)]
    domains = sorted({(x["host"], x["clock_domain"]) for x in raw})
    assert len(domains) <= 1, "Cross-clock-domain spans must not be merged"
    result["resources"] = jsonl(path / "resources.jsonl")
    anchors = [x["start_ns"] for x in raw] + [x["mono_ns"] for x in result["resources"]]
    origin = observation.get("start_ns", min(anchors, default=0))
    result["origin_kind"] = "command_start" if "start_ns" in observation else "first_observation"
    result["clock_domains"] = domains
    result["origin_ns"] = origin
    metric_files = list(path.rglob("metrics.jsonl"))
    metrics = [x.get("data", x) for p in metric_files for x in jsonl(p)]
    # Update-coordinate records can coexist with outer-step records.
    result["metrics"] = [m for m in metrics if "perf/timing/outer_step_seconds" in m]
    steps = sorted([x for x in raw if x["name"] == "timer/step"], key=lambda x: x["start_ns"])
    validation = [x for x in raw if x["name"] == "timer/testing"]
    result["steps"] = []
    top_names = [
        "gen",
        "reward",
        "teacher_transfer",
        "teacher_forward",
        "update_actor",
        "update_weights",
        "save_checkpoint",
    ]
    for i, step in enumerate(steps, 1):
        child = [
            x
            for x in raw
            if x["pid"] == step["pid"]
            and x["name"] in {"timer/" + n for n in top_names}
            and x["start_ns"] >= step["start_ns"]
            and x["end_ns"] <= step["end_ns"]
        ]
        phases = {
            n: sum((x["end_ns"] - x["start_ns"]) / 1e9 for x in child if x["name"] == "timer/" + n) for n in top_names
        }
        duration = (step["end_ns"] - step["start_ns"]) / 1e9
        covered = union_seconds([(x["start_ns"], x["end_ns"]) for x in child])
        result["steps"].append(
            {
                "step": i,
                "start": (step["start_ns"] - origin) / 1e9,
                "end": (step["end_ns"] - origin) / 1e9,
                "seconds": duration,
                "phases": phases,
                "unattributed_seconds": max(0, duration - covered),
                "phase_overlap_seconds": max(0, sum(phases.values()) - covered),
                "start_to_next_start_seconds": (
                    (steps[i]["start_ns"] - step["start_ns"]) / 1e9 if i < len(steps) else None
                ),
            }
        )
    requests = []
    for event in raw:
        event["start"] = (event["start_ns"] - origin) / 1e9
        event["end"] = (event["end_ns"] - origin) / 1e9
        if event["name"] == "rollout/request":
            isval = any(v["start_ns"] <= event["end_ns"] <= v["end_ns"] for v in validation)
            step = next((i for i, s in enumerate(steps, 1) if s["start_ns"] <= event["end_ns"] <= s["end_ns"]), None)
            requests.append(
                {
                    k: event.get(k)
                    for k in (
                        "pid",
                        "replica",
                        "request_id",
                        "input_tokens",
                        "output_tokens",
                        "finish_reason",
                        "start",
                        "end",
                    )
                }
                | {"step": step, "validation": isval, "status": event.get("status", "ok")}
            )
    result["requests"] = requests
    result["spans"] = [
        {
            k: v
            for k, v in x.items()
            if k not in ("start_ns", "end_ns", "mono_ns", "wall_ns", "host", "clock_domain", "cuda_visible")
        }
        for x in raw
        if x["name"] != "rollout/backend_metrics"
    ]
    result["backend_metrics"] = [x for x in raw if x["name"] == "rollout/backend_metrics"]
    result["engines"] = jsonl(path / "engines.jsonl")
    for row in result["resources"] + result["engines"]:
        row["seconds"] = (row["mono_ns"] - origin) / 1e9
    completed = [x for x in requests if x["status"] == "ok" and x["output_tokens"] is not None]
    result["coverage"] = {
        "events": len(raw),
        "processes": len({x["pid"] for x in raw}),
        "requests": len(completed),
        "observed_requests": len(requests),
        "failed_requests": sum(x["status"] == "error" for x in requests),
        "missing_request_output": sum(x["output_tokens"] is None for x in requests),
        "step_requests": dict(Counter(x["step"] for x in completed if x["step"] is not None)),
        "validation_requests": sum(x["validation"] for x in completed),
        "event_names": dict(Counter(x["name"] for x in raw)),
        "backend_timing_present": sum(
            bool(x.get("backend_times", {}).get("first_token_time")) for x in result["backend_metrics"]
        ),
        "engine_samples": sum(bool(x.get("metrics")) for x in result["engines"]),
    }
    result["request_summary"] = []
    for step in result["steps"]:
        rr = [x for x in completed if x["step"] == step["step"] and not x["validation"]]
        if not rr:
            continue
        start = min(x["start"] for x in rr)
        finish = [x["end"] - start for x in rr]
        result["request_summary"].append(
            {
                "step": step["step"],
                "count": len(rr),
                "tokens": sum(x.get("output_tokens") or 0 for x in rr),
                "t50": quantile(finish, 0.5),
                "t90": quantile(finish, 0.9),
                "t99": quantile(finish, 0.99),
                "t100": max(finish),
                "last10_fraction": (max(finish) - quantile(finish, 0.9)) / max(finish) if max(finish) else None,
                "length_finish_fraction": sum(x.get("finish_reason") == "length" for x in rr) / len(rr),
                "mean_length": statistics.mean(x.get("output_tokens") or 0 for x in rr),
            }
        )
    # No sampled/estimated GPU trace counts: count all events in each saved trace.
    traces = []
    for file in sorted((path / "traces").rglob("*.json*")):
        try:
            opener = gzip.open if file.suffix == ".gz" else open
            with opener(file, "rt") as stream:
                trace = json.load(stream)
            events = trace.get("traceEvents", [])
            cats = Counter(x.get("cat") for x in events)
            kernels = [x for x in events if x.get("cat") == "kernel"]
            names = Counter(x.get("name") for x in events if x.get("cat") == "user_annotation")
            from_root = {k: v for k, v in trace.items() if k != "traceEvents" and isinstance(v, (int, float, str))}
            traces.append(
                {
                    "path": str(file),
                    "bytes": file.stat().st_size,
                    "categories": dict(cats),
                    "kernel_work_ms": sum(x.get("dur", 0) for x in kernels) / 1000,
                    "metadata": from_root,
                    "semantic_ranges": {k: v for k, v in names.items() if k.startswith("full_vocab/")},
                }
            )
        except (ValueError, OSError) as exc:
            traces.append({"path": str(file), "error": str(exc)[:200]})
    result["traces"] = traces
    result["nsys_files"] = [str(p) for p in (path / "traces").rglob("*.nsys-rep")]
    result["nsys_review"] = (
        json.loads((path / "nsys-review.json").read_text()) if (path / "nsys-review.json").exists() else None
    )
    if result["steps"]:
        result["startup_to_first_step_seconds"] = result["steps"][0]["start"]
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--nsys-retry-root", type=Path)
    args = p.parse_args()
    data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_status": json.loads((args.root / "status.json").read_text()),
        "scope": (
            "Single-turn math, four H800, E13 precompute with graph rollout. "
            "Host spans are elapsed host intervals, not exclusive GPU kernel duration."
        ),
        "jobs": {
            name: collect_job(args.root / name)
            for name in ("baseline", "torch-profile", "nsys-profile")
            if (args.root / name / "resolved_config.yaml").exists()
        },
    }
    if args.nsys_retry_root:
        data["jobs"]["nsys-retry"] = collect_job(args.nsys_retry_root / "nsys-profile")
        data["previous_run_status"] = data["run_status"]
        data["run_status"] = json.loads((args.nsys_retry_root / "status.json").read_text())
        data["run_status"]["current_job"] = "nsys-retry"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n")
    print({k: v["coverage"] for k, v in data["jobs"].items()})


if __name__ == "__main__":
    main()
