#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Archive and analyze one policy-v3 run without conflating W&B events and updates.

Raw history is compressed losslessly. A recorded axis map makes subsequent
offline analysis independent of future routing changes in metric/schema.py.
Missing fields stay missing; conflicting observations at one axis value fail.
"""

import argparse
import csv
import gzip
import json
import math
import shutil
import statistics
import sys
from collections import Counter
from pathlib import Path


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def gzip_bytes(path, data):
    path.write_bytes(gzip.compress(data, mtime=0))


def read_rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def finite(value):
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def metric_names(rows):
    return sorted({k for row in rows for k, v in row.items() if finite(v) and not k.startswith(("_", "step/"))})


def normalize(rows, axis_map):
    """Keep each metric on its declared axis, including pre-update observations."""
    observations = {key: {} for key in metric_names(rows)}
    for row in rows:
        for key, value in row.items():
            if key not in observations or value is None:
                continue
            if not finite(value):
                raise ValueError(f"Nonfinite observation: {key}")
            axis = axis_map[key]
            step = row.get(axis)
            if not finite(step):
                raise ValueError(f"Missing axis {axis} for {key}")
            old = observations[key].get(step)
            if old is not None and old != value:
                raise ValueError(f"Conflicting observations: {key} at {axis}={step}")
            observations[key][step] = value
    return {key: {"axis": axis_map[key], "points": sorted(points.items())} for key, points in observations.items()}


def values(series, key):
    return [value for _, value in series[key]["points"]]


def stats(series, key):
    points = series[key]["points"]
    vals = values(series, key)
    return {
        "axis": series[key]["axis"],
        "count": len(vals),
        "constant": len(set(vals)) == 1,
        "first_step": points[0][0],
        "last_step": points[-1][0],
        "first": vals[0],
        "last": vals[-1],
        "min": min(vals),
        "max": max(vals),
        "mean": statistics.mean(vals),
        "median": statistics.median(vals),
        "first20_mean": statistics.mean(vals[:20]),
        "last20_mean": statistics.mean(vals[-20:]),
    }


def timing(series):
    """Outer excludes validation; checkpoint is already inside outer.

    Never sum nested rollout_batch/rollout_pipeline or validation_rollout/validation.
    Regular steps exclude startup and all observed evaluation steps.
    """
    prefix = "perf/timing/"
    outer = dict(series[prefix + "outer_step_seconds"]["points"])
    validation = dict(series[prefix + "validation_seconds"]["points"])
    regular = sorted(set(outer) - set(validation) - {min(outer)})
    stages = [
        "rollout_pipeline",
        "teacher_transfer",
        "teacher_forward",
        "old_log_prob",
        "update_actor",
        "update_weights",
        "adv",
    ]
    means = {}
    for name in stages + ["outer_step"]:
        data = dict(series[prefix + name + "_seconds"]["points"])
        if any(step not in data for step in regular):
            raise ValueError(f"Incomplete regular-step timer: {name}")
        means[name] = statistics.mean(data[step] for step in regular)
    outer_s, val_s = sum(outer.values()), sum(validation.values())
    return {
        "regular_count": len(regular),
        "regular_steps": regular,
        "regular_mean_seconds": means,
        "outer_seconds": outer_s,
        "validation_seconds": val_s,
        "accounted_seconds": outer_s + val_s,
        "validation_fraction": val_s / (outer_s + val_s),
        "checkpoint_seconds_included_in_outer": sum(values(series, prefix + "save_checkpoint_seconds")),
        "validation_rollout_seconds_included_in_validation": sum(values(series, prefix + "validation_rollout_seconds")),
    }


def summarize(rows, series, system_rows, source=None):
    catalog = {key: stats(series, key) for key in series}
    result = {
        "history_events": len(rows),
        "outer_steps": len({r["step/outer"] for r in rows}),
        "last_optimizer_step": max(r["step/optimizer"] for r in rows),
        "numeric_metrics_excluding_axes_and_internal": len(series),
        "families": dict(sorted(Counter(k.split("/")[0] for k in series).items())),
        "statistics": catalog,
        "timing": timing(series),
    }
    counts = dict(series["rollout/response_count"]["points"])
    lengths = dict(series["rollout/finish_reason/length_count"]["points"])
    result["training"] = {
        "responses": sum(counts.values()),
        "response_tokens": sum(values(series, "rollout/response_tokens")),
        "length_stopped_count": sum(lengths.values()),
        "length_stopped_fraction": sum(lengths.values()) / sum(counts.values()),
        "length_stopped_fraction_by_step": [[s, lengths[s] / counts[s]] for s in sorted(counts)],
        "finish_reason_counts": {
            reason: sum(values(series, f"rollout/finish_reason/{reason}_count"))
            for reason in ("length", "stop", "unknown", "aborted", "completed")
        },
        "applied_updates": sum(values(series, "optim/actor/update_applied")),
        "skipped_updates": sum(values(series, "optim/actor/update_skipped")),
        "low_ess_outer_steps": [p for p in series["policy/anchor_behavior/ess_fraction"]["points"] if p[1] < 0.98],
    }
    evaluation = {}
    for dataset in ("math500", "amc23", "aime25"):
        prefix = f"eval/{dataset}/"
        observed = {
            key.removeprefix(prefix): dict(record["points"]) for key, record in series.items() if key.startswith(prefix)
        }
        steps = sorted(observed["acc/mean"])
        selected = [
            "acc/mean",
            "acc/mean_ci_low",
            "acc/mean_ci_high",
            "acc/pass@16",
            "acc/majority@16",
            "response_count",
            "prompt_count",
            "failed_count",
            "acc/prediction_missing_count",
            "generation/response_length/mean",
            "generation/finish_reason/length_count",
        ]
        evaluation[dataset] = [dict(step=step, **{key: observed[key][step] for key in selected}) for step in steps]
    result["evaluation"] = evaluation
    result["evaluation_response_tokens_from_means"] = round(
        sum(
            row["response_count"] * row["generation/response_length/mean"]
            for dataset in evaluation.values()
            for row in dataset
        )
    )
    system = {}
    for key in sorted({k for row in system_rows for k, v in row.items() if k.startswith("system.") and finite(v)}):
        vals = [row[key] for row in system_rows if finite(row.get(key))]
        system[key] = {"count": len(vals), "min": min(vals), "max": max(vals), "sample_mean": statistics.mean(vals)}
    result["system"] = {
        "rows": len(system_rows),
        "complete": bool(
            source and source.get("kind") == "wandb-offline" and source.get("decoder_status") == "finished"
        ),
        "api_sampled": not source or source.get("kind") != "wandb-offline",
        "statistics": system,
        "note": (
            "All decoded periodic records; completeness refers to logs, not continuous monitoring. "
            if source and source.get("kind") == "wandb-offline"
            else "API sampled stream; "
        )
        + "Sample means are not time-weighted full-run means.",
    }
    return result


def snapshot(source, destination):
    if (destination / "history.jsonl.gz").exists():
        raise FileExistsError("Raw snapshot already exists; omit --snapshot-from to recompute analysis")
    destination.mkdir(parents=True, exist_ok=True)
    # Only already-scrubbed exporter outputs; intentionally exclude legacy analysis.json.
    for name in ("config.json", "source.json", "hardware.json", "package-versions.txt"):
        shutil.copyfile(source / name, destination / name)
    system = source / "system.jsonl.gz"
    if not system.is_file():
        system = source / "system-sampled.jsonl.gz"
    shutil.copyfile(system, destination / system.name)
    for name in ("summary.json", "metric-definitions.jsonl"):
        if (source / name).is_file():
            shutil.copyfile(source / name, destination / name)
    gzip_bytes(destination / "history.jsonl.gz", (source / "history.jsonl").read_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--snapshot-from", type=Path)
    args = parser.parse_args()
    if args.snapshot_from:
        snapshot(args.snapshot_from, args.data)
    rows = read_rows(args.data / "history.jsonl.gz")
    axis_path = args.data / "metric-axes.json"
    if axis_path.exists():
        axis_map = json.loads(axis_path.read_text())
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
        from verl.utils.metric.schema import metric_axis

        axis_map = {key: metric_axis(key) for key in metric_names(rows)}
        write_json(axis_path, axis_map)
    series = normalize(rows, axis_map)
    gzip_bytes(args.data / "series.json.gz", json.dumps(series, allow_nan=False).encode())
    system = args.data / "system.jsonl.gz"
    if not system.is_file():
        system = args.data / "system-sampled.jsonl.gz"
    source = json.loads((args.data / "source.json").read_text())
    result = summarize(rows, series, read_rows(system), source)
    write_json(args.data / "analysis.json", result)
    with (args.data / "metric-catalog.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["metric", *next(iter(result["statistics"].values()))])
        writer.writeheader()
        writer.writerows(dict(metric=key, **record) for key, record in result["statistics"].items())
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k
                in (
                    "history_events",
                    "outer_steps",
                    "last_optimizer_step",
                    "numeric_metrics_excluding_axes_and_internal",
                    "families",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
