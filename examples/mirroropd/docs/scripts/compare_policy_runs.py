#!/usr/bin/env python3
"""Compare explicit policy-v3 runs using recorded axes and complete outer-step boundaries.

No cloud calls or training. Archive scrubbed exports once, then derive every metric's
coverage/statistics. Evaluation uses actual checkpoints without interpolation.
Update losses are pre-optimizer observations even when logged after the update.
"""

import argparse
import csv
import gzip
import json
import math
import re
import shutil
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

from analyze_policy_run import finite, metric_names, normalize, read_rows, stats, write_json


def get(config, path, default=None):
    for key in path.split("."):
        if not isinstance(config, dict) or key not in config:
            return default
        config = config[key]
    return config


def quantile(values, fraction):
    a = sorted(values)
    if not a:
        return None
    i = (len(a) - 1) * fraction
    lo, hi = math.floor(i), math.ceil(i)
    return a[lo] + (a[hi] - a[lo]) * (i - lo)


def describe(values):
    a = [v for v in values if finite(v)]
    if not a:
        return {"count": 0}
    return {
        "count": len(a),
        "mean": st.mean(a),
        "median": st.median(a),
        "min": min(a),
        "max": max(a),
        "p10": quantile(a, 0.1),
        "p90": quantile(a, 0.9),
    }


def evaluation_detail(key):
    """Keep report fields compact; every other metric stays in the full series/catalog."""
    if not key.startswith("eval/"):
        return False
    parts = key.split("/", 2)
    if len(parts) != 3:
        return False
    metric = parts[2]
    return (
        not metric.startswith("acc/")
        or metric in {"acc/mean", "acc/mean_ci_low", "acc/mean_ci_high", "acc/prediction_missing_count"}
        or re.fullmatch(r"acc/(pass|majority|all_correct)@\d+", metric) is not None
    )


def flatten(config, prefix=""):
    result = {}
    for key, value in config.items():
        path = prefix + key
        if isinstance(value, dict) and value:
            result.update(flatten(value, path + "."))
        else:
            result[path] = value
    return result


def archive(source, destination):
    if destination.exists():
        raise FileExistsError(f"Never overwrite an archived run: {destination}")
    destination.mkdir(parents=True)
    for name in ("config.json", "summary.json", "hardware.json", "source.json", "package-versions.txt"):
        shutil.copyfile(source / name, destination / name)
    for name in ("history.jsonl", "metric-definitions.jsonl"):
        (destination / (name + ".gz")).write_bytes(gzip.compress((source / name).read_bytes(), mtime=0))
    shutil.copyfile(source / "system.jsonl.gz", destination / "system.jsonl.gz")


def recorded_axes(rows, definitions):
    axes = {}
    for record in definitions:
        name, axis = record.get("name"), record.get("step_metric")
        if name and axis:
            if name in axes and axes[name] != axis:
                raise ValueError(f"Axis changed within run: {name}")
            axes[name] = axis
    missing = set(metric_names(rows)) - axes.keys()
    if missing:
        raise ValueError(f"Metrics lack recorded axes: {sorted(missing)}")
    return {key: axes[key] for key in metric_names(rows)}


def cost(outer, through):
    selected = [r for r in outer if r["step/outer"] <= through]
    # First step, checkpoint and validation steps are excluded from steady summaries.
    regular = [
        r
        for r in selected
        if r["step/outer"] > 1
        and "perf/timing/validation_seconds" not in r
        and "perf/timing/save_checkpoint_seconds" not in r
    ]
    keys = sorted({k for r in selected for k in r if k.startswith("perf/timing/")})
    sums = {k: sum(r[k] for r in selected if finite(r.get(k))) for k in keys}
    regular_stats = {k: describe([r[k] for r in regular if finite(r.get(k))]) for k in keys}
    training = sums.get("perf/timing/outer_step_seconds", 0)
    validation = sums.get("perf/timing/validation_seconds", 0)
    return {
        "outer_records": len(selected),
        "regular_records": len(regular),
        "sums": sums,
        "regular": regular_stats,
        "accounted_hours": (training + validation) / 3600,
        "training_hours": training / 3600,
        "validation_hours": validation / 3600,
        "validation_fraction": validation / (training + validation) if training + validation else None,
    }


def reuse(updates, epochs):
    groups = defaultdict(list)
    for row in updates:
        groups[row["step/outer"]].append(row)
    batches = []
    for step, group in sorted(groups.items()):
        group.sort(key=lambda r: r["step/update"])
        if len(group) != epochs:
            continue
        indices = [r["rollout/data_reuse/update_index"] for r in group]
        if indices != list(range(1, epochs + 1)):
            raise ValueError(f"Unexpected mini-batch/update layout: {step}: {indices}")
        if "distill/full_kl" not in group[0]:
            continue
        values = [r["distill/full_kl"] for r in group]
        batches.append(
            {
                "outer": step,
                "first": values[0],
                "last": values[-1],
                "values": values,
                "first_update": group[0]["step/update"],
                "last_update": group[-1]["step/update"],
                "relative_drop": 1 - values[-1] / values[0],
                "increases": sum(b > a for a, b in zip(values[:-1], values[1:], strict=True)),
            }
        )
    return batches


def telemetry(system, outer, gpu_count):
    times = sorted({r["_timestamp"] for r in system})
    gaps = [b - a for a, b in zip(times[:-1], times[1:], strict=True)]
    gpu_times = sorted(
        {r["_timestamp"] for r in system if all(finite(r.get(f"system.gpu.{i}.gpu")) for i in range(gpu_count))}
    )
    windows = []
    for row in outer:
        # Approximate host boundaries inferred from logged wall timers, not CUDA traces.
        end = row["_timestamp"]
        validation = row.get("perf/timing/validation_seconds", 0)
        train_end = end - validation
        windows.append((train_end - row["perf/timing/outer_step_seconds"], train_end, "train"))
        if validation:
            windows.append((train_end, end, "validation"))
    windows.sort()
    outputs = {}
    for trim in (0, 10):
        groups = {"train": [], "validation": []}
        wi = 0
        for row in sorted(system, key=lambda r: r["_timestamp"]):
            t = row["_timestamp"]
            while wi < len(windows) and windows[wi][1] < t:
                wi += 1
            if wi < len(windows):
                start, end, phase = windows[wi]
                if start + trim <= t <= end - trim:
                    groups[phase].append(row)
        for phase, rows in groups.items():
            out = {"sample_times": len(rows)}
            for suffix in ("gpu", "memory", "memoryAllocatedBytes", "powerWatts", "temp"):
                samples = []
                for row in rows:
                    vals = [row.get(f"system.gpu.{i}.{suffix}") for i in range(gpu_count)]
                    if all(finite(v) for v in vals):
                        samples.append(st.mean(vals) / (2**30 if suffix == "memoryAllocatedBytes" else 1))
                out[suffix] = describe(samples)
            outputs[f"{phase}_trim{trim}"] = out
    memory = [
        v / 2**30
        for r in system
        for k, v in r.items()
        if k.startswith("system.gpu.") and k.endswith(".memoryAllocatedBytes") and finite(v)
    ]
    return {
        "rows": len(system),
        "sample_gap_seconds": describe(gaps),
        "complete_gpu_sample_gap_seconds": describe(
            [b - a for a, b in zip(gpu_times[:-1], gpu_times[1:], strict=True)]
        ),
        "phases": outputs,
        "nvml_sampled_max_gib": max(memory) if memory else None,
        "note": "All decoded periodic samples; phase bounds approximate; sample means are not time-weighted.",
    }


def analyze(path, label):
    config = json.loads((path / "config.json").read_text())
    source = json.loads((path / "source.json").read_text())
    rows = read_rows(path / "history.jsonl.gz")
    definitions = read_rows(path / "metric-definitions.jsonl.gz")
    axes = recorded_axes(rows, definitions)
    series = normalize(rows, axes)
    updates = [r for r in rows if "optim/actor/update_applied" in r]
    outer = [r for r in rows if "perf/timing/outer_step_seconds" in r]
    evals = [r for r in rows if "eval/math500/acc/mean" in r]
    epochs = get(config, "actor_rollout_ref.actor.ppo_epochs")
    full = get(config, "distillation.distillation_loss.loss_mode") == "full_reverse_kl"
    if any(r["optim/actor/update_applied"] != 1 or r.get("optim/actor/update_skipped") != 0 for r in updates):
        raise ValueError("This comparison requires a separate treatment of skipped/unknown updates")
    update_steps = [r["step/update"] for r in updates]
    if update_steps != list(range(1, len(updates) + 1)):
        raise ValueError("Update coordinate is incomplete or duplicated")
    outer_steps = [r["step/outer"] for r in outer]
    if outer_steps != list(range(1, len(outer) + 1)):
        raise ValueError("Complete outer records are not contiguous")
    if any(r["step/optimizer"] != r["step/update"] for r in updates):
        raise ValueError("Attempt and optimizer coordinates differ")
    catalog = {k: stats(series, k) for k in series}
    counts = Counter(k.split("/")[0] for k in series)
    flags = {k: v["max"] for k, v in catalog.items() if "nonfinite" in k or k.endswith("/undefined")}
    result = {
        "label": label,
        "id": path.name,
        "source": source,
        "hardware": json.loads((path / "hardware.json").read_text()),
        "configuration": {
            k: get(config, k)
            for k in (
                "trainer.experiment_name",
                "trainer.total_training_steps",
                "trainer.n_gpus_per_node",
                "data.train_batch_size",
                "data.seed",
                "actor_rollout_ref.actor.ppo_epochs",
                "actor_rollout_ref.actor.optim.lr",
                "actor_rollout_ref.actor.optim.clip_grad",
                "distillation.distillation_loss.loss_mode",
                "distillation.distillation_loss.policy_loss_mode",
                "distillation.distillation_loss.full_vocab.backend",
                "trainer.test_freq",
                "trainer.save_freq",
            )
        },
        "coverage": {
            "history_records": len(rows),
            "numeric_metrics": len(series),
            "families": dict(sorted(counts.items())),
            "constant_metrics": sum(v["constant"] for v in catalog.values()),
            "complete_outer_steps": len(outer),
            "max_observed_outer": max(r["step/outer"] for r in rows),
            "updates": len(updates),
            "last_evaluation_outer": evals[-1]["step/outer"],
            "evaluations": len(evals),
            "event_flags": flags,
            "updates_without_outer_record": [r["step/update"] for r in updates if r["step/outer"] not in outer_steps],
        },
        "time": cost(outer, len(outer)),
        "training": {},
        "evaluation": [],
        "reuse": reuse(updates, epochs),
        "selected_metrics": {
            k: v
            for k, v in catalog.items()
            if k.startswith(("optim/", "distill/", "policy/", "advantage/", "correction/", "perf/"))
        },
        "system": telemetry(read_rows(path / "system.jsonl.gz"), outer, get(config, "trainer.n_gpus_per_node")),
    }
    for through in sorted({160, len(outer)}):
        selected = [r for r in outer if r["step/outer"] <= through]
        responses = sum(r["rollout/response_count"] for r in selected)
        stopped = sum(r["rollout/finish_reason/length_count"] for r in selected)
        result["training"][str(through)] = {
            "outer_steps": len(selected),
            "responses": responses,
            "response_tokens": sum(r["rollout/response_tokens"] for r in selected),
            "mean_score": st.mean(r["rollout/score/mean"] for r in selected),
            "last20_score": st.mean(r["rollout/score/mean"] for r in selected[-20:]),
            "last20_length": st.mean(r["rollout/response_length/mean"] for r in selected[-20:]),
            "length_stopped_fraction": stopped / responses,
            "time": cost(outer, through),
        }
    clipped = [
        r
        for r in updates
        if r["optim/actor/grad_norm_preclip"] > get(config, "actor_rollout_ref.actor.optim.clip_grad")
    ]
    result["gradient_clip"] = {
        "count": len(clipped),
        "fraction": len(clipped) / len(updates),
        "last_update": max((r["step/update"] for r in clipped), default=None),
    }
    for row in evals:
        entry = {
            "outer": row["step/outer"],
            "optimizer": row["step/optimizer"],
            "accounted_hours": cost(outer, row["step/outer"])["accounted_hours"],
            "metrics": {k: v for k, v in row.items() if evaluation_detail(k) and finite(v)},
        }
        result["evaluation"].append(entry)
    if full:
        result["fresh_vs_last"] = {}
        for through in sorted({160, max(r["step/outer"] for r in updates)}):
            tail = [b for b in result["reuse"] if b["outer"] <= through][-20:]
            result["fresh_vs_last"][str(through)] = {
                "batches": len(tail),
                "first_mean": st.mean(b["first"] for b in tail),
                "last_mean": st.mean(b["last"] for b in tail),
                "relative_drop_mean": st.mean(b["relative_drop"] for b in tail),
                "within_batch_increases": sum(b["increases"] for b in tail),
            }
    coordinate = {"step/update": {}, "step/optimizer": {}, "step/outer": {}}
    for row in rows:
        for axis in coordinate:
            if axis in row:
                coordinate[axis][row[axis]] = row["step/outer"]
    return result, series, catalog, flatten(config), coordinate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", required=True, help="Explicit LABEL=RUN_ID entries")
    parser.add_argument("--snapshot-from", type=Path)
    args = parser.parse_args()
    runs = dict(value.split("=", 1) for value in args.runs)
    if args.snapshot_from:
        for run in runs.values():
            archive(args.snapshot_from / run, args.data / run)
    results, all_series, catalogs, configs, coordinates = {}, {}, {}, {}, {}
    for label, run in runs.items():
        results[label], all_series[label], catalogs[label], configs[label], coordinates[label] = analyze(
            args.data / run, label
        )
        print(label, results[label]["coverage"])
    common_eval = sorted(set.intersection(*[{r["outer"] for r in v["evaluation"]} for v in results.values()]))
    common = common_eval[-1]
    common_update = sorted(set.intersection(*[{r["optimizer"] for r in v["evaluation"]} for v in results.values()]))
    shared = set.intersection(*(set(s) for s in all_series.values()))
    union = set.union(*(set(s) for s in all_series.values()))
    result = {
        "runs": results,
        "common_last_evaluation_outer": common,
        "common_evaluation_optimizer_steps": common_update,
        "common_numeric_metrics": len(shared),
        "union_numeric_metrics": len(union),
    }
    write_json(args.data / "analysis.json", result)
    differences = {
        k: {label: c.get(k) for label, c in configs.items()}
        for k in sorted(set.union(*(set(c) for c in configs.values())))
        if len({json.dumps(c.get(k), sort_keys=True) for c in configs.values()}) > 1
    }
    write_json(args.data / "config-differences.json", differences)
    bundle = {"analysis": result, "series": all_series, "coordinates": coordinates}
    (args.data / "comparison.json.gz").write_bytes(gzip.compress(json.dumps(bundle, allow_nan=False).encode(), mtime=0))
    with (args.data / "metric-catalog.csv").open("w", newline="") as stream:
        fields = ["run", "metric", *next(iter(next(iter(catalogs.values())).values())).keys()]
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for label, catalog in catalogs.items():
            for key, record in catalog.items():
                writer.writerow({"run": label, "metric": key, **record})
    with (args.data / "metric-coverage.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["metric", *runs])
        for key in sorted(union):
            writer.writerow([key, *[catalogs[label].get(key, {}).get("count", "missing") for label in runs]])


if __name__ == "__main__":
    main()
