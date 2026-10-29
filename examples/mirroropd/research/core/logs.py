# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Recorded metric axes and scalar summaries, independent of any report."""

import math
import statistics


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
