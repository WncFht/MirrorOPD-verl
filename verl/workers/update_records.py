# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Explicit update facts and reduction of worker observations.

Spatial reduction happens here after gathering DP partitions. Update time is
never reduced into an outer mean, and scheduler advancement is not inferred.
"""

import math
from dataclasses import dataclass
from typing import Any

from verl.utils import tensordict_utils as tu
from verl.utils.metric.observations import Observation
from verl.utils.metric.schema import INTERNAL_COUNTERS, canonical_name, is_update_metric
from verl.utils.metric.utils import Metric, reduce_metrics

_MISSING = object()


def execution_consensus(statuses, dp_size):
    if len(statuses) != dp_size or any(value is not None and type(value) is not bool for value in statuses):
        raise RuntimeError(f"Incomplete optimizer execution status across {dp_size} DP ranks: {statuses}")
    if any(value != statuses[0] for value in statuses):
        raise RuntimeError(f"Inconsistent optimizer execution status across DP ranks: {statuses}")
    return statuses[0]


def take_engine_execution(output, metrics):
    """Consume the engine's explicit execution fact; metrics never carry control state."""
    if "_optimizer_step_applied" in metrics:
        raise RuntimeError("Move optimizer execution facts out of metrics into optimizer_step_applied")
    value = output.pop("optimizer_step_applied", _MISSING)
    if value is _MISSING:
        raise RuntimeError("Engine must report optimizer_step_applied (None means explicitly unknown)")
    if value is not None and type(value) is not bool:
        raise RuntimeError(f"Invalid local optimizer execution fact: {value}")
    return value


def merge_rank_metrics(partitions):
    """Preserve the raw per-rank values until reduction semantics are applied."""
    result = {}
    for part in partitions:
        for key, value in part.items():
            result.setdefault(key, []).append(value)
    return result


@dataclass(frozen=True)
class UpdateResult:
    applied: bool | None
    metrics: dict[str, Any]

    @classmethod
    def from_worker_output(cls, output):
        metrics = dict(tu.get(output, "metrics"))
        if "_optimizer_step_applied" in metrics:
            raise RuntimeError("Worker metrics must not carry optimizer execution facts")
        applied = tu.get(output, "optimizer_step_applied", default=_MISSING)
        if applied is _MISSING:
            raise RuntimeError("Worker must report optimizer_step_applied (None means explicitly unknown)")
        return cls(execution_consensus([applied], 1), metrics)


def _flatten(values):
    for value in values:
        if isinstance(value, list | tuple):
            yield from _flatten(value)
        else:
            yield value


def finalize_update(result, *, role, collect_records, collect_metrics, update_index, minibatches_per_epoch):
    """Return outer-scope metrics and one optional pre-update observation record."""
    output = dict(result.metrics)
    for key, val in output.items():
        if isinstance(val, list):
            if val and isinstance(val[0], Observation):
                observation = Observation.merged(val)
                output[key] = None if observation.is_empty() else observation.aggregate()
            elif val and isinstance(val[0], Metric):
                output[key] = Metric.aggregate_dp(val)
            else:
                flattened = list(_flatten(val))
                output[key] = (
                    (reduce_metrics({key: flattened})[key] if flattened else None) if collect_records else flattened
                )
        elif isinstance(val, Observation):
            output[key] = None if val.is_empty() else val.aggregate()
        elif isinstance(val, Metric):
            output[key] = val.aggregate()
    if not collect_records:
        return output, None
    normalized = {}
    for key, value in output.items():
        if collect_metrics and key in INTERNAL_COUNTERS:
            continue
        name = canonical_name(key, role=role)
        if name not in normalized or name == key:
            normalized[name] = value
    record = {k: v for k, v in normalized.items() if is_update_metric(k)}
    record[f"optim/{role}/execution_known"] = float(result.applied is not None)
    if result.applied is not None:
        record[f"optim/{role}/update_applied"] = float(result.applied)
        record[f"optim/{role}/update_skipped"] = float(not result.applied)
    for stat in ("loss", "grad_norm_preclip"):
        value = record.get(f"optim/{role}/{stat}")
        if value is not None:
            record[f"optim/{role}/{stat}_nonfinite"] = float(not math.isfinite(float(value)))
    if role == "actor":
        record["optim/actor/observations_available"] = float(collect_metrics)
        record["rollout/data_reuse/update_index"] = update_index + 1
        record["rollout/data_reuse/epoch_index"] = update_index // minibatches_per_epoch
    outer = {k: v for k, v in normalized.items() if not is_update_metric(k)}
    for key in outer:
        if key.endswith(("allocated_peak_gib", "reserved_peak_gib")):
            outer[key] = Metric("max", outer[key])
    return outer, {"metrics": record, "applied": result.applied}
