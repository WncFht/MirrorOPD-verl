# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Update facts and the boundary for legacy worker metric representations.

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


def consume_legacy_status(metrics, dp_size):
    status = metrics.pop("_optimizer_step_applied", None)
    if status is None:
        return None
    return execution_consensus(status if isinstance(status, list) else [status], dp_size)


def take_engine_execution(output, metrics):
    """The only engine-boundary fallback for old metric-carried execution facts."""
    explicit = output.pop("optimizer_step_applied", _MISSING)
    legacy = metrics.pop("_optimizer_step_applied", _MISSING)
    if explicit is not _MISSING and legacy is not _MISSING and explicit != legacy:
        raise RuntimeError("Conflicting engine optimizer execution facts")
    value = explicit if explicit is not _MISSING else (None if legacy is _MISSING else legacy)
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
    def from_worker_output(cls, output, dp_size):
        metrics = dict(tu.get(output, "metrics"))
        explicit = tu.get(output, "optimizer_step_applied", default=_MISSING)
        if explicit is _MISSING:
            applied = consume_legacy_status(metrics, dp_size)
        else:
            applied = execution_consensus([explicit], 1)
            if "_optimizer_step_applied" in metrics and consume_legacy_status(metrics, dp_size) != applied:
                raise RuntimeError("Conflicting worker optimizer execution facts")
        return cls(applied, metrics)


def _flatten(values):
    for value in values:
        if isinstance(value, (list, tuple)):
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
