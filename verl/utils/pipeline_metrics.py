# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Training pipeline telemetry helpers; no worker logger or scheduling state."""

import logging
import math
import os
import time
from functools import cache, lru_cache
from uuid import uuid4

FSDP_PEAK_MEMORY = "memory/fsdp_worker_peak_allocated_gib_max"
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def monotonic_clock_domain() -> str | None:
    """Identify a Linux clock domain, not just a hostname (time namespaces may differ)."""
    import ray

    if not ray.is_initialized():
        return None
    try:
        namespace = os.readlink("/proc/self/ns/time")
    except OSError:
        return None  # Do not compare clocks when the domain cannot be verified.
    return f"{ray.get_runtime_context().get_node_id()}:{namespace}"


def make_rollout_submission(expected_requests: int) -> dict:
    """Call immediately before dispatch. The ID prevents mixing different submissions."""
    domain = monotonic_clock_domain()
    return {
        "id": uuid4().hex,
        "expected_requests": int(expected_requests),
        "clock_domain": domain,
        "started_at_ns": time.monotonic_ns(),
    }


@cache
def _warn_once(reason: str):
    logger.warning("Pipeline metrics omitted: %s", reason)


def _nonnegative(value) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def aggregate_pipeline_requests(keys: list[str], rows: list[dict]) -> dict[str, float]:
    """Reduce real trajectories, excluding padding before calling this helper.

    A single-turn submission must be complete and contain one output per session.
    Failure, cancellation, filtering and a sampled subset cannot pass this check.
    Teacher latency remains a request mean, never a sum of overlapping RPCs.
    """
    if not rows or len(keys) != len(rows) or not all(isinstance(row, dict) for row in rows):
        _warn_once("missing request telemetry")
        return {}
    metrics = {}
    latencies = [row.get("teacher_request_latency_s") for row in rows]
    if all(_nonnegative(value) for value in latencies):
        metrics["teacher/request_latency_s/mean"] = sum(latencies) / len(latencies)
    elif any(value is not None for value in latencies):
        _warn_once("incomplete teacher request latency")

    submissions = [row.get("rollout_submission") for row in rows]
    submission = submissions[0]
    sessions = [key.rsplit("_", 2) for key in keys]
    complete = (
        isinstance(submission, dict)
        and bool(submission.get("id"))
        and submission.get("expected_requests") == len(rows)
        and all(item == submission for item in submissions)
        and all(len(parts) == 3 and parts[2] == "0" for parts in sessions)
        and len(set(keys)) == len(keys)
    )
    if not complete:
        _warn_once("not one complete single-turn submission (unsupported mode, missing or duplicate requests)")
        return metrics

    # These are overlapping request latencies, not additive GPU/outer costs.
    # The TQ payload already carries them; no extra RPC or synchronization is needed.
    for stage in ("generate_sequences", "compute_score"):
        values = [row.get(stage) for row in rows]
        if all(_nonnegative(value) for value in values):
            prefix = f"perf/request_latency/{stage}"
            metrics[f"{prefix}/min_seconds"] = min(values)
            metrics[f"{prefix}/max_seconds"] = max(values)
            metrics[f"{prefix}/mean_seconds"] = sum(values) / len(values)
        elif any(value is not None for value in values):
            _warn_once(f"incomplete {stage} request latency")

    # Counts do not need synchronized clocks, but do need every request to report them.
    preemptions = [row.get("num_preempted") for row in rows]
    if all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in preemptions):
        metrics["rollout/preemptions"] = sum(preemptions)
    else:
        _warn_once("backend did not report preemptions for every request; unknown is not zero")

    if any(row.get("finish_reason") is not None for row in rows):
        reasons = [row.get("finish_reason") or "unknown" for row in rows]
        for reason in ("stop", "length", "repetition", "aborted", "completed", "unknown"):
            count = sum(r == reason or (reason == "aborted" and r == "abort") for r in reasons)
            metrics[f"rollout/finish_reason/{reason}_count"] = count
    start = submission.get("started_at_ns")
    domain = submission.get("clock_domain")
    finishes = [row.get("generation_finished_at_ns") for row in rows]
    if (
        domain
        and isinstance(start, int)
        and start >= 0
        and all(row.get("generation_clock_domain") == domain for row in rows)
        and all(isinstance(finish, int) and finish >= start for finish in finishes)
    ):
        metrics["timing_s/rollout_batch"] = (max(finishes) - start) / 1e9
    else:
        _warn_once("missing completion timestamp or different/unverifiable monotonic clock domains")
    return metrics


def aggregate_validation_windows(windows: list[dict[str, float]]) -> dict[str, float]:
    """Sum complete validation windows, excluding gaps and never hiding missing windows."""
    if not windows:
        return {}
    return {
        f"validation/{key}": sum(window[key] for window in windows)
        for key in ("timing_s/rollout_batch", "rollout/preemptions")
        if all(key in window for window in windows)
    }
