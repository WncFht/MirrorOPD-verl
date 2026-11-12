# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Pure boundary plans and bounded packing; no Ray handles or model side effects."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationPlan:
    outer: int
    optimizer_steps: int
    tasks: tuple[str, ...]
    generate_current: bool


def compile_plan(config, *, outer, total, optimizer_steps):
    if type(outer) is not int or type(total) is not int or not 0 <= outer <= total or total < 1:
        raise ValueError("Evaluation requires a boundary within a positive training budget")
    if outer == 0:
        # Fixed-bank reductions and teacher gaps require a common initial
        # reference even when periodic schedules begin at a later boundary.
        return EvaluationPlan(
            0,
            optimizer_steps,
            (("source_heldout",) if config.source_enabled else ())
            + ("public_initial", "public_teacher")
            + (("accuracy",) if config.accuracy_enabled else ())
            + ("public_current",)
            + (("teacher_accuracy",) if config.accuracy_enabled else ()),
            False,
        )
    tasks = []
    if config.source_enabled and config.schedules["source"].due(outer, total):
        tasks.append("source_heldout")
    if config.history_enabled and outer > 0 and config.schedules["history"].due(outer, total):
        tasks.append("history")
    if config.schedules["banks"].due(outer, total):
        tasks.extend(("public_initial", "public_teacher"))
    bench = config.schedules["bench"].due(outer, total)
    if bench:
        if config.accuracy_enabled:
            tasks.append("accuracy")
        tasks.append("public_current")
    return EvaluationPlan(outer, optimizer_steps, tuple(tasks), bench and outer > 0)


def payload_windows(rows, *, hidden_size, payload_bytes, dtype_bytes=2):
    """First-fit-decreasing on complete trajectories; bound logical hidden+positions.

    GPU microbatches and host-copy reserves are separate limits. Padding has no
    effective response positions and is added later by the inference balancer.
    """
    if hidden_size <= 0 or dtype_bytes <= 0 or payload_bytes <= 0:
        raise ValueError("Positive scoring dimensions/budget required")
    entries = []
    seen = set()
    for row in rows:
        if row["id"] in seen:
            raise ValueError("Duplicate scoring trajectory ID")
        seen.add(row["id"])
        count = row.get("valid_tokens")
        if count is None:
            count = sum(row["response_mask"])
        if count < 0:
            raise ValueError("Negative response count")
        size = count * (hidden_size * dtype_bytes + 8)
        if size > payload_bytes:
            raise ValueError(f"One complete trajectory exceeds the hidden window: {row['id']}")
        entries.append((size, row))
    windows = []
    for size, row in sorted(entries, key=lambda x: (-x[0], x[1]["id"])):
        for window in windows:
            if window[0] + size <= payload_bytes:
                window[0] += size
                window[1].append(row)
                break
        else:
            windows.append([size, [row]])
    return tuple(tuple(rows) for _, rows in windows)
