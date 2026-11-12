# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Explicit per-trajectory normalization shared by requests and diagnostics."""


def validate_reduction(reduction):
    if reduction not in ("fixed_cap", "valid_tokens"):
        raise ValueError(f"Unknown response reduction: {reduction!r}")
    return reduction


def response_denominator(length, cap, reduction="fixed_cap"):
    """Normalize a nonempty, already truncated response; never truncate silently.

    fixed_cap preserves historical H-normalization. valid_tokens gives each
    question its own token mean before group/question averaging.
    """
    validate_reduction(reduction)
    if not 0 < length <= cap:
        raise ValueError("response length must be positive and no greater than cap")
    return cap if reduction == "fixed_cap" else length


def record_denominator(row, task):
    """Read measured normalization, with a fallback for historical fixed-H data."""
    reduction = validate_reduction(task.get("reduction", "fixed_cap"))
    if "token_count" not in row:
        if reduction != "fixed_cap":
            raise ValueError("valid_tokens results require per-record token_count")
        return task["cap"]
    expected = response_denominator(row["token_count"], task["cap"], reduction)
    if row.get("denominator", expected) != expected:
        raise ValueError("record denominator disagrees with measurement task")
    return expected
