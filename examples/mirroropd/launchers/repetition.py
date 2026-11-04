# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Resolve launcher repetition options for native evaluation protocols."""


def validation_repetition(env):
    # Keep plan/config-only generation independent of Torch and worker imports.
    enabled = (env.get("VAL_REPETITION_DETECTION") or "true").lower()
    if enabled in {"false", "0"}:
        return None
    if enabled not in {"true", "1"}:
        raise ValueError("VAL_REPETITION_DETECTION must be true or false")
    params = {}
    for name, key, default in (
        ("MIN_PATTERN", "min_pattern_size", "16"),
        ("MAX_PATTERN", "max_pattern_size", "128"),
        ("MIN_COUNT", "min_count", "8"),
    ):
        value = env.get(f"VAL_REPETITION_{name}") or default
        if not value.isascii() or not value.isdecimal():
            raise ValueError(f"VAL_REPETITION_{name} must be a positive integer")
        params[key] = int(value)
    if min(params.values()) < 1 or params["min_count"] < 2 or params["min_pattern_size"] > params["max_pattern_size"]:
        raise ValueError("Validation repetition detection requires positive values, min pattern <= max and count >= 2")
    return params
