# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Read append-only metrics with explicit, hash-verified incident exclusions."""

import hashlib
import json


def trusted_metrics(path, raw=None):
    raw = path.read_bytes() if raw is None else raw
    ledger = path.with_name("metrics-exclusions.json")
    if not ledger.exists():
        return raw, None
    specification = ledger.read_bytes()
    spans = json.loads(specification)["excluded_byte_ranges"]
    parts, cursor = [], 0
    for span in spans:
        start, end = span["start"], span["end"]
        if not cursor <= start < end <= len(raw):
            raise ValueError("Invalid metrics exclusion interval")
        if hashlib.sha256(raw[start:end]).hexdigest() != span["sha256"]:
            raise ValueError("Excluded metrics bytes changed; refusing an ambiguous run")
        if start and raw[start - 1 : start] != b"\n" or raw[end - 1 : end] != b"\n":
            raise ValueError("Metrics exclusion must align with complete JSONL records")
        parts.append(raw[cursor:start])
        cursor = end
    parts.append(raw[cursor:])
    return b"".join(parts), specification
