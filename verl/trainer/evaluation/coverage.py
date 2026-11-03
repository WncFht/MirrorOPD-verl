# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Bind RKL event coverage to the planned bank selections before publication."""

from collections import defaultdict

from .artifacts import identity
from .metrics import summarize_rkl


def rkl_coverage(expected, observed):
    """Require exact (bank, id) coverage; retain invalid scientific records.

    A trajectory may appear in several banks. Duplicates within a bank, missing
    selections and unexpected results fail the event before metric reduction.
    Empty selections are artifact-complete but contain no scientific readout.
    """
    expected, observed = list(expected), list(observed)

    def index(rows):
        result = {}
        for row in rows:
            key = row["bank"], row["id"]
            if not all(isinstance(value, str) and value for value in key):
                raise ValueError("RKL coverage requires nonempty bank and selection IDs")
            if key in result:
                raise ValueError("Duplicate RKL bank selection in event coverage")
            result[key] = row
        return result

    planned, actual = index(expected), index(observed)
    missing, extra = planned.keys() - actual.keys(), actual.keys() - planned.keys()
    if missing or extra:
        raise ValueError(f"RKL event coverage mismatch: missing={len(missing)}, unexpected={len(extra)}")
    banks = defaultdict(list)
    for key, row in actual.items():
        reference = planned[key]
        if row.get("trajectory_id", row["id"]) != reference.get("trajectory_id", reference["id"]):
            raise ValueError("RKL event coverage changed trajectory identity")
        count = reference.get("valid_tokens")
        if count is None and "response_mask" in reference:
            count = sum(reference["response_mask"])
        if count is not None and row["tokens"] != count:
            raise ValueError("RKL event coverage changed response mask count")
        banks[key[0]].append(row)
    summaries = {}
    for bank, rows in sorted(banks.items()):
        summary = summarize_rkl(rows)
        summaries[bank] = {
            "expected_rows": len(rows),
            "observed_rows": len(rows),
            "valid_rows": summary["valid_rows"],
            "invalid_rows": summary["invalid_rows"],
            "missing_rows": 0,
            "scientific_complete": summary["valid_rows"] == len(rows),
        }
    valid = sum(summary["valid_rows"] for summary in summaries.values())
    return {
        "expected_rows": len(expected),
        "observed_rows": len(observed),
        "valid_rows": valid,
        "invalid_rows": len(observed) - valid,
        "missing_rows": 0,
        "selection_sha256": identity(sorted(planned)),
        "scientific_complete": bool(expected) and valid == len(expected),
        "banks": summaries,
    }
