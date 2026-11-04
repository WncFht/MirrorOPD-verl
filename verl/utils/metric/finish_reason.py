# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Backend stop causes, shared by training and both evaluation consumers."""

from collections import Counter


def finish_reason_metrics(reasons):
    """Use every supplied response slot as denominator; unrecognized causes stay unknown.

    ``truncated`` is the observed union of length and repetition stops. It does
    not estimate how many repeated answers would have reached the length cap.
    ``completed`` carries no evidence of an EOS/stop token and stays separate.
    """
    names = ("stop", "length", "repetition", "aborted", "completed", "unknown")
    counts = Counter()
    for reason in reasons:
        reason = "aborted" if reason == "abort" else reason
        counts[reason if reason in names else "unknown"] += 1
    total = counts.total()
    if not total:
        return {}
    result = {"response_count": total}
    for reason in names:
        result[f"{reason}_count"] = counts[reason]
        result[f"{reason}_fraction"] = counts[reason] / total
    truncated = counts["length"] + counts["repetition"]
    result.update(truncated_count=truncated, truncated_fraction=truncated / total)
    return result
