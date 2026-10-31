# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Question-weighted RKL reports; no GPU work or changes to training loss."""

import math
from collections import defaultdict


def summarize_rkl(rows, *, equal_groups=None):
    """Do not publish a scientific mean when any requested record is invalid."""
    rows = list(rows)
    groups = defaultdict(list)
    valid, tokens = 0, 0
    seen = set()
    for row in rows:
        key = row["id"]
        if key in seen:
            raise ValueError("Duplicate RKL trajectory identity")
        seen.add(key)
        total, count = row["rkl_sum"], row["tokens"]
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0 or not math.isfinite(total):
            continue
        group = row[equal_groups] if equal_groups else None
        if equal_groups and group is None:
            raise ValueError("Missing RKL aggregation group")
        groups[group].append(float(total) / count)
        valid += 1
        tokens += count
    result = {"rows": len(rows), "valid_rows": valid, "invalid_rows": len(rows) - valid, "tokens": tokens}
    if rows and valid == len(rows):
        result["mean"] = math.fsum(math.fsum(v) / len(v) for v in groups.values()) / len(groups)
    return result


def readout_metrics(rows, *, baseline=None, previous_history=()):
    """Report source risk, balanced history risk and genuinely paired changes."""
    source = [r for r in rows if r["bank"] == "source_heldout"]
    history = [r for r in rows if r["bank"] == "history"]
    metrics = {"eval/rkl/rows": len(rows)}
    for label, selected, group in [("source_heldout", source, None), ("history", history, "history_batch")]:
        if not selected:
            continue
        summary = summarize_rkl(selected, equal_groups=group)
        prefix = f"eval/rkl/{label}"
        for key in ("rows", "valid_rows", "invalid_rows", "tokens"):
            metrics[f"{prefix}/{key}"] = summary[key]
        if "mean" in summary:
            name = "equal_batch_question_mean" if group else "question_mean"
            metrics[f"{prefix}/{name}"] = summary["mean"]
    current = summarize_rkl(source)
    if baseline is not None and source:
        if {(r["id"], r["tokens"]) for r in source} != {(r["id"], r["tokens"]) for r in baseline}:
            raise ValueError("Fixed source baseline has different trajectory identities or masks")
        initial = summarize_rkl(baseline)
        if "mean" in initial and "mean" in current:
            metrics["eval/rkl/source_heldout/absolute_reduction"] = initial["mean"] - current["mean"]
            if initial["mean"] > 0:
                metrics["eval/rkl/source_heldout/reduction_percent"] = 100 * (1 - current["mean"] / initial["mean"])
    previous = {r["id"]: r for r in previous_history}
    paired = [r for r in history if r["id"] in previous]
    metrics["eval/rkl/history/paired_rows"] = len(paired)
    if paired:
        old = [previous[r["id"]] for r in paired]
        if any(a["tokens"] != b["tokens"] for a, b in zip(old, paired, strict=True)):
            raise ValueError("Paired history masks changed")
        before = summarize_rkl(old, equal_groups="history_batch")
        after = summarize_rkl(paired, equal_groups="history_batch")
        if "mean" in before and "mean" in after:
            metrics["eval/rkl/history/paired_absolute_reduction"] = before["mean"] - after["mean"]
    return metrics
