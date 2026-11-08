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
        metrics.update(position_rkl_metrics(selected, prefix=prefix + "/position"))
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
    if history:
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


def position_rkl_metrics(rows, *, prefix, field="rkl_position_sum", total_field="rkl_sum"):
    """Token-weighted deciles; legacy rows expose coverage without invented values."""
    from verl.utils.metric.position import POSITION_BINS

    rows = list(rows)
    present = [r for r in rows if r.get(field) is not None and r.get("rkl_position_count") is not None]
    metrics = {f"{prefix}/rows": len(present)}
    if not rows or len(present) != len(rows):
        return metrics
    for row in rows:
        sums, counts = row[field], row["rkl_position_count"]
        if len(sums) != len(POSITION_BINS) or len(counts) != len(POSITION_BINS):
            raise ValueError("RKL position statistics must contain ten bins")
        if any(type(n) is not int or n < 0 for n in counts) or sum(counts) != row["tokens"]:
            raise ValueError("RKL position counts differ from trajectory mask")
        if all(math.isfinite(v) for v in sums) and math.isfinite(row[total_field]):
            if not math.isclose(math.fsum(sums), row[total_field], rel_tol=1e-5, abs_tol=1e-7):
                raise ValueError("RKL position sums differ from trajectory total")
    for index, label in enumerate(POSITION_BINS):
        finite = all(math.isfinite(r[field][index]) for r in rows)
        total = math.fsum(r[field][index] for r in rows) if finite else math.nan
        count = sum(r["rkl_position_count"][index] for r in rows)
        stem = f"{prefix}/{label}"
        metrics[stem + "/sum"] = total
        metrics[stem + "/token_count"] = count
        if count and math.isfinite(total):
            metrics[stem + "/mean"] = total / count
    return metrics
