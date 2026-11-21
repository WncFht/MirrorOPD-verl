# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Disjoint calibration data and learning-rate selection for E94–E114."""

import json
import math


def prompt_text(messages):
    if isinstance(messages, str):
        messages = json.loads(messages)
    return " ".join(" ".join(m["content"].split()).casefold() for m in messages)


def development_split(training, heldout):
    """Reserve calibration data beyond the 40×64 formal-training prefix."""
    excluded = {prompt_text(r["prompt"]) for r in training[:2560]}
    excluded.update(prompt_text(r["messages"]) for r in heldout)
    unique = []
    for row in training[2560:]:
        key = prompt_text(row["prompt"])
        if key not in excluded:
            excluded.add(key)
            unique.append(row)
        if len(unique) == 288:
            return unique[:256], unique[256:]
    raise ValueError("Need 256 calibration-training and 32 disjoint development questions")


def candidate_score(events):
    """Average relative RKL reduction over both banks and the last two outers."""
    values = []
    for bank in ("public_initial", "public_teacher"):
        prefix = f"eval/rkl/{bank}/math/"
        baseline = events[0][prefix + "question_mean"]
        if not math.isfinite(baseline) or baseline <= 0:
            raise ValueError("Calibration requires a positive finite baseline RKL")
        for metrics in events:
            value = metrics[prefix + "question_mean"]
            if not math.isfinite(value) or metrics[prefix + "invalid_rows"] or metrics[prefix + "valid_rows"] != 32:
                raise ValueError("Incomplete or nonfinite development-bank coverage")
        values.extend(1 - metrics[prefix + "question_mean"] / baseline for metrics in events[-2:])
    return sum(values) / len(values)


def next_lr(results, expansions):
    """Bracket edge winners with at most two additional logarithmic probes."""
    best = max(results, key=lambda row: (row["score"], -row["lr"]))
    if expansions >= 2:
        return None
    lrs = sorted(row["lr"] for row in results)
    if best["lr"] == lrs[0]:
        return lrs[0] / 3
    if best["lr"] == lrs[-1]:
        return lrs[-1] * 3
    return None
