# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Separate teacher length, EOS coverage, and conditional student cap recovery."""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

from examples.mirroropd.research.core.io import read_rows as read

CAPS = (7168, 16384, 31744)


def metrics(generated, scored):
    n = len(generated)
    lengths = [r["response_length"] for r in generated]
    stops = [r for r in generated if r["finish_reason"] == "stop"]
    result = {
        "n": n,
        "mean_observed_length": statistics.mean(lengths),
        "length_quantiles": dict(
            zip(["p50", "p90", "p95", "p99"], np.quantile(lengths, [0.5, 0.9, 0.95, 0.99]).tolist(), strict=False)
        ),
        "max_observed_length": max(lengths),
        "normal_stops": len(stops),
        "stop_token_counts": dict(Counter(str(r["response_ids"][-1]) for r in stops)),
        "role_marker_responses": sum(151644 in r["response_ids"] for r in generated),
        "by_cap": {},
    }
    ids = {r["request_id"] for r in generated}
    assert len(ids) == n
    for cap in CAPS:
        rows = [r for r in scored if r["request_id"] in ids and r["cap"] == cap]
        assert len(rows) == n and all(r["status"] == "ok" for r in rows)
        result["by_cap"][str(cap)] = {
            "stopped": sum(r["stopped_within_cap"] for r in rows),
            "correct": sum(bool(r["acc"]) for r in rows),
            "correct_and_stopped": sum(bool(r["acc"]) and r["stopped_within_cap"] for r in rows),
            "mean_emitted_tokens": statistics.mean(r["response_length"] for r in rows),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    teacher = read(root / "teacher/generated.jsonl")
    teacher_scores = read(root / "teacher/scored.jsonl")
    assert len(teacher) == 448 and len(teacher_scores) == 448 * 3
    random_teacher = [r for r in teacher if r["cohort"] == "random_training"]
    assert len(random_teacher) == 320
    result = {"teacher_random_training": metrics(random_teacher, teacher_scores), "continuations": {}}
    teacher_groups = {"teacher_random_training": random_teacher}
    if (root / "teacher-native/scored.jsonl").exists():
        native = read(root / "teacher-native/generated.jsonl")
        native_scores = read(root / "teacher-native/scored.jsonl")
        assert len(native) == 320 and len(native_scores) == 960
        assert [(r["request_id"], r["seed"]) for r in native] == [(r["request_id"], r["seed"]) for r in random_teacher]
        result["teacher_native_random_training"] = metrics(native, native_scores)
        teacher_groups["teacher_native_random_training"] = native
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(teacher[0]["model"], local_files_only=True)
    for label, rows in teacher_groups.items():
        texts = tokenizer.batch_decode([r["response_ids"] for r in rows], skip_special_tokens=True)
        lengths_nonempty = [r["response_length"] for r, text in zip(rows, texts, strict=True) if text.strip()]
        result[label]["empty_text_count"] = len(rows) - len(lengths_nonempty)
        result[label]["nonempty_mean_length"] = statistics.mean(lengths_nonempty)
    rng = np.random.default_rng(20261028)
    lengths = np.array([r["response_length"] for r in random_teacher])
    means = lengths[rng.integers(0, 320, size=(10000, 320))].mean(axis=1)
    result["teacher_random_training"]["mean_question_bootstrap_95ci"] = np.quantile(means, [0.025, 0.975]).tolist()
    observed = json.loads((root / "observed-student.json").read_text())
    for label, outer, expected in [("student5", 6, 50), ("student10", 11, 58)]:
        generated = read(root / label / "generated.jsonl")
        scores = read(root / label / "scored.jsonl")
        assert len(generated) == expected and len(scores) == expected * 3
        baseline = {r["request_id"]: r for r in scores if r["cap"] == 7168}
        assert all(r["score"] == r["original_score"] for r in baseline.values()), "Original-prefix grading changed"
        item = metrics(generated, scores)
        question_ids = set(baseline)
        item["teacher_from_same_questions"] = metrics(
            [r for r in teacher if r["request_id"] in question_ids], teacher_scores
        )
        item["transitions"] = {}
        for cap in CAPS[1:]:
            rows = [r for r in scores if r["cap"] == cap]
            item["transitions"][str(cap)] = {
                "wrong_to_correct": sum(not baseline[r["request_id"]]["acc"] and bool(r["acc"]) for r in rows),
                "correct_to_wrong": sum(bool(baseline[r["request_id"]]["acc"]) and not r["acc"] for r in rows),
                "newly_correct_and_stopped": sum(
                    not baseline[r["request_id"]]["acc"] and bool(r["acc"]) and r["stopped_within_cap"] for r in rows
                ),
                "already_correct_now_stopped": sum(
                    bool(baseline[r["request_id"]]["acc"]) and bool(r["acc"]) and r["stopped_within_cap"] for r in rows
                ),
            }
        extras = [r["response_length"] - 7168 for r in generated if r["finish_reason"] == "stop"]
        item["extra_tokens_until_stop"] = {
            "n": len(extras),
            "median": statistics.median(extras) if extras else None,
            "within_128": sum(x <= 128 for x in extras),
            "within_1024": sum(x <= 1024 for x in extras),
        }
        original = [r for r in observed if r["cohort"] == f"observed_outer{outer}"]
        assert len(original) == 64
        earlier = [r for r in original if r["original_stopped"]]
        assert len(earlier) + expected == 64
        item["combined_original_batch"] = {}
        for cap in CAPS:
            rows = [r for r in scores if r["cap"] == cap]
            item["combined_original_batch"][str(cap)] = {
                "n": 64,
                "stopped": len(earlier) + sum(r["stopped_within_cap"] for r in rows),
                "correct": sum(r["original_score"] > 0 for r in earlier) + sum(bool(r["acc"]) for r in rows),
                "correct_and_stopped": sum(r["original_score"] > 0 for r in earlier)
                + sum(bool(r["acc"]) and r["stopped_within_cap"] for r in rows),
                "mean_emitted_tokens": (
                    sum(r["original_length"] for r in earlier) + sum(r["response_length"] for r in rows)
                )
                / 64,
            }
        result["continuations"][label] = item
    result["combined_student_batches"] = {}
    for cap in CAPS:
        rows = [x["combined_original_batch"][str(cap)] for x in result["continuations"].values()]
        n = sum(x["n"] for x in rows)
        tokens = sum(x["mean_emitted_tokens"] * x["n"] for x in rows)
        stopped = sum(x["stopped"] for x in rows)
        result["combined_student_batches"][str(cap)] = {
            "n": n,
            "stopped": stopped,
            "correct": sum(x["correct"] for x in rows),
            "correct_and_stopped": sum(x["correct_and_stopped"] for x in rows),
            "mean_emitted_tokens": tokens / n,
            "sampled_eos_per_1000_tokens": 1000 * stopped / tokens,
        }
    result["scope"] = "Frozen model draws and conditional continuation; no longer-cap training was run."
    (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
