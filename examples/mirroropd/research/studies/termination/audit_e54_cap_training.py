# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Recompute paired cap-study effects and inspect newly covered ending states.

CPU-only analysis. Bootstrap units are the 64 fixed questions, not training
seeds. A local stopping-logit derivative is not a parameter-gradient norm.
"""

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from examples.mirroropd.research.core.io import read_rows as read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    report = {"paired_evaluation": {}, "training": {}}
    rng = np.random.default_rng(20261029)
    draws = rng.integers(0, 64, size=(10000, 64))
    for dataset in ("math", "dapo"):
        arms = {}
        for cap in (7168, 16384):
            rows = read(root / f"evaluation/cap{cap}-{dataset}/scored.jsonl")
            assert len(rows) == 64 and all(r["status"] == "ok" for r in rows)
            arms[cap] = {r["prompt_id"]: r for r in rows}
            assert len(arms[cap]) == 64
        assert arms[7168].keys() == arms[16384].keys()
        ids = sorted(arms[7168])
        assert all(arms[7168][i]["prompt_ids"] == arms[16384][i]["prompt_ids"] for i in ids)
        assert all(arms[7168][i]["seed"] == arms[16384][i]["seed"] for i in ids)
        fields = {
            "accuracy": lambda r: float(r["acc"]),
            "mean_length": lambda r: float(r["response_length"]),
            "cap_fraction": lambda r: float(r["finish_reason"] == "length"),
            "correct_and_stopped": lambda r: float(bool(r["acc"]) and r["finish_reason"] == "stop"),
        }
        report["paired_evaluation"][dataset] = {}
        for key, value in fields.items():
            delta = np.array([value(arms[16384][i]) - value(arms[7168][i]) for i in ids])
            report["paired_evaluation"][dataset][key] = {
                "difference_16k_minus_7k": float(delta.mean()),
                "paired_question_bootstrap_95ci": np.quantile(delta[draws].mean(axis=1), [0.025, 0.975]).tolist(),
            }
    first_batches = {}
    for cap in (7168, 16384):
        arm = root / f"cap{cap}"
        trajectories = [r for step in (3, 4, 5) for r in read(arm / f"debug/rollout-{step}.jsonl")]
        assert len(trajectories) == 192
        first_batches[cap] = {tuple(r["prompt_ids"]): r for r in trajectories if r["outer"] == 3}
        states = [r for p in (arm / "debug").glob("states-rank-*.jsonl") for r in read(p)]
        ends = [r for r in states if r["inner"] == 1 and r["terminal_target"] and r["position"] >= 7168]
        late = [r for r in trajectories if r["stopped_on_eos"] and r["length"] > 7168]
        assert len(ends) == len(late)
        tokens = sum(r["length"] for r in trajectories)
        report["training"][str(cap)] = {
            "trajectories": len(trajectories),
            "response_tokens": tokens,
            "token_update_visits": tokens * 16,
            "fraction_tokens_beyond_7168": sum(max(0, r["length"] - 7168) for r in trajectories) / tokens,
            "natural_ends_beyond_7168": len(late),
            "judged_correct_natural_ends_beyond_7168": sum(r["score"] > 0 for r in late),
            "teacher_stop_at_new_ends": {
                "mean": statistics.mean(r["teacher_stop"] for r in ends) if ends else None,
                "median": statistics.median(r["teacher_stop"] for r in ends) if ends else None,
                "above_099": sum(r["teacher_stop"] > 0.99 for r in ends),
                "below_0001": sum(r["teacher_stop"] < 0.0001 for r in ends),
                "local_stop_gradient_positive": sum(r["stop_logit_gradient"] > 0 for r in ends),
            },
            "new_terminal_sites": [
                {
                    k: r[k]
                    for k in ["sample_key", "outer", "position", "student_stop", "teacher_stop", "stop_logit_gradient"]
                }
                for r in ends
            ],
        }
    assert first_batches[7168].keys() == first_batches[16384].keys()
    shared_prefixes = sum(
        a["response_ids"] == first_batches[16384][key]["response_ids"][: len(a["response_ids"])]
        for key, a in first_batches[7168].items()
    )
    report["first_branch_batch"] = {"matched_questions": 64, "exact_short_response_prefixes": shared_prefixes}
    report["scope"] = "One training seed; paired-question intervals do not quantify training-seed variance."
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
