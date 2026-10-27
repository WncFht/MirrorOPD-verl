#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Independently check a smoke run's recorded scalars against final sessions.

No model imports or network calls. Voting uses explicit subset enumeration,
independent of the training aggregator's vectorized implementation (N <= 8).
"""

import argparse
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def close(actual, expected, name):
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-10):
        raise AssertionError((name, actual, expected))


def verify(root, correction=False):
    rows = [json.loads(line)["data"] for line in (root / "metrics.jsonl").read_text().splitlines()]
    updates = [row for row in rows if "optim/actor/loss" in row]
    assert updates, "No optimizer observations"
    assert [r["step/update"] for r in updates] == list(range(1, len(updates) + 1))
    successful = 0
    for row in updates:
        successful += int(row["optim/actor/update_applied"])
        assert row["step/optimizer"] == successful
        assert row["optim/actor/loss_nonfinite"] == row["optim/actor/grad_norm_preclip_nonfinite"] == 0
        assert "policy/current_behavior/sequence_log_ratio_m2" in row
        assert "policy/current_behavior/ratio_p99" in row
        assert 0 <= row["policy/current_behavior/ess_fraction"] <= 1
        assert row["policy/current_behavior/nonfinite_fraction"] == 0
    keys = sorted(set().union(*(r.keys() for r in rows)))
    assert all(math.isfinite(value) for row in rows for value in row.values())
    validations = []
    aliases = {"MATH-500": "math500", "AMC23": "amc23", "AIME24": "aime24", "AIME25": "aime25", "AIME26": "aime26"}
    for file in sorted((root / "training/validation/records").glob("*.jsonl"), key=lambda p: int(p.stem)):
        outer = int(file.stem)
        matched = [r for r in rows if r.get("step/outer") == outer and "eval/math500/acc/mean" in r]
        assert len(matched) == 1, (outer, len(matched))
        logged = matched[0]
        records = [json.loads(line) for line in file.read_text().splitlines()]
        groups = defaultdict(list)
        for record in records:
            groups[(record["data_source"], record["prompt_id"])].append(record)
        sources = defaultdict(list)
        for (source, _), responses in groups.items():
            responses.sort(key=lambda r: r["sample_index"])
            assert [r["sample_index"] for r in responses] == list(range(len(responses)))
            assert len(responses) <= 8, "This independent exhaustive smoke verifier requires N <= 8"
            sources[source].append(responses)
        details = {}
        for source, prompts in sources.items():
            prefix = f"eval/{aliases[source]}/"
            expected = sum(len(p) for p in prompts)
            observed = sum(
                isinstance(r.get("acc"), int | float) and math.isfinite(r["acc"]) for p in prompts for r in p
            )
            close(logged[prefix + "response_count"], expected, prefix + "response_count")
            close(logged[prefix + "failed_count"], expected - observed, prefix + "failed_count")
            k_max = min(map(len, prompts))
            for k in range(1, k_max + 1):
                estimates = defaultdict(list)
                for prompt in prompts:
                    acc = [
                        float(r["acc"]) if isinstance(r.get("acc"), int | float) and math.isfinite(r["acc"]) else 0
                        for r in prompt
                    ]
                    pred = [
                        json.dumps(r["pred"], sort_keys=True, ensure_ascii=False)
                        if r.get("pred") not in (None, "")
                        else None
                        for r in prompt
                    ]
                    values_by_pred = defaultdict(list)
                    for p, value in zip(pred, acc, strict=True):
                        if p is not None:
                            values_by_pred[p].append(value)
                    subset_pass, subset_all, subset_vote = [], [], []
                    for subset in itertools.combinations(range(len(prompt)), k):
                        subset_pass.append(max(acc[i] for i in subset))
                        subset_all.append(min(acc[i] for i in subset))
                        votes = Counter(pred[i] for i in subset if pred[i] is not None)
                        if not votes:
                            subset_vote.append(0.0)
                        else:
                            winners = [p for p, n in votes.items() if n == max(votes.values())]
                            subset_vote.append(
                                sum(sum(values_by_pred[p]) / len(values_by_pred[p]) for p in winners) / len(winners)
                            )
                    for name, vals in [("pass", subset_pass), ("all_correct", subset_all), ("majority", subset_vote)]:
                        estimates[name].append(sum(vals) / len(vals))
                    estimates["prefix_mean"].append(sum(acc[:k]) / k)
                for name, estimates_for_prompts in estimates.items():
                    field = prefix + f"acc/{name}@{k}"
                    close(logged[field], sum(estimates_for_prompts) / len(prompts), field)
                for name in ["pass", "all_correct", "majority"]:
                    lower, upper = (logged[prefix + f"acc/{name}@{k}_ci_{end}"] for end in ["low", "high"])
                    assert 0 <= lower <= upper <= 1
            details[source] = {
                "prompts": len(prompts),
                "responses": expected,
                "failed": expected - observed,
                "mean": logged[prefix + "acc/mean"],
                "pass@K": logged[prefix + f"acc/pass@{k_max}"],
                "majority@K": logged[prefix + f"acc/majority@{k_max}"],
                "verified_budgets": list(range(1, k_max + 1)),
            }
        validations.append({"outer": outer, "optimizer": logged["step/optimizer"], "datasets": details})
    if correction:
        assert not validations
        for row in updates:
            raw = row["policy/current_behavior/valid_token_count"]
            used = row["advantage/distillation/token_count"]
            close(row["correction/is/applied_token_count"], raw, "original weight denominator")
            close(row["correction/rejection/kept_token_count"], used, "kept tokens")
            close(row["correction/rejection/token_fraction"], (raw - used) / raw, "RS fraction")
            close(row["correction/is/applied_ess"], raw * row["correction/is/applied_ess_fraction"], "absolute ESS")
            assert "correction/anchor_ppo/dual_clip_fraction" in row
    else:
        assert len(validations) >= 2, "Require initial and post-update evaluation"
        for row in updates:
            outer = next(r for r in rows if r.get("step/outer") == row["step/outer"] and "rollout/response_tokens" in r)
            close(row["rollout/training_tokens"], outer["rollout/response_tokens"], "unfiltered token workload")
    result = {
        "scalar_rows": len(rows),
        "field_count": len(keys),
        "update_attempts": len(updates),
        "optimizer_steps": successful,
        "mode": "correction" if correction else "train_eval",
        "validations": validations,
        "keys": keys,
        "independent_checks_passed": True,
    }
    (root / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "keys"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--correction", action="store_true")
    args = parser.parse_args()
    verify(args.run_root, args.correction)
