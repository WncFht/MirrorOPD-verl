# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Paired fixed-question diagnostics and explicit E54 training budgets."""

import argparse
import json
import statistics
from pathlib import Path


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np

    evaluation = args.root / "evaluation"
    expected = {r["prompt_id"]: r for r in json.loads((evaluation / "prompts.json").read_text())}
    labels = ["outer2", "outer5", "r16-outer10", "r16-outer15", "r4-outer14"]
    budgets = [(32, 128), (80, 320), (160, 640), (240, 960), (80, 896)]
    all_rows, report = {}, {"evaluation": [], "paired_differences": [], "training": {}}
    for label, (updates, prompts) in zip(labels, budgets, strict=True):
        data = read(evaluation / label / "scored.jsonl")
        rows = {r["prompt_id"]: r for r in data}
        assert rows.keys() == expected.keys() and len(rows) == len(data) == 64
        assert all(r["seed"] == expected[key]["seed"] and r["status"] == "ok" for key, r in rows.items())
        assert all(r["response_length"] == len(r["response_ids"]) <= 31744 for r in rows.values())
        assert all(r["finish_reason"] in ("stop", "length") for r in rows.values())
        all_rows[label] = rows
        metrics = json.loads((evaluation / label / "metrics.json").read_text())
        assert abs(metrics["accuracy"] - statistics.mean(float(r["acc"]) for r in data)) < 1e-12
        metrics["correct_and_stopped_count"] = sum(bool(r["acc"]) and r["finish_reason"] == "stop" for r in data)
        metrics["correct_capped_count"] = sum(bool(r["acc"]) and r["finish_reason"] == "length" for r in data)
        report["evaluation"].append(
            {"model": label, "optimizer_updates": updates, "training_prompts": prompts} | metrics
        )
    rng = np.random.default_rng(20261028)
    draws = rng.integers(0, 64, size=(10000, 64))
    ids = sorted(expected)
    for first, second in [
        ("outer2", "outer5"),
        ("outer2", "r16-outer10"),
        ("outer2", "r16-outer15"),
        ("outer5", "r16-outer10"),
        ("r16-outer10", "r16-outer15"),
        ("outer5", "r16-outer15"),
        ("outer5", "r4-outer14"),
    ]:
        comparison = {"first": first, "second": second, "direction": "second minus first"}
        fields = {
            "accuracy": lambda r: float(r["acc"]),
            "mean_length": lambda r: float(r["response_length"]),
            "cap_fraction": lambda r: float(r["finish_reason"] == "length"),
        }
        for name, value in fields.items():
            delta = np.array([value(all_rows[second][i]) - value(all_rows[first][i]) for i in ids])
            means = delta[draws].mean(axis=1)
            comparison[name] = {
                "difference": float(delta.mean()),
                "paired_prompt_bootstrap_95ci": np.quantile(means, [0.025, 0.975]).tolist(),
            }
        report["paired_differences"].append(comparison)
    initial = read(args.initial / "debug/rollout-summary.jsonl")
    continued = read(args.root / "continue-r16/debug/rollout-summary.jsonl")
    branch = read(args.root / "branch-r4/debug/rollout-summary.jsonl")
    for name, rows in [("continue-r16", continued), ("branch-r4", branch)]:
        report["training"][name] = {
            "batches": len(rows),
            "responses": sum(r["samples"] for r in rows),
            "mean_length": statistics.mean(r["mean_length"] for r in rows),
            "cap_count": sum(r["cap_count"] for r in rows),
            "training_accuracy": statistics.mean(r["mean_score"] for r in rows),
        }
    report["token_update_visits"] = {
        "outer5": sum(r["mean_length"] * r["samples"] * 16 for r in initial),
        "r16-outer15": sum(r["mean_length"] * r["samples"] * 16 for r in initial + continued),
        "r4-outer14": sum(r["mean_length"] * r["samples"] * 16 for r in initial[:2])
        + sum(r["mean_length"] * r["samples"] * 4 for r in branch),
    }
    report["scope"] = (
        "One stochastic response per fixed question; paired bootstrap resamples these 64 questions, "
        "not training seeds. Training response-token visits are not FLOPs or gradient norms."
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
