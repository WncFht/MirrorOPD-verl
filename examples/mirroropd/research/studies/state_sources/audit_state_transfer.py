#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent consistency checks on completed scientific probe artifacts."""

import argparse
import json
import math
from pathlib import Path


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--data", type=Path, required=True)
    args = a.parse_args()
    p = args.data
    prompts = json.loads((p / "prompts.json").read_text())
    assert len(prompts) == 24
    assert len({r["prompt"] for r in prompts}) == 24
    traces = [r for s in ["student", "justrl", "polaris"] for r in json.loads((p / f"rollouts-{s}.json").read_text())]
    assert len(traces) == 72
    for r in traces:
        assert r["prompt_ids"] == prompts[r["id"]]["prompt_ids"]
        assert len(r["response_ids"]) <= 768 and all(0 <= pos < len(r["response_ids"]) for pos in r["positions"])
    nstates = sum(len(r["positions"]) for r in traces)
    assert nstates == 3376
    initial = json.loads((p / "initial-state-metrics.json").read_text())
    assert len(initial) == 2 * nstates
    basekeys = {
        t: {(r["prompt_id"], r["source"], r["position"]) for r in initial if r["teacher"] == t}
        for t in ["justrl", "polaris"]
    }
    completed = []
    noop = None
    for folder in sorted(p.glob("*-split*-t*-lr*")):
        config = json.loads((folder / "protocol.json").read_text())
        train = json.loads((folder / "training.json").read_text())
        ev = json.loads((folder / "evaluation.json").read_text())
        assert [r["step"] for r in train] == list(range(1, config["steps"] + 1))
        assert all(math.isfinite(r["grad_norm"]) and math.isfinite(r["pre_update_train_kl"]) for r in train)
        assert set(config["train_ids"]) in [set(range(4)), set(range(4, 8))]
        assert set(config["train_ids"]).isdisjoint(range(8, 24))
        expected = [0, 1, 8, 16] if config["steps"] == 16 else [0, 1]
        assert sorted({r["step"] for r in ev}) == expected
        for step in expected:
            rows = [r for r in ev if r["step"] == step]
            assert (
                len(rows) == nstates
                and {(r["prompt_id"], r["source"], r["position"]) for r in rows} == basekeys[config["teacher"]]
            )
            assert all(math.isfinite(r["kl"]) and r["kl"] >= -1e-5 for r in rows)
        if config["lr"] == 0:
            lookup = {(r["prompt_id"], r["source"], r["position"]): r["kl"] for r in ev if r["step"] == 0}
            noop = max(
                abs(r["kl"] - lookup[(r["prompt_id"], r["source"], r["position"])]) for r in ev if r["step"] == 1
            )
            assert noop == 0
        completed.append(
            dict(
                arm=folder.name,
                steps=len(train),
                parameter_count=config["params"],
                clipped_updates=sum(r["grad_norm"] > 1 for r in train),
                peak_gib=max(r["allocated_peak_gib"] for r in train),
            )
        )
    assert len(completed) == 11 and sum(r["steps"] for r in completed) == 161
    geo = json.loads((p / "within-trace-gradient-geometry.json").read_text())
    assert len(geo) == 16
    assert {(r["teacher"], r["prompt_id"]) for r in geo} == {(t, i) for t in ["justrl", "polaris"] for i in range(8)}
    assert all("fixed response-prefix length 248" in r["units"] for r in geo)
    result = dict(
        prompts=24,
        rollouts=72,
        states=nstates,
        target_comparisons=len(initial),
        same_inputs_for_pre_post=True,
        no_train_eval_prompt_overlap=True,
        noop_max_kl_change=noop,
        completed_arms=completed,
        geometry_fixed_input=True,
        finite_losses_and_gradients=True,
    )
    (p / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
