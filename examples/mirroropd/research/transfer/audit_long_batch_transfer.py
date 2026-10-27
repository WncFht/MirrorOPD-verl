#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Check one-pass exposure, fixed banks, diagnostic isolation and optimizer continuity."""

import argparse
import collections
import json
import math
import re
from pathlib import Path

import torch

from ..common.io import read
from ..common.protocol import LONG_PASS, validate_long_pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    a = ap.parse_args()
    root = a.data
    validate_long_pass(read(root / "protocol.json"))
    bank = read(root / "bank.json")
    byid = {r["id"]: r for r in bank}
    assert len(bank) == LONG_PASS.examples and set(byid) == set(range(LONG_PASS.examples))
    assert len({re.sub(r"\s+", " ", r["prompt"]).strip() for r in bank}) == LONG_PASS.examples
    counts = collections.Counter(r["group"] for r in bank)
    assert len(counts) == 40 and set(counts.values()) == {16}
    assert {r["id"] for r in bank if r["split"] == "train"} == set(range(320))
    assert all(r["domain"] == "math" for r in bank if r["split"] == "train")
    assert collections.Counter(r["domain"] for r in bank if r["split"] == "heldout") == {
        "math": 192,
        "code": 64,
        "if": 64,
    }
    results = []
    for teacher in ["justrl", "polaris"]:
        dest = root / teacher
        cfg = read(dest / "protocol.json")
        history = read(dest / "training.json")
        assert read(dest / "complete.json") == dict(completed_batches=20, updates=320, finished=True)
        assert cfg["lr"] == 1e-6 and cfg["world_size"] == 2 and cfg["parameters"] == 1777088000
        assert [r["step"] for r in history] == list(range(1, 321))
        visits = collections.Counter()
        for row in history:
            b = (row["step"] - 1) // 16 + 1
            group = chr(64 + b)
            assert row["training_batch"] == b and row["group"] == group
            assert row["within_batch_update"] == (row["step"] - 1) % 16 + 1
            assert row["prompt_ids"] == list(range(16 * (b - 1), 16 * b))
            assert row["normalizer"] == 16384
            assert row["actual_tokens"] == sum(len(byid[i]["response_ids"]) for i in row["prompt_ids"])
            assert all(math.isfinite(row[k]) for k in ["loss", "grad_norm"])
            visits.update(row["prompt_ids"])
        assert set(visits) == set(range(320)) and set(visits.values()) == {16}
        initial = {r["id"]: r for r in read(dest / "fixed-00.json")}
        q_change = 0.0
        for b in range(21):
            rows = read(dest / f"fixed-{b:02d}.json")
            assert len(rows) == LONG_PASS.examples and {r["id"] for r in rows} == set(range(LONG_PASS.examples))
            for row in rows:
                assert row["completed_batches"] == b and row["step"] == 16 * b
                assert row["length"] == len(byid[row["id"]]["response_ids"])
                assert row["stats"]["all"]["tokens"] == row["length"]
                assert all(math.isfinite(v) for band in row["stats"].values() for v in band.values())
                q_change = max(q_change, abs(row["stats"]["all"]["hq"] - initial[row["id"]]["stats"]["all"]["hq"]))
        assert q_change < 1e-5
        probes = read(dest / "update-probes.json")
        assert [r["training_batch"] for r in probes] == list(range(1, 21))
        for r in probes:
            assert r["diagnostic_groups"] == ["M01", "M02", "M03", "M04"]
            assert abs(r["baseline_loss_consistency"]) < 5e-6 and r["parameter_step_norm"] > 0
            assert abs(r["held_after"] - r["held_before"] - r["finite_held_change"]) < 1e-12
            assert abs(r["raw_train_held_cosine"]) <= 1.001 and abs(r["descent_alignment"]) <= 1.001
        for b in [0, 5, 10, 15, 20]:
            rows = read(dest / f"fresh-{b:02d}.json")
            assert len(rows) == 64
            expected = {(i, s + 1000 * (i % 2)) for i in range(320, 352) for s in [77101, 77102]}
            assert {(r["id"], r["seed"]) for r in rows} == expected
            assert all(
                r["mode"] == "student" and r["completed_batches"] == b and r["cap"] == LONG_PASS.horizon for r in rows
            )
            assert all(r["stats"]["all"]["alive"] == len(r["response_ids"]) for r in rows)
        checkpoints = []
        for b in [1, 5, 10, 15, 20]:
            target = a.cache / "checkpoints" / teacher / f"batch-{b:02d}"
            meta = read(target / "complete.json")
            assert meta["replica_max_difference"] == 0
            payload = torch.load(target / "optimizer.pt", map_location="cpu", weights_only=True, mmap=True)
            steps = {float(v["step"]) for v in payload["optimizer"]["state"].values()}
            assert steps == {16.0 * b}
            assert payload["completed_batches"] == b and payload["step"] == 16 * b
            checkpoints.append(dict(batch=b, optimizer_step=16 * b, replicas_equal=True))
            del payload
        results.append(
            dict(
                teacher=teacher,
                logical_updates=320,
                one_contiguous_visit_per_batch=True,
                gradient_visits_per_training_prompt=16,
                holdout_optimizer_visits=0,
                fixed_evaluation_rows=21 * LONG_PASS.examples,
                probe_steps=20,
                fresh_trajectories=5 * 64,
                maximum_teacher_entropy_drift=q_change,
                checkpoints=checkpoints,
                clipped_updates=sum(r["grad_norm"] > 1 for r in history),
                actual_training_tokens=sum(r["actual_tokens"] for r in history),
            )
        )
    left = read(root / "justrl/fresh-00.json")
    right = read(root / "polaris/fresh-00.json")
    assert [(r["id"], r["seed"], r["response_ids"]) for r in left] == [
        (r["id"], r["seed"], r["response_ids"]) for r in right
    ]
    output = dict(
        passed=True,
        unique_prompts=LONG_PASS.examples,
        training_batches=20,
        permanent_holdout_batches=20,
        total_logical_optimizer_updates=LONG_PASS.examples,
        initial_fresh_paths_identical=True,
        teachers=results,
        note=(
            "Held gradients are cleared before train gradients; only train "
            "gradients reach optimizer.step in the audited implementation."
        ),
    )
    (root / "verification.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
