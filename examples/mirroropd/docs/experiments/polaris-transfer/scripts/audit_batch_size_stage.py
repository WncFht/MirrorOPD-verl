#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Verify budget matching and first gradients against independent Gram probes."""

import argparse
import collections
import json
import math
from pathlib import Path

import numpy as np


def read(p):
    return json.loads(p.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    a = ap.parse_args()
    root = a.data
    pool, bank = read(root / "pool.json"), read(root / "bank.json")
    assert len(pool) == 512 and len(bank) == 160
    pool_ids = [r["id"] for r in pool]
    byid = {r["id"]: r for r in pool + bank}
    held = [r for r in bank if r["split"] == "heldout"]
    assert len(set(pool_ids)) == 512 and len(held) == 96
    normalize = lambda s: " ".join(s.split())
    assert len({normalize(r["prompt"]) for r in pool + held}) == 608
    arms = []
    differences = []
    for teacher in ["justrl", "polaris"]:
        for phase in ["initial", "warm128"]:
            noise = read(root / f"noise-{teacher}-{phase}.json")
            g = np.array(noise["gram"])
            assert noise["parameter_count"] == 1777088000 and g.shape == (34, 34)
            assert np.isfinite(g).all() and np.allclose(g, g.T, atol=1e-9)
            assert np.linalg.eigvalsh(g).min() > -1e-5 * g.diagonal().max()
            assert sum(noise["prompt_ids"][:32], []) == pool_ids
            for batch in [16, 64, 128, 512] if (teacher, phase) == ("polaris", "warm128") else [16, 64, 128]:
                d = root / f"{teacher}-{phase}-b{batch}"
                cfg = read(d / "protocol.json")
                train = read(d / "training.json")
                complete = read(d / "complete.json")
                expected = 4096 if batch == 512 else 2048
                assert complete == dict(finished=True, steps=expected // batch, examples=expected)
                assert len(train) == expected // batch
                assert cfg["lr"] == 1e-6
                ids = sum([r["prompt_ids"] for r in train], [])
                assert ids == pool_ids * (expected // 512)
                assert set(collections.Counter(ids).values()) == {expected // 512}
                for i, r in enumerate(train, 1):
                    assert r["step"] == i and r["examples"] == i * batch and r["normalizer"] == batch * 1024
                    assert all(math.isfinite(r[k]) for k in ["loss", "grad_norm"])
                    assert r["tokens"] == sum(len(byid[j]["response_ids"]) for j in r["prompt_ids"])
                m = batch // 16
                loss_error = train[0]["loss"] - sum(noise["losses"][:m]) / m
                grad_expected = float(g[:m, :m].mean() ** 0.5)
                grad_relative_error = abs(train[0]["grad_norm"] - grad_expected) / grad_expected
                assert abs(loss_error) < 2e-6 and grad_relative_error < 1e-3
                differences.append(
                    dict(arm=d.name, initial_loss_error=loss_error, initial_gradient_relative_error=grad_relative_error)
                )
                for f in d.glob("fixed-*.json"):
                    rows = read(f)
                    assert len(rows) == 160 and {r["id"] for r in rows} == set(range(160))
                    for r in rows:
                        assert r["length"] == len(byid[r["id"]]["response_ids"])
                        assert all(math.isfinite(v) for band in r["stats"].values() for v in band.values())
                for f in d.glob("fresh-*.json"):
                    rows = read(f)
                    assert len(rows) == 64
                    assert {(r["id"], r["seed"]) for r in rows} == {
                        (i, s) for i in range(64, 96) for s in [65101, 65102]
                    }
                    assert all(r["mode"] == "student" and r["source"].startswith("fresh-") for r in rows)
                arms.append(
                    dict(
                        arm=d.name,
                        steps=len(train),
                        examples=expected,
                        actual_tokens=sum(r["tokens"] for r in train),
                        clipped_steps=sum(r["grad_norm"] > 1 for r in train),
                        fixed_evaluations=len(list(d.glob("fixed-*.json"))),
                        fresh_evaluations=len(list(d.glob("fresh-*.json"))),
                    )
                )
    assert len(arms) == 13 and sum(r["steps"] for r in arms) == 712
    for phase in ["initial", "warm128"]:
        for source in ["bank", "fresh-initial", "fresh-warm128"]:
            suffix = "" if source == "bank" else "-" + source
            probe = read(root / f"overlap-mass-{phase}{suffix}.json")
            inputs = (
                [r for r in bank if r["group"] in ["E", "F"]]
                if source == "bank"
                else read(root / f"baseline-polaris-{source.removeprefix('fresh-')}/fresh.json")
            )
            expected = {(r["id"], r.get("seed")): len(r["response_ids"]) for r in inputs}
            assert {(r["id"], r.get("seed")): r["tokens"] for r in probe["records"]} == expected
            means = probe["means"]
            assert abs(sum(means[f"p_mass_{i}"] for i in range(4)) - 1) < 1e-6
            assert abs(sum(means[f"q_mass_{i}"] for i in range(4)) - 1) < 1e-6
            assert abs(sum(means[f"gkl_{i}"] for i in range(4)) - means["rkl"]) < 1e-6
    for source in ["fresh-initial", "fresh-warm128"]:
        first = read(root / f"overlap-mass-initial-{source}.json")["means"]
        last = read(root / f"overlap-mass-warm128-{source}.json")["means"]
        assert abs(first["hq"] - last["hq"]) < 1e-9
    out = dict(
        passed=True,
        arms=arms,
        total_updates=712,
        processed_trajectories=sum(r["examples"] for r in arms),
        all_training_LR=1e-6,
        unique_train_prompts=512,
        heldout_prompts=96,
        full_gradient_vectors=136,
        initial_gradient_consistency=differences,
        overlap_probes=6,
        limitations=[
            "Warm checkpoints restore weights only; every arm resets AdamW.",
            "Finite-pool gradient noise is conditional on frozen initial-student prefixes.",
            "One training seed per recipe. Prompt bootstrap does not cover training randomness.",
        ],
    )
    (root / "verification.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
