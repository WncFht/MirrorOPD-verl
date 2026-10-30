# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Audit completed conditional-support R1 stages without claiming task accuracy is complete."""

import argparse
import math
from pathlib import Path

from examples.mirroropd.assets.paths import resolve_location
from examples.mirroropd.research.core.artifacts import file_hash, fingerprint
from examples.mirroropd.research.core.io import dump, read, write_csv
from examples.mirroropd.research.core.statistics import paired_mean_difference
from examples.mirroropd.research.studies.boundary import small_to_large as run

PANELS = {
    "train_pool": list("ABCDEFGHIJKLMNOPQRST"),
    "old_ad": list("ABCD"),
    "held_math": ["M01", "M02", "M03", "M04"],
    "held_code": ["Code01", "Code02"],
    "held_if": ["IF01", "IF02"],
    "held_all": ["M01", "M02", "M03", "M04", "Code01", "Code02", "IF01", "IF02"],
}


def analyze(plan, partial=False):
    run.configure(plan)
    output = run.OUT / "analysis"
    bank = run.rows()
    expected_ids = [r["id"] for r in bank]
    results, verified, missing = [], [], []
    numerical = read(run.OUT / "numerical/complete.json")
    assert numerical["bank_identity"] == fingerprint(bank)
    assert numerical["flash_attention_deterministic"] == run.FA_DETERMINISTIC
    for probe in numerical["probes"]:
        assert abs(probe["reference_loss"] - probe["candidate_loss"]) < 1e-5
        assert probe["relative_l2"] < 0.01 and probe["cosine"] > 0.9999
    for arm in run.ARMS:
        root = run.OUT / "arms" / arm
        if not (root / "complete.json").exists():
            missing.append(arm)
            continue
        saved = read(root / "complete.json")
        assert saved["updates"] == run.UPDATES and saved["support"] == run.SUPPORT
        assert saved["inputs"] == dict(signature=run.resume_signature(arm), updates=run.UPDATES)
        initial_acceptance = read(root / "initial-replay-acceptance.json")
        assert initial_acceptance["prompts"] == 256 and max(initial_acceptance["max_errors"].values()) < 1e-8
        history = read(root / "training.json")
        if arm in run.RESUME_FROM:
            origin = Path(run.RESUME_FROM[arm])
            source = read(origin / "resume/complete.json")
            probe = read(origin / "resume/gpu-verified.json")
            assert saved["resume_from"] == str(origin) and saved["replay_from_initial"] is False
            assert saved["new_updates"] == run.UPDATES - source["step"]
            assert source["signature"] == fingerprint(run.resume_signature(arm))
            assert all(probe[k] == source[k] for k in ["step", "signature", "weight_identity"])
            assert probe["exact_loaded_values"] and probe["rng_exact"] and probe["scientific_updates"] == 0
            assert history[: source["step"]] == read(origin / "training.json")
            for step in range(0, source["step"] + 1, 16):
                assert file_hash(root / f"fixed-{step:03d}.json") == file_hash(origin / f"fixed-{step:03d}.json")
        assert [r["step"] for r in history] == list(range(1, run.UPDATES + 1))
        assert all(r["group"] == run.training_group(r["step"]) for r in history)
        assert all(math.isfinite(r[k]) for r in history for k in ["loss", "grad_norm"])
        snapshots = {}
        max_entropy_error, max_block_error = 0.0, 0.0
        for step in range(0, run.UPDATES + 1, 16):
            rows = read(root / f"fixed-{step:03d}.json")
            assert [r["id"] for r in rows] == expected_ids
            assert all(
                r["source"] == b["source"] and r["length"] == len(b["response_ids"]) and r["group"] == b["group"]
                for r, b in zip(rows, bank, strict=True)
            )
            snapshots[step] = rows
            max_entropy_error = max(
                max_entropy_error,
                max(
                    abs(r["stats"]["all"]["hq"] - a["stats"]["all"]["hq"])
                    for r, a in zip(rows, snapshots[0], strict=True)
                ),
            )
            if step < run.UPDATES:
                group = run.training_group(step + 1)
                loss = sum(r["stats"]["all"]["rkl"] for r in rows if r["group"] == group) / (16 * 1024)
                max_block_error = max(max_block_error, abs(loss - history[step]["loss"]))
        assert max_entropy_error < 1e-4 and max_block_error < 1e-5
        weights = resolve_location(saved["weights"])
        manifest = read(weights / "complete.json")
        assert manifest["step"] == run.UPDATES and manifest["support"] == run.SUPPORT
        assert read(weights / "config.json")["vocab_size"] == 152064
        assert manifest["weight_identity"] == fingerprint(
            dict(weights=manifest["files"], config=file_hash(weights / "config.json"))
        )
        for item in manifest["files"]:
            assert Path(item["name"]).name == item["name"]
            assert (weights / item["name"]).stat().st_size == item["bytes"]
        resume = read(root / "resume/complete.json")
        assert resume["step"] == run.UPDATES and resume["weight_identity"] == manifest["weight_identity"]
        assert resume["signature"] == fingerprint(run.resume_signature(arm))
        if not resume.get("optimizer_pruned"):
            assert (root / "resume/optimizer.pt").stat().st_size == resume["bytes"]
        window = 64 if run.UPDATES >= 320 else 16
        for panel, groups in PANELS.items():
            for metric in ["rkl", "js"]:
                for start in [0, run.UPDATES - window]:
                    values = [
                        [r["stats"]["all"][metric] / 1024 for r in snapshots[step] if r["group"] in groups]
                        for step in [start, run.UPDATES]
                    ]
                    result = paired_mean_difference(*values)
                    results.append(
                        dict(
                            arm=arm,
                            panel=panel,
                            metric=metric,
                            before_step=start,
                            after_step=run.UPDATES,
                            drop_percent=100 * (1 - result["after"] / result["before"])
                            if result["before"] > 0
                            else None,
                            **result,
                        )
                    )
        record = dict(
            arm=arm,
            updates=run.UPDATES,
            support=run.SUPPORT,
            deterministic=run.FA_DETERMINISTIC,
            max_teacher_entropy_error=max_entropy_error,
            max_block_loss_error=max_block_error,
            weights_manifest_verified=True,
            weight_payload_rehashed=False,
            student_tail_mass_per_valid_token={
                str(step): sum(r["stats"]["all"]["student_tail_mass"] for r in snapshots[step])
                / sum(r["stats"]["all"]["tokens"] for r in snapshots[step])
                for step in [0, run.UPDATES]
            },
            accuracy_verified=False,
        )
        dump(output / f"verified-{arm}.json", record)
        verified.append(record)
    if results:
        write_csv(output / "conditional-risks.csv", results)
    summary = dict(verified=verified, missing=missing, scope="conditional-RKL stages", accuracy_verified=False)
    dump(output / "summary.json", summary)
    if not partial:
        assert not missing, missing
        dump(output / "verification.json", dict(summary, complete=True))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    print(analyze(args.plan, partial=args.partial))


if __name__ == "__main__":
    main()
