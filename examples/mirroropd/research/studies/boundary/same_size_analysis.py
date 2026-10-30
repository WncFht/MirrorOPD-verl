# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Audit R1 replay-plus-extension evidence without counting replay as new science."""

import argparse
import math
from pathlib import Path

from examples.mirroropd.research.core.artifacts import file_hash, fingerprint
from examples.mirroropd.research.core.io import dump, read, write_csv
from examples.mirroropd.research.core.statistics import paired_mean_difference
from examples.mirroropd.research.studies.boundary.fast_train import extension_plan
from examples.mirroropd.research.studies.boundary.native import bank
from examples.mirroropd.research.studies.fixed_transfer.analysis import PANELS


def analyze(plan, partial=False):
    spec = read(plan)
    replica = spec.get("execution_condition") == "deterministic_from_initial"
    output = Path(spec["output"])
    (output / "analysis/verification.json").unlink(missing_ok=True)
    required = [r for r in bank("r1") if r["group"] in spec["scoring"]["groups"]]
    ids = [r["id"] for r in required]
    panels = dict(PANELS, train_all=list("ABCDEFGHIJKLMNOPQRST"))
    panels["held_all"] = panels["held_math"] + panels["held_code"] + panels["held_if"]
    results, verified, missing = [], [], []
    for name, arm in spec["arms"].items():
        (output / "analysis" / f"verified-{name}.json").unlink(missing_ok=True)
        extension_plan(plan, name)
        root = output / "arms" / name
        if not (root / "complete.json").exists():
            missing.append(name)
            continue
        done = read(root / "complete.json")
        protocol = read(root / "protocol.json")
        assert fingerprint(protocol["extension_plan"]) == fingerprint(spec)
        assert protocol["optimizer_exports"] and protocol["weights_dtype"] == "FP32"
        if replica:
            assert protocol["flash_attention_deterministic"] and protocol["resume"]["fresh"]
        assert done["plan_identity"] == fingerprint(spec) and done["stop"] == 320
        assert done["start"] == arm["start"] and done["new_updates"] == (
            320 if replica else 320 - arm["prior_completed_stop"]
        )
        history = read(root / "training.json")
        assert [r["step"] for r in history] == list(range(arm["start"] + 1, 321))
        assert all(r["group"] == chr(65 + (r["step"] - 1) // 16) for r in history)
        assert all(math.isfinite(r[k]) for r in history for k in ["loss", "grad_norm"])
        snapshots = {}
        for step in sorted({0, *range(arm["start"], 321, 16)}):
            rows = read(root / f"fixed-{step:03d}.json")
            assert [r["id"] for r in rows] == ids
            assert all(
                r["source"] == b["source"] and r["length"] == len(b["response_ids"]) and r["group"] == b["group"]
                for r, b in zip(rows, required, strict=True)
            )
            assert all(math.isfinite(r["stats"]["all"]["rkl"]) for r in rows)
            snapshots[step] = {r["id"]: r for r in rows}
        block_error = 0.0
        entropy_error = max(
            abs(row["stats"]["all"]["hq"] - snapshots[0][i]["stats"]["all"]["hq"])
            for snapshot in snapshots.values()
            for i, row in snapshot.items()
        )
        assert entropy_error < 1e-4
        for record in history:
            if (record["step"] - 1) % 16 == 0:
                rows = snapshots[record["step"] - 1].values()
                loss = sum(r["stats"]["all"]["rkl"] for r in rows if r["group"] == record["group"]) / (16 * 1024)
                block_error = max(block_error, abs(record["loss"] - loss))
        assert block_error < 1e-5
        old = Path(spec["legacy_output"]) / "arms" / name
        old_history = read(old / "training.json")
        old_steps = {r["step"] for r in old_history}
        matched = [r for r in history if r["step"] in old_steps]
        assert [r["step"] for r in matched] == [r["step"] for r in old_history]
        loss_error = max(abs(a["loss"] - b["loss"]) for a, b in zip(matched, old_history, strict=True))
        reference = read(old / f"fixed-{arm['replay_through']:03d}.json")
        current = snapshots[arm["replay_through"]]
        risk_error = max(
            abs(r["stats"]["all"]["rkl"] - current[r["id"]]["stats"]["all"]["rkl"]) / 1024 for r in reference
        )
        if not replica:
            assert loss_error <= spec["replay_acceptance"]["max_loss_absolute_tolerance"]
            assert risk_error <= spec["replay_acceptance"]["per_prompt_rkl_absolute_tolerance"]
        else:
            gate = read(root / "deterministic-gate.json")
            assert gate["unequal_gradient_tensors"] == 0 and gate["scientific_updates"] == 0
            assert gate["candidate_loss"] == gate["repeat_loss"]
            assert abs(gate["candidate_loss"] - gate["reference_loss"]) < 1e-5
            assert gate["relative_l2"] < 0.01 and gate["cosine"] > 0.9999
        replay = read(root / "replay-acceptance.json")
        assert replay["optimizer_reset"] is False and replay["step"] == arm["replay_through"]
        if replica:
            assert replay["reference_policy"] == "diagnostic" and replay["exact_old_trajectory_claim"] is False
        assert math.isclose(replay["loss_max_error"], loss_error, abs_tol=1e-12)
        assert math.isclose(replay["risk_max_error"], risk_error, abs_tol=1e-12)
        weights = Path(done["weights"])
        manifest, state = read(weights / "complete.json"), read(root / "resume/complete.json")
        assert manifest["step"] == state["step"] == 320 and manifest["optimizer_saved"] is True
        assert manifest["weight_identity"] == state["weight_identity"]
        assert state["signature"] == fingerprint(
            dict(
                models=spec["models"],
                training=spec["training"],
                teacher=arm["teacher"],
                bank=fingerprint(required),
                **(dict(execution_condition=spec["execution_condition"]) if replica else {}),
            )
        )
        assert manifest["weight_identity"] == fingerprint(
            dict(weights=manifest["files"], config=file_hash(weights / "config.json"))
        )
        for f in manifest["files"]:
            assert Path(f["name"]).name == f["name"]
            assert (weights / f["name"]).stat().st_size == f["bytes"]
        assert (root / "resume/optimizer.pt").stat().st_size == state["bytes"]
        for start, stop in [(0, 320), (arm["prior_completed_stop"], 320), (192, 256), (256, 320)]:
            for panel, groups in panels.items():
                chosen = [i for i in ids if snapshots[start][i]["group"] in groups]
                for metric in ["rkl", "js"]:
                    value = paired_mean_difference(
                        *[[snapshots[s][i]["stats"]["all"][metric] / 1024 for i in chosen] for s in [start, stop]]
                    )
                    results.append(
                        dict(
                            arm=name,
                            teacher=arm["teacher"],
                            panel=panel,
                            metric=metric,
                            before_step=start,
                            after_step=stop,
                            prompts=len(chosen),
                            drop_percent=100 * (1 - value["after"] / value["before"]),
                            **value,
                        )
                    )
        proof = dict(
            arm=name,
            complete=True,
            scope="fixed-risk deterministic replica" if replica else "fixed-risk and replay",
            step=320,
            new_updates=arm["new_updates"],
            replay_updates=0 if replica else arm["replay_through"] - arm["start"],
            budget_increment_over_prior=320 - arm["prior_completed_stop"],
            replay_risk_max_error=risk_error,
            replay_loss_max_error=loss_error,
            block_loss_max_error=block_error,
            weight_identity=manifest["weight_identity"],
            teacher_entropy_max_error=entropy_error,
            payload_hashes_rechecked=False,
            accuracy_verified=False,
            plan_identity=fingerprint(spec),
            bank_identity=fingerprint(required),
            evidence=[
                dict(path=str(p), sha256=file_hash(p))
                for p in [
                    root / "complete.json",
                    root / "protocol.json",
                    *([root / "deterministic-gate.json"] if replica else []),
                    root / "training.json",
                    root / "replay-acceptance.json",
                    *sorted(root.glob("fixed-*.json")),
                    weights / "complete.json",
                    weights / "config.json",
                    root / "resume/complete.json",
                    old / "training.json",
                    old / f"fixed-{arm['replay_through']:03d}.json",
                ]
            ],
        )
        dump(output / "analysis" / f"verified-{name}.json", proof)
        verified.append(proof)
    if missing and not partial:
        raise ValueError(f"Unfinished R1 extensions: {missing}")
    if results:
        write_csv(output / "analysis/risks.csv", results)
    dump(
        output / "analysis/verification.json",
        dict(
            complete=not missing,
            verified=verified,
            missing=missing,
            new_scientific_updates=sum(r["new_updates"] for r in verified),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    analyze(args.plan, args.partial)


if __name__ == "__main__":
    main()
