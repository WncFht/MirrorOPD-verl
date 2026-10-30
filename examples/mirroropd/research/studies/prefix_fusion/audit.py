# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Scientific readout completeness is separate from successful optimizer updates."""

from collections import Counter

from examples.mirroropd.research.core.artifacts import file_hash
from examples.mirroropd.research.core.io import read
from examples.mirroropd.research.studies.prefix_fusion.protocol import protocol_identity


def audit_readouts(run, pair, *, smoke=False):
    errors = []
    outers = [0, 1, 2] if smoke else list(range(0, 41, 5))
    for outer in outers:
        dest = run / "science" / f"outer-{outer:03d}"
        try:
            receipt = read(dest / "complete.json")
            if (receipt["outer"], receipt["optimizer_step"], receipt["protocol"], receipt["smoke"]) != (
                outer,
                outer * 16,
                protocol_identity(),
                smoke,
            ):
                errors.append(f"Outer {outer}: wrong evaluation identity")
            fixed = read(dest / "fixed/complete.json")
            rows = read(dest / "fixed/per-question.json")
            if receipt["fixed_receipt"] != file_hash(dest / "fixed/complete.json"):
                errors.append(f"Outer {outer}: changed fixed-risk receipt")
            if fixed["metrics_sha256"] != file_hash(dest / "fixed/per-question.json"):
                errors.append(f"Outer {outer}: changed risk scores")
            if Counter(r["source"] for r in rows) != {"S0": 2 if smoke else 128, pair: 2 if smoke else 128}:
                errors.append(f"Outer {outer}: missing shared-state scores")
            if fixed["optimizer_updates"] != 0 or fixed["inputs"]["horizon"] != (32 if smoke else 1024):
                errors.append(f"Outer {outer}: wrong scoring horizon/update contract")
            if smoke:
                state = read(dest / "weights/state-audit.json")
                if state["before"] != state["after"]:
                    errors.append(f"Outer {outer}: actor optimizer/RNG changed")
            elif outer in {0, 20, 40}:
                long = read(dest / "long/complete.json")
                if long["questions"] != 64 or long["inputs"]["horizon"] != 7168:
                    errors.append(f"Outer {outer}: incomplete long-state scores")
                if long["metrics_sha256"] != file_hash(dest / "long/per-question.json"):
                    errors.append(f"Outer {outer}: changed long-state scores")
            if outer == (2 if smoke else 40) or (not smoke and outer == 20):
                accuracy = read(dest / "accuracy/complete.json")
                expected_questions = 2 if smoke else 630 if outer == 40 else 64
                if accuracy["records"] != expected_questions * 4 or accuracy["questions"] != expected_questions:
                    errors.append(f"Outer {outer}: incomplete task evaluation")
                for name, key in [
                    ("raw-responses.jsonl", "raw_sha256"),
                    ("records.jsonl", "records_sha256"),
                    ("metrics.json", "metrics_sha256"),
                ]:
                    if accuracy[key] != file_hash(dest / "accuracy" / name):
                        errors.append(f"Outer {outer}: changed task evidence {name}")
            retain = smoke or outer in {20, 40}
            if receipt["retained_weights"] != retain:
                errors.append(f"Outer {outer}: incorrect readout retention")
            if retain and not list((dest / "weights/huggingface").glob("*.safetensors")):
                errors.append(f"Outer {outer}: missing persistent FP32 weights")
        except (OSError, KeyError, ValueError) as exc:
            errors.append(f"Outer {outer}: missing/invalid scientific artifact: {exc}")
    checkpoint = run / "runtime" / f"global_step_{2 if smoke else 40}"
    for name in ["data.pt", "trainer_state.json"]:
        if not (checkpoint / name).exists():
            errors.append(f"Missing resumable state: {name}")
    if not list((checkpoint / "actor").glob("optim_world_size_*.pt")):
        errors.append("Missing resumable Adam state")
    return errors
