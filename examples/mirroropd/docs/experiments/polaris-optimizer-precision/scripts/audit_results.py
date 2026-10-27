#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent checks of split separation, paired views and completed updates."""

import argparse
import json
import math
from pathlib import Path


def read(p):
    return json.loads(p.read_text())


p = argparse.ArgumentParser()
p.add_argument("--data", type=Path, required=True)
a = p.parse_args()
root = a.data
records = read(root / "records.json")
assert len(records) == 120 and len({r["uid"] for r in records}) == 120
test = [r for r in records if r["role"] == "test"]
testids = {r["id"] for r in test}
assert len(test) == 80 and len(testids) == 64
trainids = {r["id"] for r in records if r["role"] in ["train", "warm_train", "calibration"]}
assert not trainids & testids
assert {tuple(r["prompt_ids"]) for r in records if r["id"] in trainids}.isdisjoint(
    {tuple(r["prompt_ids"]) for r in test}
)
seen = []
updates = 0
for folder in sorted(root.glob("*-fold*")):
    if not folder.is_dir():
        continue
    cfg = read(folder / "protocol.json")
    hist = read(folder / "training.json")
    ev = read(folder / "evaluation.json")
    done = read(folder / "complete.json")
    assert done["finished"] and done["steps"] == cfg["steps"] == len(hist)
    assert [r["step"] for r in hist] == list(range(1, cfg["steps"] + 1))
    assert cfg["parameters"] == 1777088000 and not set(cfg["train_ids"]) & testids
    assert all(
        math.isfinite(r[k])
        for r in hist
        for k in ["grad_norm", "step_l2", "pre_update_kl", "bf16_visible_fraction", "bf16_step_error_ratio"]
    )
    baseline = {(r["uid"], r["view"], r["region"]): r for r in ev if r["step"] == 0}
    assert all((r["uid"], r["view"], r["region"]) in baseline for r in ev)
    for view in ["fp32", "bf16", "rounded_fp32"]:
        final = [
            r
            for r in ev
            if r["step"] == cfg["steps"] and r["role"] == "test" and r["view"] == view and r["region"] == "all"
        ]
        assert len(final) == 80 and {r["uid"] for r in final} == {r["uid"] for r in test}
    assert all(math.isfinite(r["rkl"]) and r["rkl"] >= -1e-6 and 0 <= r["tv"] <= 1.00001 for r in ev)
    if cfg["calibration"] is not None:
        c = cfg["calibration"]
        assert abs(c["selected"]["drift"] / c["target"] - 1) < 0.015
    updates += len(hist)
    seen.append(
        dict(
            arm=folder.name,
            steps=len(hist),
            clipped=sum(r["grad_norm"] > 1 for r in hist),
            peak_gib=max(r["peak_gib"] for r in hist),
        )
    )
assert len(seen) == 22 and updates == 592, (len(seen), updates)
base = read(root / "baseline-initial.json")
fp = {(r["uid"], r["region"]): r for r in base if r["view"] == "fp32"}
rounding_identity = max(abs(r["rkl"] - fp[(r["uid"], r["region"])]["rkl"]) for r in base if r["view"] == "rounded_fp32")
assert rounding_identity < 1e-7
warm = read(root / "baseline-warm64.json")
rounded = read(root / "baseline-warm-rounded.json")
w = {(r["uid"], r["region"]): r for r in warm if r["view"] == "rounded_fp32"}
matched = max(abs(r["rkl"] - w[(r["uid"], r["region"])]["rkl"]) for r in rounded if r["view"] == "fp32")
assert matched < 1e-6
teacher = read(root / "teacher-precision.json")
weighted = read(root / "teacher-precision-student-weighted.json")
assert len(teacher) == 80 and len(weighted) == 24
result = dict(
    complete_arms=len(seen),
    optimizer_steps=updates,
    heldout_records=80,
    heldout_unique_prompts=64,
    no_train_test_prompt_overlap=True,
    paired_evaluation_views=True,
    cold_rounding_identity_error=rounding_identity,
    warm_function_preserving_start_error=matched,
    teacher_precision_records=80,
    teacher_student_weighted_records=24,
    arms=seen,
)
(root / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
