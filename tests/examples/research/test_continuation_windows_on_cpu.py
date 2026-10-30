# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Continuation decisions use pooled prompt risk with fixed-H normalization."""

from types import SimpleNamespace

import pytest

from examples.mirroropd.research.core.io import dump
from examples.mirroropd.research.studies.fixed_transfer.continuation import compare_windows


def test_windows_keep_short_eos_budget_and_validate_paired_states(tmp_path):
    arm = SimpleNamespace(name="test", updates=128)
    study = SimpleNamespace(
        directory=tmp_path,
        arms=[arm],
        continuation=None,
        training=SimpleNamespace(score_interval=16, horizon=1024),
    )
    root = tmp_path / "arms/test"
    dump(root / "complete.json", {"updates": 128})
    for step, risk in [(0, 1024.0), (64, 512.0), (128, 256.0)]:
        rows = [
            dict(
                id=i,
                group=group,
                source="fixed-bank",
                length=8,
                stats={"all": dict(rkl=risk, js=risk / 2, tokens=8)},
            )
            for i, group in enumerate(["M01", "Code01", "IF01"])
        ]
        dump(root / f"fixed-{step:03d}.json", rows)
    result = compare_windows(study)
    pooled = next(r for r in result if r["panel"] == "held_all" and r["before_step"] == 64 and r["metric"] == "rkl")
    assert pooled["before"] == 0.5 and pooled["after"] == 0.25
    assert pooled["prompts"] == 3 and pooled["relative_drop_percent"] == 50
    rows[0]["source"] = "different-bank"
    dump(root / "fixed-128.json", rows)
    with pytest.raises(ValueError, match="states changed"):
        compare_windows(study)
