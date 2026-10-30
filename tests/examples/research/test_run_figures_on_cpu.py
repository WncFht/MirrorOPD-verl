# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Figure data preserves frozen identities, budget limits and risk denominators."""

import json

import pytest

from examples.mirroropd.research.studies.fixed_transfer.figure_data import load_run


def fixture_run(tmp_path):
    (tmp_path / "protocol.json").write_text(json.dumps(dict(horizon=1024, reuse=16, batch_size=16)))
    groups = ["A", "B", "M01", "M02", "M03", "M04", "Code01", "Code02", "IF01", "IF02"]
    for step in [0, 16, 32, 336]:
        rows = []
        for group in groups:
            initial = (
                2 if group.startswith("M") else 4 if group.startswith("Code") else 8 if group.startswith("IF") else 1
            )
            value = initial - step / 64
            for _ in range(16):
                rows.append(
                    dict(
                        id=len(rows),
                        group=group,
                        source="frozen",
                        length=512,
                        step=step,
                        stats={
                            "all": {"tokens": 512, "rkl": value * 1024},
                            "early": {"tokens": 256, "rkl": value * 256},
                        },
                    )
                )
        (tmp_path / f"fixed-{step:03d}.json").write_text(json.dumps(rows))
    return dict(
        initial_source=str(tmp_path / "fixed-000.json"), final_source=str(tmp_path / "fixed-032.json"), budget=32
    )


def test_figures_use_fixed_h_and_observed_position_denominators(tmp_path):
    data = load_run(fixture_run(tmp_path))
    assert list(data["snapshots"]) == [0, 16, 32]
    assert data["panels"]["Training scored"] == ["A", "B"]
    assert "Old A-D" not in data["panels"]
    held = next(r for r in data["curves"] if r["step"] == 0 and r["panel"] == "All held-out")
    assert held["value"] == 4  # prompt weighting: (64*2 + 32*4 + 32*8)/128
    assert all(r["decrease"] == 0.25 for r in data["effects"])
    assert data["forgetting"][0]["forgetting"] == -0.25
    assert data["forgetting"][1]["forgetting"] == 0
    position = next(r for r in data["positions"] if r["panel"] == "Math" and r["step"] == 0)
    assert position["rkl"] == 2 and position["tokens"] == 64 * 256
    assert {r["relation"] for r in data["effects"]} == {"current", "seen", "not_yet_seen", "held_out"}


def test_rejects_changed_frozen_state_and_duplicate_prompts(tmp_path):
    spec = fixture_run(tmp_path)
    path = tmp_path / "fixed-016.json"
    rows = json.loads(path.read_text())
    rows[0]["source"] = "another-bank"
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="identity changed"):
        load_run(spec)
    rows[0]["source"] = "frozen"
    rows.append(rows[0])
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="Duplicate prompt"):
        load_run(spec)


def test_missing_readout_is_not_interpolated(tmp_path):
    spec = fixture_run(tmp_path)
    (tmp_path / "fixed-016.json").unlink()
    data = load_run(spec)
    assert not data["effects"] and not data["forgetting"]
    assert {r["step"] for r in data["curves"]} == {0, 32}
    spec["budget"] = 336
    with pytest.raises(ValueError, match="budget"):
        load_run(spec)
