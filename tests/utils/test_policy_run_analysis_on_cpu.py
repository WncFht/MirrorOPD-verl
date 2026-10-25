# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for policy-v3 event axes, sparse data and timer accounting."""

import gzip
import importlib.util
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "examples/mirroropd/docs/experiments/scripts"
DATA = (
    Path(os.environ.get("WANDB_LOGS_REPO", ROOT.parent / "wandb-logs"))
    / "MirrorOPD-verl/analysis/e15-krkmq098-20261023"
)


@pytest.fixture
def analyze():
    spec = importlib.util.spec_from_file_location("policy_analysis", SCRIPTS / "analyze_policy_run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preupdate_axis_and_missing_values(analyze):
    rows = [
        {
            "step/outer": 1,
            "step/update": 0,
            "step/optimizer": 0,
            "policy/anchor_behavior/k3": 0.1,
            "optim/actor/loss": None,
        },
        {
            "step/outer": 1,
            "step/update": 1,
            "step/optimizer": 1,
            "policy/anchor_behavior/k3": None,
            "optim/actor/loss": 0.2,
        },
        {
            "step/outer": 1,
            "step/update": 1,
            "step/optimizer": 1,
            "policy/anchor_behavior/k3": None,
            "optim/actor/loss": None,
        },
    ]
    series = analyze.normalize(rows, {"policy/anchor_behavior/k3": "step/outer", "optim/actor/loss": "step/update"})
    assert series["policy/anchor_behavior/k3"]["points"] == [(1, 0.1)]
    assert series["optim/actor/loss"]["points"] == [(1, 0.2)]


def test_conflicting_duplicate_and_missing_axis_fail(analyze):
    with pytest.raises(ValueError, match="Conflicting"):
        analyze.normalize(
            [{"step/update": 1, "optim/loss": 0}, {"step/update": 1, "optim/loss": 1}], {"optim/loss": "step/update"}
        )
    with pytest.raises(ValueError, match="Missing axis"):
        analyze.normalize([{"optim/loss": 0}], {"optim/loss": "step/update"})


def test_timers_do_not_double_count_nested_regions(analyze):
    prefix = "perf/timing/"
    series = {}
    for name in (
        "rollout_pipeline",
        "teacher_transfer",
        "teacher_forward",
        "old_log_prob",
        "update_actor",
        "update_weights",
        "adv",
    ):
        series[prefix + name + "_seconds"] = {"points": [(1, 1), (2, 1), (3, 1)]}
    for name, points in {
        "outer_step": [(1, 15), (2, 10), (3, 12)],
        "validation": [(3, 100)],
        "save_checkpoint": [(3, 2)],
        "validation_rollout": [(3, 90)],
        "rollout_batch": [(1, 1000), (2, 1000), (3, 1000)],
    }.items():
        series[prefix + name + "_seconds"] = {"points": points}
    result = analyze.timing(series)
    assert result["accounted_seconds"] == 137
    assert result["regular_steps"] == [2]
    assert result["checkpoint_seconds_included_in_outer"] == 2


@pytest.mark.parametrize("system_name", ["system-sampled.jsonl.gz", "system.jsonl.gz"])
def test_lossless_archive_and_no_overwrite(analyze, tmp_path, system_name):
    source, target = tmp_path / "source", tmp_path / "snapshot"
    source.mkdir()
    raw = b'{"step/outer": 1, "value": null}\n'
    (source / "history.jsonl").write_bytes(raw)
    for name in ("config.json", "source.json", "hardware.json", "package-versions.txt", system_name):
        (source / name).write_bytes(b"fixture")
    analyze.snapshot(source, target)
    assert gzip.decompress((target / "history.jsonl.gz").read_bytes()) == raw
    assert (target / system_name).read_bytes() == b"fixture"
    with pytest.raises(FileExistsError):
        analyze.snapshot(source, target)


@pytest.mark.skipif(not DATA.is_dir(), reason="Optional integration evidence requires wandb-logs analysis archive")
def test_frozen_e15_axes_and_accounting(analyze):
    rows = analyze.read_rows(DATA / "history.jsonl.gz")
    axes = json.loads((DATA / "metric-axes.json").read_text())
    series = analyze.normalize(rows, axes)
    result = analyze.summarize(rows, series, [])
    assert (result["history_events"], result["outer_steps"], result["last_optimizer_step"]) == (600, 200, 200)
    assert result["numeric_metrics_excluding_axes_and_internal"] == 2377
    assert result["training"]["applied_updates"] == 200
    assert result["training"]["skipped_updates"] == 0
    assert result["timing"]["regular_count"] == 189
    assert result["timing"]["accounted_seconds"] == pytest.approx(28464.034, abs=0.01)
    assert sum(result["training"]["finish_reason_counts"].values()) == result["training"]["responses"] == 12800
    for dataset, expected in [("math500", 8000), ("amc23", 640), ("aime25", 480)]:
        assert len(result["evaluation"][dataset]) == 10
        assert all(row["response_count"] == expected for row in result["evaluation"][dataset])


def test_gpu_missing_observations_not_zero_filled(monkeypatch):
    pytest.importorskip("matplotlib")
    monkeypatch.syspath_prepend(str(SCRIPTS))
    monkeypatch.syspath_prepend(str(ROOT / "examples/mirroropd/docs/scripts"))
    # Plot styling is an optional external dependency, but data selection is tested when available.
    import fvlocate

    try:
        fvlocate.ensure()
    except FileNotFoundError:
        pytest.skip("figure-viz style package is not installed")
    import plot_policy_run

    full = {"_timestamp": 1, **{f"system.gpu.{i}.gpu": 50 for i in range(8)}}
    partial = {"_timestamp": 2, "system.gpu.0.gpu": 0}
    observed = plot_policy_run.gpu_observations([full, partial], "gpu")
    assert observed.tolist() == [[1, 50, 50, 50]]
