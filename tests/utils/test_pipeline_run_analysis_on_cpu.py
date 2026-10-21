# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Protect comparisons from sparse W&B rows, mismatched steps and overlapping timers."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "examples/mirroropd/docs/scripts"


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    modules = []
    for name in ("analyze_run", "compare_pipeline_runs"):
        spec = importlib.util.spec_from_file_location(f"pipeline_test_{name}", SCRIPTS / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    return modules


def row(step, seconds=10, tokens=640):
    return {
        "training/global_step": step,
        "_timestamp": 100 + step * 20,
        "timing_s/step": seconds,
        "timing_s/gen": seconds / 2,
        "perf/total_num_tokens": tokens,
        "perf/throughput": tokens / seconds / 8,
        "prompt_length/mean": 1,
        "response_length/mean": tokens / 64 - 1,
        "timing_s/testing": None,
        "timing_s/save_checkpoint": None,
    }


def gpu_row(stamp=100, value=50):
    return {
        "_timestamp": stamp,
        **{f"system.gpu.{i}.gpu": value for i in range(8)},
        **{f"system.gpu.{i}.memoryAllocatedBytes": 1024**3 for i in range(8)},
    }


def run(rows):
    return {"train": rows, "source": {}, "config": {}, "hardware": {}, "system": [gpu_row()]}


def test_sparse_summary_accounts_for_shared_teacher_without_double_counting_rollout(modules, tmp_path):
    analyze, _ = modules
    r = row(2, seconds=15)
    r.update(
        {
            "timing_s/gen": 6,
            "timing_s/teacher_forward": 3,
            "timing_s/teacher_transfer": 2,
            "timing_s/update_actor": 3,
            "timing_s/rollout_batch": 100,
        }
    )
    analyze.write_rows(tmp_path / "history.jsonl", [row(1), r, {"timing_s/step": None}])
    analyze.summarize(tmp_path, 8, 64)
    result = json.loads((tmp_path / "analysis.json").read_text())
    assert result["training_rows"] == 2
    assert result["observed_testing_seconds"] == 0
    assert result["stage_seconds"] == {
        "gen": 6,
        "teacher_forward": 3,
        "teacher_transfer": 2,
        "update_actor": 3,
        "other": 1,
    }
    assert "save_checkpoint" not in result["stage_seconds"]


def test_checkpoint_time_is_amortized_over_all_selected_steps(modules, tmp_path):
    analyze, _ = modules
    a, b = row(2), row(3)
    a["timing_s/save_checkpoint"] = 4
    analyze.write_rows(tmp_path / "history.jsonl", [a, b])
    analyze.summarize(tmp_path, 8, 64)
    result = json.loads((tmp_path / "analysis.json").read_text())
    assert result["stage_seconds"]["save_checkpoint"] == 2
    assert result["stage_seconds"]["other"] == 3


def test_throughput_uses_ratio_of_sums(modules):
    _, comparison = modules
    out = comparison.group_stats([row(2, 1, 100), row(3, 3, 900)])
    assert out["valid_tokens_per_second"] == 250  # mean(100/1, 900/3) would be 200.
    assert out["valid_tokens_per_gpu_second"] == 31.25


def test_only_matching_post_warmup_steps_enter_primary_comparison(modules):
    _, comparison = modules
    out = comparison.compare(run([row(1), row(2, 2), row(3, 1)]), run([row(1), row(2, 4)]))
    assert out["matched_steps"] == [2]
    assert out["comparison"]["shared_over_baseline_step_time"] == 2
    assert out["comparison"]["shared_over_baseline_token_throughput"] == 0.5
    assert out["runs"]["shared"]["validation_seconds"] is None


def test_teacher_transfer_stays_in_one_conservative_block(modules):
    _, comparison = modules
    r = row(2, 20)
    r.update(
        {
            "_timestamp": 140,
            "timing_s/testing": 20,
            "timing_s/gen": 8,
            "timing_s/teacher_forward": 3,
            "timing_s/teacher_transfer": 2,
            "timing_s/old_log_prob": 2,
            "timing_s/adv": 1,
            "timing_s/update_actor": 2,
            "timing_s/update_weights": 1,
        }
    )
    windows = {w["phase"]: w for w in comparison.phase_windows([r], guard=1)}
    teacher = windows["teacher_block"]
    assert (teacher["approx_start"], teacher["approx_end"]) == (108, 113)
    assert (teacher["core_start"], teacher["core_end"]) == (110, 112)
    assert "teacher_transfer" not in windows
    assert windows["validation"]["approx_start"] == 120


def test_null_system_fields_do_not_become_zero_utilization(modules):
    _, comparison = modules
    valid, missing = gpu_row(), gpu_row(stamp=115, value=None)
    out = comparison.system_stats([valid, missing])["groups"]["all_eight"]
    assert out["busy_percent"]["n"] == 1
    assert out["busy_percent"]["mean"] == 50
    valid["system.gpu.4.gpu"] = None
    out = comparison.system_stats([valid, missing])
    assert out["groups"]["all_eight"]["busy_percent"] is None
    assert out["per_gpu"]["0"]["busy_percent"]["n"] == 1


def test_duplicate_global_steps_and_overlapping_timers_are_rejected(modules):
    _, comparison = modules
    with pytest.raises(ValueError, match="Duplicate"):
        comparison.training_rows([row(2), row(2)])
    r = row(2)
    r["timing_s/update_actor"] = 10
    with pytest.raises(ValueError, match="timers exceed"):
        comparison.group_stats([r])
