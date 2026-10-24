# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Budget acceptance must reject skipped updates and incomplete snapshots."""

import importlib.util
from pathlib import Path

import pytest
from omegaconf import OmegaConf

SCRIPT = Path(__file__).resolve().parents[2] / "examples/mirroropd/tools/verify_math_update_budget.py"
spec = importlib.util.spec_from_file_location("verify_budget", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def run():
    cfg = OmegaConf.create(
        {
            "trainer": {
                "v1": {"trainer_mode": "sync"},
                "total_training_steps": 3,
                "total_epochs": 1,
                "test_freq": 2,
                "resume_mode": "disable",
            },
            "data": {"train_batch_size": 64},
            "actor_rollout_ref": {"actor": {"ppo_epochs": 2, "ppo_mini_batch_size": 64}, "rollout": {"n": 1}},
            "distillation": {"enabled": True, "distillation_loss": {"loss_mode": "full_reverse_kl"}},
        }
    )
    rows = []
    for step in range(1, 4):
        for i in range(1, 3):
            u = 2 * (step - 1) + i
            rows.append(
                {
                    "step/outer": step,
                    "step/update": u,
                    "step/optimizer": u,
                    "optim/actor/update_applied": 1,
                    "optim/actor/update_skipped": 0,
                    "distill/full_kl": 1 / u,
                }
            )
        row = {
            "step/outer": step,
            "step/update": 2 * step,
            "step/optimizer": 2 * step,
            "perf/timing/outer_step_seconds": 1,
        }
        if step in (2, 3):
            row["eval/math500/acc/mean"] = 0.8
        rows.append(row)
    return cfg, rows


def test_completed_updates_and_forced_final_validation(run):
    result = module.verify(*run, 6)
    assert result["validation_optimizer_steps"] == [4, 6]
    assert result["configured_prompt_records"] == 192


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("skip", "Skipped/unconfirmed"),
        ("missing_update", "per-update"),
        ("last_outer", "completed outer"),
        ("missing_eval", "validation checkpoints"),
        ("wrong_axis", "Wrong outer"),
        ("nan", "nonfinite KL"),
        ("config", "Configured update budget"),
        ("conflict", "Conflicting update"),
    ],
)
def test_invalid_or_partial_run_cannot_pass(run, mutation, match):
    cfg, rows = run
    if mutation == "skip":
        rows[0]["optim/actor/update_applied"] = 0
    elif mutation == "missing_update":
        rows.pop(0)
    elif mutation == "last_outer":
        rows.pop()
    elif mutation == "missing_eval":
        rows[-1].pop("eval/math500/acc/mean")
    elif mutation == "wrong_axis":
        rows[0]["step/outer"] = 2
    elif mutation == "nan":
        rows[0]["distill/full_kl"] = float("nan")
    elif mutation == "config":
        cfg.actor_rollout_ref.actor.ppo_epochs = 4
    elif mutation == "conflict":
        rows.append(rows[0] | {"distill/full_kl": 10})
    with pytest.raises(ValueError, match=match):
        module.verify(cfg, rows, 6)


def test_identical_repeated_log_row_is_not_an_extra_update(run):
    cfg, rows = run
    assert module.verify(cfg, rows + [rows[0].copy()], 6)["successful_updates"] == 6
