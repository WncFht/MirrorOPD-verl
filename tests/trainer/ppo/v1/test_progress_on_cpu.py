# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
import pytest

from verl.trainer.ppo.v1.progress import (
    OptimizerProgress,
    load_optimizer_progress,
    save_optimizer_progress,
    update_log_events,
)


def test_unknown_execution_does_not_erase_attempts_or_invent_applied_steps():
    events = [{"applied": x, "metrics": {}} for x in [True, None, False, True]]
    logs = list(update_log_events(events, OptimizerProgress(10, 9), outer=4))
    assert [v["step/update"] for v, _ in logs] == [11, 12, 13, 14]
    assert logs[0][0]["step/optimizer"] == 10
    assert all("step/optimizer" not in v for v, _ in logs[1:])
    assert [step for _, step in logs] == [11, 12, 13, 14]


def test_actor_and_critic_progress_roundtrip_preserves_schema(tmp_path):
    actor, critic = OptimizerProgress(8, 7), OptimizerProgress(12, 10)
    save_optimizer_progress(tmp_path, 3, actor, critic)
    assert load_optimizer_progress(tmp_path, 3) == (actor, critic)
    assert critic.coordinates("critic") == {"step/critic_update": 12, "step/critic_optimizer": 10}
    with pytest.raises(ValueError, match="global_step"):
        load_optimizer_progress(tmp_path, 4)


def test_missing_checkpoint_remains_unknown():
    state = OptimizerProgress(None, None).advance(4, 3)
    assert state == OptimizerProgress(None, None)
    assert state.coordinates() == {}
    assert state.legacy_metrics() == {}
