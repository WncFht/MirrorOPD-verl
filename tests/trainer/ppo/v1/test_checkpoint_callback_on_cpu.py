# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from verl.trainer.ppo.checkpoint_callback import CheckpointCallback
from verl.trainer.ppo.v1.trainer_base import PPOTrainer


class _StubTrainer(PPOTrainer):
    def on_step_end(self):
        pass

    def on_sample_end(self):
        pass


class _RecordingCallback(CheckpointCallback):
    def __init__(self, config=None, events=None):
        super().__init__(config=config)
        self.events = events if events is not None else []

    def on_save(self, trainer, global_step, checkpoint_dir, async_save=False, **kwargs):
        self.events.append(("on_save", global_step, checkpoint_dir, async_save))


def _make_trainer(tmp_path, events, async_save=False):
    trainer = _StubTrainer.__new__(_StubTrainer)
    trainer.trainer_mode = "sync"
    trainer.global_steps = 3
    trainer.use_critic = False
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "default_local_dir": str(tmp_path),
                "default_hdfs_dir": None,
                "resume_mode": "auto",
                "resume_from_path": None,
                "del_local_ckpt_after_load": False,
                "checkpoint_callback_class": None,
            },
            "actor_rollout_ref": {"actor": {"checkpoint": {"async_save": async_save}}},
        }
    )
    trainer.checkpoint_callback = _RecordingCallback(config=trainer.config, events=events)
    trainer.actor_rollout_wg = MagicMock()
    trainer.actor_rollout_wg.save_checkpoint.side_effect = lambda *a, **k: events.append(("wg_save",))
    dataloader = MagicMock()
    dataloader.state_dict.return_value = {}
    trainer.train_dataloader = dataloader
    return trainer


def test_save_fires_on_save_after_worker_save(tmp_path):
    events = []
    trainer = _make_trainer(tmp_path, events)

    trainer._save_checkpoint()

    expected_dir = os.path.join(str(tmp_path), "global_step_3")
    assert events == [
        ("wg_save",),
        ("on_save", 3, expected_dir, False),
    ]
    tracker = os.path.join(str(tmp_path), "latest_checkpointed_iteration.txt")
    with open(tracker) as f:
        assert f.read() == "3"


def test_async_save_fires_on_save_with_flag(tmp_path):
    events = []
    trainer = _make_trainer(tmp_path, events, async_save=True)

    trainer._save_checkpoint()

    expected_dir = os.path.join(str(tmp_path), "global_step_3")
    assert events[-1] == ("on_save", 3, expected_dir, True)
    assert not os.path.exists(os.path.join(str(tmp_path), "latest_checkpointed_iteration.txt"))


def test_on_save_not_fired_on_worker_failure(tmp_path):
    events = []
    trainer = _make_trainer(tmp_path, events)
    trainer.actor_rollout_wg.save_checkpoint.side_effect = RuntimeError("save failed")

    with pytest.raises(RuntimeError, match="save failed"):
        trainer._save_checkpoint()

    assert events == []


def test_callback_exception_propagates(tmp_path):
    events = []
    trainer = _make_trainer(tmp_path, events)
    trainer.checkpoint_callback.on_save = MagicMock(side_effect=RuntimeError("callback failed"))

    with pytest.raises(RuntimeError, match="callback failed"):
        trainer._save_checkpoint()


def test_progress_restores_then_accumulates_different_epoch_budget(tmp_path):
    import json

    from verl.utils import tensordict_utils as tu

    trainer = _make_trainer(tmp_path, [])
    trainer.actor_optimizer_step_attempts, trainer.actor_optimizer_step = 12, 11
    trainer._save_checkpoint()
    checkpoint = tmp_path / "global_step_3"
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    assert state["actor_optimizer_step"] == 11
    resumed = _make_trainer(tmp_path, [])
    resumed._load_checkpoint()
    assert resumed.global_steps == 3
    resumed._accumulate_actor_optimizer_steps(
        tu.get_tensordict({}, {"optimizer_step_attempts": 2, "optimizer_steps": 2})
    )
    assert resumed.actor_optimizer_step_attempts == 14 and resumed.actor_optimizer_step == 13
    assert resumed._actor_optimizer_metrics()["training/actor_optimizer_skipped_updates"] == 1


def test_old_checkpoint_progress_stays_unknown_after_updates(tmp_path):
    from verl.utils import tensordict_utils as tu

    trainer = _make_trainer(tmp_path, [])
    trainer._load_actor_optimizer_progress(tmp_path)
    trainer._accumulate_actor_optimizer_steps(
        tu.get_tensordict({}, {"optimizer_step_attempts": 4, "optimizer_steps": 4})
    )
    assert trainer._actor_optimizer_metrics() == {}
    trainer._save_actor_optimizer_progress(tmp_path)
    trainer._load_actor_optimizer_progress(tmp_path)
    assert trainer.actor_optimizer_step_attempts is None and trainer.actor_optimizer_step is None


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": 2},
        {"schema_version": True},
        {"global_step": 4},
        {"global_step": 3.0},
        {"actor_optimizer_step": 13},
        {"actor_optimizer_step": -1},
        {"actor_optimizer_step": True},
        {"actor_optimizer_step_attempts": 12.0},
    ],
)
def test_corrupt_progress_is_not_silently_reset(tmp_path, change):
    import json

    trainer = _make_trainer(tmp_path, [])
    state = {"schema_version": 1, "global_step": 3, "actor_optimizer_step_attempts": 12, "actor_optimizer_step": 12}
    state.update(change)
    (tmp_path / "trainer_state.json").write_text(json.dumps(state))
    with pytest.raises(ValueError):
        trainer._load_actor_optimizer_progress(tmp_path)


@pytest.mark.parametrize("text", ["{", "[]", '{"schema_version":1,"global_step":3}'])
def test_invalid_json_or_missing_counts_is_rejected(tmp_path, text):
    trainer = _make_trainer(tmp_path, [])
    (tmp_path / "trainer_state.json").write_text(text)
    with pytest.raises(ValueError):
        trainer._load_actor_optimizer_progress(tmp_path)


def test_failed_state_publication_keeps_previous_latest_and_skips_callback(tmp_path, monkeypatch):
    events = []
    trainer = _make_trainer(tmp_path, events)
    tracker = tmp_path / "latest_checkpointed_iteration.txt"
    tracker.write_text("2")

    def fail(*args):
        raise OSError("state write failed")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError, match="state write failed"):
        trainer._save_checkpoint()
    assert tracker.read_text() == "2"
    assert events == [("wg_save",)]
    assert not list((tmp_path / "global_step_3").glob("trainer_state*"))


def test_fresh_run_starts_at_zero_and_async_save_does_not_publish_state(tmp_path):
    trainer = _make_trainer(tmp_path, [], async_save=True)
    trainer.config.trainer.resume_mode = "disable"
    trainer._load_checkpoint()
    assert trainer.actor_optimizer_step == trainer.actor_optimizer_step_attempts == 0
    trainer._save_checkpoint()
    assert not (tmp_path / "global_step_0" / "trainer_state.json").exists()
