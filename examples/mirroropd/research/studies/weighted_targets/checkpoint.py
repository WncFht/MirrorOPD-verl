# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Keep one durable full training state and a separate final HF model for E83/E84."""

import json
import os
import shutil
from pathlib import Path

from verl.trainer.ppo.checkpoint_callback import CheckpointCallback


class LatestTrainingState(CheckpointCallback):
    def on_save(self, trainer, global_step, checkpoint_dir, async_save=False, **kwargs):
        if async_save:
            raise ValueError("LatestTrainingState requires synchronous checkpoint saves")
        current = Path(checkpoint_dir).resolve()
        root = Path(self.config.trainer.default_local_dir).resolve()
        if current.parent != root or current.name != f"global_step_{global_step}":
            raise ValueError("Checkpoint is outside the configured training-state root")
        if (root / "latest_checkpointed_iteration.txt").read_text().strip() != str(global_step):
            raise ValueError("Checkpoint pointer was not durably advanced")
        actor = current / "actor"
        world = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        required = [current / "data.pt", current / "trainer_state.json"]
        required += [
            actor / f"{kind}_world_size_{world}_rank_{rank}.pt"
            for kind in ("model", "optim", "extra_state")
            for rank in range(world)
        ]
        if missing := [str(p) for p in required if not p.is_file()]:
            raise ValueError(f"Incomplete training state: {missing}")
        hf = actor / "huggingface"
        index = json.loads((hf / "model.safetensors.index.json").read_text())
        if not index["weight_map"] or any(not (hf / p).is_file() for p in index["weight_map"].values()):
            raise ValueError("Incomplete HF weight export")
        if global_step == self.config.trainer.total_training_steps:
            destination = root.parent / "final-model"
            temporary = root.parent / "final-model.partial"
            shutil.copytree(hf, temporary)
            os.rename(temporary, destination)
        for previous in root.glob("global_step_*"):
            if previous != current and previous.is_dir() and previous.name.removeprefix("global_step_").isdigit():
                if int(previous.name.removeprefix("global_step_")) < global_step:
                    shutil.rmtree(previous)
