# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Attempt/applied progress and checkpoint ABI, independent of scheduler state."""

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from verl.utils.metric.schema import finite_metrics

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OptimizerProgress:
    attempts: int | None
    applied: int | None

    def __post_init__(self):
        for name, value in [("attempts", self.attempts), ("steps", self.applied)]:
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"Invalid actor optimizer {name}: {value!r}")
        if self.attempts is not None and self.applied is not None and self.applied > self.attempts:
            raise ValueError("Actor optimizer steps cannot exceed attempts")

    def advance(self, attempts, applied):
        delta = OptimizerProgress(attempts, applied)
        return OptimizerProgress(
            self.attempts + delta.attempts if self.attempts is not None and delta.attempts is not None else None,
            self.applied + delta.applied if self.applied is not None and delta.applied is not None else None,
        )

    def coordinates(self, role="actor"):
        axes = ("step/update", "step/optimizer") if role == "actor" else ("step/critic_update", "step/critic_optimizer")
        return {key: value for key, value in zip(axes, (self.attempts, self.applied), strict=True) if value is not None}

    def checkpoint_fields(self, role):
        return {f"{role}_optimizer_step_attempts": self.attempts, f"{role}_optimizer_step": self.applied}

    def legacy_metrics(self, role="actor"):
        result = {f"training/{key}": value for key, value in self.checkpoint_fields(role).items() if value is not None}
        if self.attempts is not None and self.applied is not None:
            result[f"training/{role}_optimizer_skipped_updates"] = self.attempts - self.applied
        return result


def update_log_events(records, initial, outer, role="actor"):
    """Attach explicit event coordinates; never average the update-time axis."""
    progress = initial
    for record in records:
        applied = record["applied"]
        if applied is not None and type(applied) is not bool:
            raise ValueError("Invalid per-update optimizer execution fact")
        progress = progress.advance(1, None if applied is None else int(applied))
        metrics = dict(record["metrics"])
        if progress.attempts is not None and progress.applied is not None:
            metrics[f"optim/{role}/skipped_updates"] = progress.attempts - progress.applied
        values = finite_metrics(metrics) | {"step/outer": outer} | progress.coordinates(role)
        yield values, progress.attempts if progress.attempts is not None else outer


def save_optimizer_progress(checkpoint_dir, global_step, actor, critic=None):
    state = {"schema_version": 1, "global_step": global_step, **actor.checkpoint_fields("actor")}
    if critic is not None:
        state.update(critic.checkpoint_fields("critic"))
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=checkpoint_dir, prefix="trainer_state-", delete=False) as f:
            temporary_path = f.name
            json.dump(state, f, indent=2, allow_nan=False)
            f.write("\n")
        os.replace(temporary_path, os.path.join(checkpoint_dir, "trainer_state.json"))
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def load_optimizer_progress(checkpoint_dir, global_step):
    path = Path(checkpoint_dir) / "trainer_state.json"
    if not path.exists():
        logger.warning("Checkpoint has no trainer_state.json; absolute actor optimizer counts are unknown")
        return OptimizerProgress(None, None), OptimizerProgress(None, None)
    state = json.loads(path.read_text())
    if not isinstance(state, dict) or type(state.get("schema_version")) is not int or state["schema_version"] != 1:
        raise ValueError(f"Invalid trainer state schema in {path}")
    if type(state.get("global_step")) is not int or state["global_step"] != global_step:
        raise ValueError(f"Trainer state global_step does not match checkpoint directory: {path}")
    if "actor_optimizer_step_attempts" not in state or "actor_optimizer_step" not in state:
        raise ValueError(f"Missing actor optimizer counts in {path}")
    return tuple(
        OptimizerProgress(state.get(f"{role}_optimizer_step_attempts"), state.get(f"{role}_optimizer_step"))
        for role in ("actor", "critic")
    )
