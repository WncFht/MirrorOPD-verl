# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Driver-only rollout reuse control from globally reduced, pre-update RKL.

Decisions use completed RPCs and apply to the next rollout batch. No worker
configuration is mutated, no probe forward is added, and no rank decides alone.
"""

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path


def positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class UpdateBudget:
    total: int
    eval_every: int
    save_every: int

    def __post_init__(self):
        for name, value in asdict(self).items():
            positive_int(value, name)

    def epochs(self, completed, reuse):
        """Never cross an evaluation, checkpoint, or terminal update boundary."""
        if type(completed) is not int or not 0 <= completed < self.total:
            raise ValueError("Update budget is exhausted or progress is unknown")
        return min(
            reuse,
            self.total - completed,
            self.eval_every - completed % self.eval_every,
            self.save_every - completed % self.save_every,
        )

    def due(self, completed, *, save=False):
        return completed > 0 and (
            completed == self.total or completed % (self.save_every if save else self.eval_every) == 0
        )


@dataclass(frozen=True)
class ReuseOptions:
    mode: str = "adaptive"
    minimum: int = 4
    window: int = 5
    patience: int = 2
    threshold: float = 0.9
    warmup_updates: int = 160
    min_drop: float = 1e-8

    def __post_init__(self):
        if self.mode not in {"fixed", "adaptive"}:
            raise ValueError("Reuse mode must be fixed or adaptive")
        for name in ("minimum", "window", "patience"):
            positive_int(getattr(self, name), name)
        if self.mode == "adaptive" and self.minimum < 2:
            raise ValueError("At least two pre-update observations are required")
        if type(self.warmup_updates) is not int or self.warmup_updates < 0:
            raise ValueError("warmup_updates must be a nonnegative integer")
        for name in ("threshold", "min_drop"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {name}")


class ReuseController:
    def __init__(self, initial, options):
        positive_int(initial, "initial reuse")
        if initial < options.minimum:
            raise ValueError("Initial reuse must be at least minimum reuse")
        self.initial, self.options = initial, options
        self.current = initial
        self.previous = None
        self.drops, self.gaps = [], []
        self.high_windows = 0

    def _reset_window(self):
        self.previous = None
        self.drops.clear()
        self.gaps.clear()
        self.high_windows = 0

    def observe(self, records, *, used, completed):
        """Consume one complete batch; partial boundary batches never drive control."""
        if len(records) != used or any(record["applied"] is not True for record in records):
            raise RuntimeError("Matched-update reuse requires one known successful update per record")
        result = {"reuse/used": used, "reuse/next": self.current, "reuse/switched": 0, "reuse/valid": 0}
        losses = [record["metrics"].get("distill/teacher_rkl") for record in records]
        if any(value is None for value in losses):
            raise ValueError("Reuse observations require native distill/teacher_rkl update records")
        # One pre-update observation cannot identify a within-batch reduction.
        # Preserve the training loss, but never invent a ratio for online OPD.
        if used == 1 and math.isfinite(losses[0]):
            result.update({"reuse/first_rkl": float(losses[0]), "reuse/s_defined": 0})
        if (
            used < 2
            or used != self.current
            or completed - used < self.options.warmup_updates
            or any(not math.isfinite(value) for value in losses)
        ):
            self._reset_window()
            return result
        first, end = float(losses[0]), float(losses[-1])
        result.update({"reuse/first_rkl": first, "reuse/last_pre_update_rkl": end})
        if self.previous is not None:
            old_first, old_end = self.previous
            drop, gap = old_first - old_end, first - old_end
            self.drops.append(drop)
            self.gaps.append(gap)
            result.update({"reuse/drop": drop, "reuse/refresh_gap": gap, "reuse/fresh_progress": drop - gap})
        self.previous = (first, end)
        if len(self.drops) == self.options.window:
            denominator = math.fsum(self.drops)
            if denominator > self.options.min_drop * self.options.window:
                ratio = math.fsum(self.gaps) / denominator
                result.update({"reuse/valid": 1, "reuse/s_defined": 1, "reuse/gap_over_drop": ratio})
                self.high_windows = (
                    min(self.options.patience, self.high_windows + 1) if ratio > self.options.threshold else 0
                )
            else:
                self.high_windows = 0
            self.drops.clear()
            self.gaps.clear()
            if (
                self.options.mode == "adaptive"
                and self.high_windows >= self.options.patience
                and self.current > self.options.minimum
            ):
                self.current = max(self.options.minimum, self.current // 2)
                self._reset_window()
                result.update({"reuse/next": self.current, "reuse/switched": 1})
        return result

    def state_dict(self):
        return dict(
            initial=self.initial,
            options=asdict(self.options),
            current=self.current,
            previous=self.previous,
            drops=self.drops,
            gaps=self.gaps,
            high_windows=self.high_windows,
        )

    def load_state_dict(self, state):
        if (
            set(state) != set(self.state_dict())
            or state["initial"] != self.initial
            or state["options"] != asdict(self.options)
        ):
            raise ValueError("Reuse checkpoint configuration differs")
        allowed, value = {self.initial}, self.initial
        while value > self.options.minimum:
            value = max(self.options.minimum, value // 2)
            allowed.add(value)
        if type(state["current"]) is not int or state["current"] not in allowed:
            raise ValueError("Invalid checkpoint reuse")
        previous, drops, gaps = state["previous"], state["drops"], state["gaps"]
        if previous is not None and (not isinstance(previous, list | tuple) or len(previous) != 2):
            raise ValueError("Invalid checkpoint endpoints")
        if (
            not isinstance(drops, list)
            or not isinstance(gaps, list)
            or not len(drops) == len(gaps) < self.options.window
        ):
            raise ValueError("Invalid checkpoint window")
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in list(previous or ()) + drops + gaps):
            raise ValueError("Nonfinite checkpoint signal")
        streak = state["high_windows"]
        if type(streak) is not int or not 0 <= streak <= self.options.patience:
            raise ValueError("Invalid checkpoint patience")
        self.current, self.previous = state["current"], previous
        self.drops, self.gaps, self.high_windows = list(drops), list(gaps), streak


def build_training_control(config):
    """Admit only synchronous full-batch OPD; legacy training remains opt-in."""
    budget, options = config.trainer.get("update_budget"), config.trainer.get("reuse_control")
    if budget is None and options is None:
        return None, None
    if budget is None or options is None:
        raise ValueError("update_budget and reuse_control must be configured together")
    budget = UpdateBudget(**budget)
    actor = config.actor_rollout_ref.actor
    distillation = config.distillation.distillation_loss
    if (
        config.trainer.v1.trainer_mode != "sync"
        or config.trainer.v1.sync.get("parameter_sync_step", 1) != 1
        or config.data.train_batch_size != actor.ppo_mini_batch_size
        or config.actor_rollout_ref.rollout.n != 1
        or distillation.get("use_task_rewards", False)
        or distillation.get("use_policy_gradient", False)
        or not config.distillation.enabled
        or distillation.loss_mode != "mirror"
        or distillation.mirror.estimator != "full_vocab"
        or distillation.mirror.objective != "mdpo"
        or distillation.mirror.kappa != 0
        or distillation.mirror.eta != 1
        or actor.strategy not in {"fsdp", "fsdp2", "fsdp_turbo"}
        or actor.optim.lr_scheduler_type != "constant"
        or actor.checkpoint.get("async_save", False)
    ):
        raise ValueError(
            "Reuse control requires synchronous full-batch full-vocab OPD, constant LR and synchronous saves"
        )
    if config.trainer.total_training_steps < budget.total or config.trainer.save_freq <= 0:
        raise ValueError("Require an outer safety cap >= update budget and enabled resumable checkpoints")
    return budget, ReuseController(actor.ppo_epochs, ReuseOptions(**options))


def save_training_control(directory, budget, controller, completed):
    state = dict(version=1, budget=asdict(budget), completed=completed, controller=controller.state_dict())
    path = Path(directory) / "reuse_control.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, allow_nan=False) + "\n")
    os.replace(temporary, path)


def load_training_control(directory, budget, controller, completed):
    state = json.loads((Path(directory) / "reuse_control.json").read_text())
    if state.get("version") != 1 or state.get("budget") != asdict(budget) or state.get("completed") != completed:
        raise ValueError("Reuse checkpoint budget or optimizer progress differs")
    controller.load_state_dict(state["controller"])
