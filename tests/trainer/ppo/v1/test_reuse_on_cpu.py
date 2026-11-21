# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Behavioral contracts for refresh decisions, exact budgets and durable state."""

import copy
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl.trainer.ppo.v1.reuse import (
    ReuseController,
    ReuseOptions,
    UpdateBudget,
    build_training_control,
    load_training_control,
    save_training_control,
)


def records(first=1.0, end=0.5, n=16):
    return [
        {"applied": True, "metrics": {"distill/teacher_rkl": first + (end - first) * j / max(1, n - 1)}}
        for j in range(n)
    ]


def observe(controller, first=1.0, end=0.5, completed=1000, used=None):
    used = controller.current if used is None else used
    return controller.observe(records(first, end, used), used=used, completed=completed)


def test_two_nonoverlapping_windows_switch_next_batch_and_reset_evidence():
    controller = ReuseController(16, ReuseOptions(window=2, patience=2, warmup_updates=0))
    for i in range(4):
        assert observe(controller)["reuse/next"] == 16
    result = observe(controller)
    assert result["reuse/used"] == 16
    assert result["reuse/next"] == 8
    assert result["reuse/switched"] == 1
    assert result["reuse/gap_over_drop"] == 1
    assert controller.previous is None and not controller.drops
    for _ in range(5):
        observe(controller)
    assert controller.current == 4
    for _ in range(20):
        observe(controller)
    assert controller.current == 4
    restored = ReuseController(16, controller.options)
    restored.load_state_dict(controller.state_dict())


@pytest.mark.parametrize("next_first,expected", [(0.25, -0.5), (1.25, 1.5)])
def test_signed_gap_ratio_is_not_clamped(next_first, expected):
    controller = ReuseController(16, ReuseOptions(window=1, patience=2, warmup_updates=0))
    observe(controller)
    result = observe(controller, first=next_first)
    assert result["reuse/gap_over_drop"] == expected
    assert result["reuse/fresh_progress"] == pytest.approx(1 - next_first)


def test_high_window_followed_by_progress_resets_patience():
    controller = ReuseController(16, ReuseOptions(window=1, patience=2, warmup_updates=0))
    observe(controller)
    observe(controller)
    assert controller.high_windows == 1
    observe(controller, first=0.6, end=0.3)
    assert controller.high_windows == 0 and controller.current == 16


def test_fixed_reuse_observes_the_same_s_without_modulation():
    options = dict(window=2, patience=2, warmup_updates=0)
    fixed = ReuseController(16, ReuseOptions(mode="fixed", **options))
    adaptive = ReuseController(16, ReuseOptions(**options))
    for _ in range(5):
        fixed_result = observe(fixed)
        adaptive_result = observe(adaptive)
        assert fixed_result.get("reuse/gap_over_drop") == adaptive_result.get("reuse/gap_over_drop")
    assert fixed.current == 16 and adaptive.current == 8
    assert fixed_result["reuse/switched"] == 0


def test_online_is_one_update_per_fresh_batch_and_s_is_undefined():
    online = ReuseController(1, ReuseOptions(mode="fixed", minimum=1, warmup_updates=0))
    for _ in range(20):
        result = observe(online)
        assert result["reuse/used"] == result["reuse/next"] == 1
        assert result["reuse/s_defined"] == 0
        assert "reuse/gap_over_drop" not in result
        assert result["reuse/first_rkl"] == 1.0
    restored = ReuseController(1, online.options)
    restored.load_state_dict(online.state_dict())


@pytest.mark.parametrize("first,end", [(1.0, 1.0), (1.0, 1.1), (1.0, 1.0 - 1e-10)])
def test_nonpositive_or_negligible_drop_cannot_trigger_refresh(first, end):
    controller = ReuseController(16, ReuseOptions(window=1, patience=1, warmup_updates=0))
    observe(controller, first, end)
    result = observe(controller)
    assert result["reuse/valid"] == 0 and controller.current == 16


def test_warmup_partial_batch_and_nonfinite_signal_clear_history():
    controller = ReuseController(16, ReuseOptions(window=1, patience=2, warmup_updates=32))
    observe(controller, completed=32)
    assert controller.previous is None
    observe(controller, completed=48)
    observe(controller, completed=64)
    assert controller.high_windows == 1
    observe(controller, used=3, completed=67)
    assert controller.previous is None and controller.high_windows == 0
    observe(controller, completed=83)
    observe(controller, first=float("nan"), completed=99)
    assert controller.previous is None


@pytest.mark.parametrize("applied", [False, None])
@pytest.mark.parametrize("mode", ["fixed", "adaptive"])
def test_skipped_or_unknown_updates_fail_instead_of_claiming_matched_budget(applied, mode):
    controller = ReuseController(16, ReuseOptions(mode=mode))
    events = records()
    events[5]["applied"] = applied
    with pytest.raises(RuntimeError, match="successful"):
        controller.observe(events, used=16, completed=15)


def test_missing_loss_is_a_protocol_error():
    controller = ReuseController(16, ReuseOptions())
    with pytest.raises(ValueError, match="native"):
        controller.observe([{"applied": True, "metrics": {}}] * 16, used=16, completed=16)


def test_exact_budget_including_nondivisible_eval_save_and_terminal_boundaries():
    budget = UpdateBudget(total=37, eval_every=10, save_every=13)
    completed, evaluations, saves, epochs = 0, [], [], []
    while completed < budget.total:
        used = budget.epochs(completed, 16)
        epochs.append(used)
        completed += used
        if budget.due(completed):
            evaluations.append(completed)
        if budget.due(completed, save=True):
            saves.append(completed)
    assert completed == 37 and all(0 < n <= 16 for n in epochs)
    assert budget.epochs(36, 16) == 1
    assert evaluations == [10, 20, 30, 37]
    assert saves == [13, 26, 37]
    with pytest.raises(ValueError, match="exhausted"):
        budget.epochs(completed, 16)


@pytest.mark.parametrize("cut", range(1, 14))
def test_checkpoint_resume_preserves_every_decision_and_partial_window(tmp_path, cut):
    options = ReuseOptions(window=2, patience=2, warmup_updates=0)
    uninterrupted = ReuseController(16, options)
    budget = UpdateBudget(640, 160, 160)
    completed, expected = 0, []
    for i in range(20):
        used = uninterrupted.current
        completed += used
        expected.append(observe(uninterrupted, completed=completed))
        if i == cut:
            save_training_control(tmp_path, budget, uninterrupted, completed)
            saved_completed = completed
    resumed = ReuseController(16, options)
    load_training_control(tmp_path, budget, resumed, saved_completed)
    for i in range(cut + 1, 20):
        saved_completed += resumed.current
        assert observe(resumed, completed=saved_completed) == expected[i]
    assert resumed.state_dict() == uninterrupted.state_dict()
    with pytest.raises(ValueError, match="progress"):
        load_training_control(tmp_path, budget, resumed, saved_completed + 1)


@pytest.mark.parametrize(
    "options", [{"window": 0}, {"minimum": 1}, {"threshold": float("nan")}, {"warmup_updates": True}]
)
def test_invalid_options_rejected(options):
    with pytest.raises(ValueError):
        ReuseOptions(**options)


def config():
    return OmegaConf.create(
        dict(
            trainer=dict(
                update_budget=dict(total=640, eval_every=160, save_every=160),
                reuse_control={},
                total_training_steps=640,
                save_freq=1,
                v1=dict(trainer_mode="sync", sync=dict(parameter_sync_step=1)),
            ),
            data=dict(train_batch_size=64),
            actor_rollout_ref=dict(
                actor=dict(
                    ppo_epochs=16,
                    ppo_mini_batch_size=64,
                    strategy="fsdp2",
                    optim=dict(lr_scheduler_type="constant"),
                    checkpoint=dict(async_save=False),
                ),
                rollout=dict(n=1),
            ),
            distillation=dict(
                enabled=True,
                distillation_loss=dict(
                    loss_mode="mirror",
                    mirror=dict(estimator="full_vocab", objective="mdpo", kappa=0, eta=1),
                ),
            ),
        )
    )


@pytest.mark.parametrize(
    "path,value",
    [
        ("trainer.v1.trainer_mode", "separate_async"),
        ("trainer.v1.sync.parameter_sync_step", 2),
        ("actor_rollout_ref.actor.ppo_mini_batch_size", 32),
        ("actor_rollout_ref.rollout.n", 2),
        ("distillation.distillation_loss.mirror.kappa", 0.1),
        ("actor_rollout_ref.actor.optim.lr_scheduler_type", "cosine"),
        ("actor_rollout_ref.actor.checkpoint.async_save", True),
    ],
)
def test_unsupported_training_semantics_rejected_before_workers(path, value):
    cfg = config()
    assert build_training_control(cfg)[1].current == 16
    OmegaConf.update(cfg, path, value)
    with pytest.raises(ValueError):
        build_training_control(cfg)


def test_corrupt_checkpoint_or_configuration_change_is_not_silently_accepted():
    controller = ReuseController(16, ReuseOptions())
    for key, value in [("current", 7), ("drops", [float("nan")]), ("high_windows", -1)]:
        state = copy.deepcopy(controller.state_dict())
        state[key] = value
        with pytest.raises(ValueError):
            controller.load_state_dict(state)
    changed = ReuseController(16, ReuseOptions(threshold=0.95))
    with pytest.raises(ValueError, match="configuration"):
        changed.load_state_dict(controller.state_dict())


def test_trainer_passes_capped_epochs_and_consumes_native_worker_records():
    from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
    from verl.utils import tensordict_utils as tu

    trainer = PPOTrainerSync.__new__(PPOTrainerSync)
    trainer.config = config()
    actor = trainer.config.actor_rollout_ref.actor
    actor.update(calculate_entropy=False, entropy_coeff=0, shuffle=False, data_loader_seed=42, policy_loss={})
    trainer.config.actor_rollout_ref.rollout.temperature = 1.0
    trainer.config.distillation.enabled = False  # Bypass unrelated loss dataclass construction in this fixture.
    trainer.update_budget = UpdateBudget(total=18, eval_every=18, save_every=18)
    trainer.reuse_controller = ReuseController(16, ReuseOptions(warmup_updates=0))
    trainer.actor_optimizer_step = trainer.actor_optimizer_step_attempts = 16
    trainer.distillation_config = None
    trainer.readout_callback = trainer.logger = None
    trainer._uses_full_vocab = lambda: True
    trainer._uses_mirror = lambda: False
    batch = SimpleNamespace(extra_info={})

    def update_actor(batch):
        assert batch.extra_info["epochs"] == 2
        return tu.get_tensordict(
            {}, dict(metrics={}, update_records=records(n=2), optimizer_step_attempts=2, optimizer_steps=2)
        )

    trainer.actor_rollout_wg = SimpleNamespace(update_actor=update_actor)
    metrics = {}
    assert trainer._update_actor(batch, metrics) is batch
    assert trainer.actor_optimizer_step == trainer.actor_optimizer_step_attempts == 18
    assert metrics["reuse/used"] == 2 and trainer.reuse_controller.current == 16


def test_evaluator_schedules_on_updates_without_relabeling_outer(monkeypatch):
    from verl.trainer.evaluation.config import EvaluationConfig
    from verl.trainer.evaluation.coordinator import EvaluationCoordinator
    from verl.trainer.evaluation.plan import compile_plan

    coordinator = EvaluationCoordinator.__new__(EvaluationCoordinator)
    coordinator.options = EvaluationConfig.parse(
        dict(
            output="unused",
            bootstrap="unused",
            source_enabled=False,
            history_enabled=False,
            datasets=[dict(name="math", domain="math", path="unused", verifier="math")],
            protocols=dict(public={}),
            schedules=dict(banks=dict(every=160), bench=dict(at=[], final=False)),
            final_export=False,
        )
    )
    coordinator.pending_outer = 10
    coordinator.completed, coordinator.pending_events = set(), {}
    trainer = SimpleNamespace(
        update_budget=UpdateBudget(640, 160, 160), actor_optimizer_step=160, total_training_steps=640
    )

    class Captured(Exception):
        pass

    def capture(*args, **kwargs):
        plan = compile_plan(*args, **kwargs)
        assert kwargs["outer"] == 160 and kwargs["total"] == 640
        assert set(plan.tasks) == {"public_initial", "public_teacher"}
        raise Captured

    monkeypatch.setattr("verl.trainer.evaluation.coordinator.compile_plan", capture)
    with pytest.raises(Captured):
        coordinator.validate(trainer)


@pytest.mark.parametrize(
    "budget,source,history,allowed",
    [
        (True, False, False, True),
        (False, False, False, False),
        (True, True, False, False),
        (True, False, True, False),
    ],
)
def test_repeated_training_questions_only_allowed_in_update_budget_reporting_runs(budget, source, history, allowed):
    from verl.trainer.evaluation.coordinator import EvaluationCoordinator

    coordinator = EvaluationCoordinator.__new__(EvaluationCoordinator)
    coordinator.options = SimpleNamespace(source_enabled=source, history_enabled=history)
    coordinator.batches = {1: [dict(source_id="training-question")]}
    coordinator.fixed = dict(source_heldout=[dict(source_id="heldout-question")])
    trainer = SimpleNamespace(update_budget=UpdateBudget(1280, 160, 160) if budget else None)
    rows = [dict(source_id="training-question")]
    if allowed:
        coordinator._validate_training_sources(trainer, rows)
    else:
        with pytest.raises(ValueError, match="repeated"):
            coordinator._validate_training_sources(trainer, rows)
    with pytest.raises(ValueError, match="heldout"):
        coordinator._validate_training_sources(trainer, [dict(source_id="heldout-question")])


def test_fit_uses_successful_update_budget_across_data_epochs_and_exports_at_actual_last_outer(monkeypatch):
    from unittest.mock import MagicMock

    from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync

    trainer = PPOTrainerSync.__new__(PPOTrainerSync)
    trainer.config = config()
    trainer.config.trainer.update(
        project_name="test",
        experiment_name="test",
        logger=["console"],
        val_before_train=False,
        total_epochs=1,
        test_freq=1,
    )
    trainer.config.global_profiler = dict(steps=None)
    trainer.update_budget = UpdateBudget(18, 18, 18)
    trainer.total_training_steps, trainer.steps_per_epoch = 18, 1
    trainer.global_steps = trainer.actor_optimizer_step = trainer.actor_optimizer_step_attempts = 0
    trainer.use_critic = False
    trainer.readout_callback = None
    for method in (
        "on_train_begin",
        "on_train_end",
        "on_step_begin",
        "on_step_end",
        "on_validate_begin",
        "on_validate_end",
        "_start_profiling",
        "_stop_profiling",
        "_reissue_inflight_prompts",
        "_shutdown_dump_executor",
        "_compute_metrics",
        "_save_checkpoint",
    ):
        setattr(trainer, method, MagicMock())
    trainer._validate = MagicMock(return_value={"eval/math/test": 1.0})
    trainer._consume_sync_metrics = lambda: {}
    boundaries = []

    def step(metrics, timing):
        trainer.actor_optimizer_step += trainer.update_budget.epochs(trainer.actor_optimizer_step, 16)
        trainer.actor_optimizer_step_attempts = trainer.actor_optimizer_step
        boundaries.append((trainer.global_steps, trainer.actor_optimizer_step))
        return SimpleNamespace(keys=[], partition_id="train")

    trainer.step = step
    prefix = "verl.trainer.ppo.v1.trainer_base."
    for name in (
        "Tracking",
        "ValidationGenerationsLogger",
        "DapoFilteredRewardTableLogger",
        "SkipManager",
        "tq.kv_clear",
    ):
        monkeypatch.setattr(prefix + name, MagicMock())
    trainer.fit(None)
    assert boundaries == [(1, 16), (2, 18)]
    assert trainer.total_training_steps == 2
    trainer._save_checkpoint.assert_called_once()
    trainer._validate.assert_called_once()
