# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Unified entry must compose real Hydra configs without changing experiment budgets."""

import pytest
from omegaconf import OmegaConf
from test_mirroropd_math_launcher_on_cpu import SCRIPT, _assert_full_kl, _config, _launch

from verl.utils.config import omega_conf_to_dataclass, validate_config

ENTRY = SCRIPT.parent / "run_math_opd_ablation_fsdp2.sh"


@pytest.mark.parametrize("preset", ["E1", "E13", "E17", "E28"])
@pytest.mark.parametrize(
    "method,estimator",
    [
        ("sampled", "sampled"),
        ("full_kl", "full_vocab"),
        ("mdpo", "sampled"),
        ("mdpo", "full_vocab"),
        ("bpo", "full_vocab"),
    ],
)
def test_methods_share_preset_models_and_budgets(tmp_path, preset, method, estimator):
    baseline = _config(tmp_path, command=["bash", str(ENTRY), preset])
    cfg = _config(tmp_path, command=["bash", str(ENTRY), preset, "--method", method, "--estimator", estimator])
    OmegaConf.resolve(cfg)
    validate_config(cfg, use_reference_policy=False, use_critic=False)
    for path in (
        "data",
        "actor_rollout_ref.model",
        "actor_rollout_ref.actor.optim",
        "trainer.total_training_steps",
        "trainer.test_freq",
        "trainer.save_freq",
        "actor_rollout_ref.actor.ppo_epochs",
        "distillation.teacher_models",
        "actor_rollout_ref.rollout.val_kwargs",
    ):
        assert OmegaConf.select(cfg, path) == OmegaConf.select(baseline, path), path
    loss = omega_conf_to_dataclass(cfg.distillation).distillation_loss
    if method == "sampled":
        assert loss.loss_mode == "k1"
    elif method == "full_kl":
        _assert_full_kl(loss)
    else:
        assert loss.loss_settings.objective == method
        assert loss.loss_settings.payload_kind == ("hidden" if estimator == "full_vocab" else "sampled")
        assert loss.loss_settings.eta == 0.3
        assert loss.use_policy_gradient == (estimator == "sampled")


def test_method_options_environment_and_hydra_precedence(tmp_path):
    cfg = _config(
        tmp_path,
        command=[
            "bash",
            str(ENTRY),
            "E13",
            "--method=mdpo",
            "--eta=.4",
            "--kappa",
            ".25",
            "distillation.distillation_loss.mirror.eta=0.7",
            "actor_rollout_ref.actor.ppo_epochs=2",
        ],
        EXPERIMENT_METHOD="sampled",
        MIRROR_ETA="0.2",
        PPO_EPOCHS="3",
        FULL_KL_BACKEND="precompute_fused",
    )
    assert cfg.distillation.distillation_loss.mirror.eta == 0.7
    assert cfg.distillation.distillation_loss.mirror.kappa == 0.25
    assert cfg.actor_rollout_ref.actor.ppo_epochs == 2
    assert cfg.distillation.distillation_loss.full_vocab.backend == "precompute_fused"
    cfg = _config(
        tmp_path,
        command=[
            "bash",
            str(ENTRY),
            "E1",
            "--advantage",
            "rollout",
            "--policy-loss",
            "token_is",
            "distillation.distillation_loss.advantage_source=current",
        ],
        OPD_ADVANTAGE_SOURCE="current",
        OPD_POLICY_LOSS_MODE="reinforce",
    )
    assert cfg.distillation.distillation_loss.advantage_source == "current"
    assert cfg.distillation.distillation_loss.policy_loss_mode == "token_is"
    cfg = _config(
        tmp_path, command=["bash", str(ENTRY), "E1"], OPD_ADVANTAGE_SOURCE="current", OPD_POLICY_LOSS_MODE="reinforce"
    )
    assert cfg.distillation.distillation_loss.advantage_source == "current"
    assert cfg.distillation.distillation_loss.policy_loss_mode == "reinforce"


@pytest.mark.parametrize(
    "args",
    [
        ["E13", "--method", "bpo", "--estimator", "sampled"],
        ["E13", "--method", "unknown"],
        ["E13", "--eta"],
        ["E13", "--eta", "--method", "mdpo"],
        ["E13", "--method", "full_kl", "--eta", ".3"],
        ["E15", "--estimator", "full_vocab"],
        ["E13", "--policy-loss", "token_ppo"],
        ["E15", "--advantage", "unknown"],
        ["E13-valid-teacher", "--method", "bpo"],
    ],
)
def test_invalid_or_ambiguous_combinations_fail_before_launch(tmp_path, args):
    assert _launch(tmp_path, command=["bash", str(ENTRY), *args]).returncode == 2


def test_removed_public_aliases_are_absent():
    for name in ("run_full_reverse_kl.sh", "run_mirror.sh", "run_math_deepseek_r1_1_5b_fsdp2.sh"):
        assert not (SCRIPT.parent / name).exists()


def test_overridden_method_has_distinct_automatic_run_name(tmp_path):
    cfg = _config(tmp_path, command=["bash", str(ENTRY), "E13", "--method", "bpo", "--eta", ".4"], EXPERIMENT_NAME="")
    assert "E13-bpo-full_vocab-k0.5-eta.4" in cfg.trainer.experiment_name
    assert cfg.trainer.total_training_steps == 250


def test_custom_domain_name_and_data_stay_consistent(tmp_path):
    cfg = _config(
        tmp_path, command=["bash", str(ENTRY), "custom", "--method", "bpo"], DOMAIN="code", EXPERIMENT_NAME=""
    )
    assert cfg.trainer.experiment_name.startswith("code-opd-custom-bpo-")
    assert cfg.data.train_files[0].endswith("/code/TACO/train.parquet")


def test_custom_entry_accepts_environment_method(tmp_path):
    cfg = _config(tmp_path, EXPERIMENT_METHOD="bpo")
    assert cfg.distillation.distillation_loss.mirror.objective == "bpo"
    assert not cfg.distillation.distillation_loss.use_policy_gradient


@pytest.mark.parametrize(
    "preset,objective,kappa,eta",
    [("E44", "mdpo", 0, 1), ("E45", "mdpo", 0, 0.3), ("E46", "bpo", 0, 0.3), ("E47", "mdpo", 0.5, 0.3)],
)
def test_registered_mirror_comparison_matches_e20_protocol(tmp_path, preset, objective, kappa, eta):
    cfg = _config(tmp_path, command=["bash", str(ENTRY), preset])
    baseline = _config(tmp_path, command=["bash", str(ENTRY), "E20"])
    OmegaConf.resolve(cfg)
    validate_config(cfg, use_reference_policy=False, use_critic=False)
    for path in (
        "data",
        "actor_rollout_ref.model",
        "actor_rollout_ref.actor.optim",
        "actor_rollout_ref.actor.ppo_epochs",
        "actor_rollout_ref.actor.ppo_mini_batch_size",
        "actor_rollout_ref.rollout.val_kwargs",
        "trainer.total_training_steps",
        "trainer.test_freq",
        "trainer.val_before_train",
        "trainer.max_actor_ckpt_to_keep",
        "algorithm.filter_groups",
        "trainer.v1.sampler.sync_refill_failed_groups",
    ):
        assert OmegaConf.select(cfg, path) == OmegaConf.select(baseline, path), path
    assert cfg.trainer.total_training_steps == 256
    assert cfg.actor_rollout_ref.actor.ppo_epochs == 16
    assert cfg.trainer.test_freq == 32 and cfg.trainer.save_freq == 128
    assert not cfg.trainer.val_before_train
    assert cfg.data.train_batch_size == cfg.actor_rollout_ref.actor.ppo_mini_batch_size == 64
    assert cfg.actor_rollout_ref.rollout.val_kwargs.n == 16
    assert cfg.data.seed == cfg.actor_rollout_ref.rollout.seed == cfg.actor_rollout_ref.actor.data_loader_seed == 42
    loss = omega_conf_to_dataclass(cfg.distillation).distillation_loss
    assert loss.loss_settings.objective == objective
    assert loss.loss_settings.kappa == kappa and loss.loss_settings.eta == eta
    assert loss.loss_settings.payload_kind == "hidden"
    assert loss.loss_settings.diagnostics
    assert loss.full_vocab.backend == "precompute_fused" and loss.full_vocab.token_chunk_size == 128
    assert loss.full_vocab.async_write and loss.full_vocab.prefetch
    assert not loss.use_policy_gradient
    if preset == "E44":
        assert loss.loss_settings.teacher_only_kl and loss.loss_settings.anchor_payload is None
    else:
        assert loss.loss_settings.anchor_payload == "hidden"


def test_registered_mirror_validation_uses_formal_eval_budget(tmp_path):
    cfg = _config(tmp_path, command=["bash", str(ENTRY), "E45-valid-student"])
    assert cfg.trainer.val_only
    assert cfg.actor_rollout_ref.rollout.val_kwargs.n == 16
    assert cfg.actor_rollout_ref.rollout.val_kwargs.max_tokens == 31744
