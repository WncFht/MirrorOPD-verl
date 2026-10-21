# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Ablation gradients, packed token alignment, and the legacy boundary."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl.trainer.distillation.losses import distillation_loss, distillation_ppo_loss
from verl.utils import tensordict_utils as tu
from verl.utils.config import validate_sampled_token_distillation
from verl.workers.config import DistillationLossConfig


def loss_config(mode="token_is", source="current", **overrides):
    kwargs = dict(
        loss_mode="k1",
        use_task_rewards=False,
        use_policy_gradient=True,
        loss_max_clamp=None,
        log_prob_min_clamp=None,
        policy_loss_mode=mode,
        advantage_source=source,
    )
    kwargs.update(overrides)
    return DistillationLossConfig(**kwargs)


def inputs(student, teacher, behavior, *, lengths=None, prompt_lengths=None, mask=None, dp_size=1, token_count=None):
    """Pack next-token scores with unequal prompts; behavior is already response-aligned."""
    bsz, width = student.shape
    lengths = lengths or [width] * bsz
    prompt_lengths = prompt_lengths or [1] * bsz
    prompts = [torch.ones(p, dtype=torch.long) for p in prompt_lengths]
    responses = [torch.ones(r, dtype=torch.long) for r in lengths]
    packed_student, packed_teacher = [], []
    for i, (p, r) in enumerate(zip(prompt_lengths, lengths, strict=True)):
        packed_student.append(torch.cat((student.new_zeros(p - 1), student[i, :r], student.new_zeros(1))))
        packed_teacher.append(
            torch.cat((teacher.new_full((p - 1,), float("nan")), teacher[i, :r], teacher.new_zeros(1)))
        )
    if mask is None:
        mask = torch.arange(width)[None, :] < torch.tensor(lengths)[:, None]
    data = TensorDict(
        {
            "prompts": torch.nested.nested_tensor(prompts, layout=torch.jagged),
            "responses": torch.nested.nested_tensor(responses, layout=torch.jagged),
            "response_mask": mask,
            "teacher_logprobs": torch.cat(packed_teacher).unsqueeze(-1),
            "rollout_log_probs": behavior,
            "old_log_probs": torch.full_like(behavior, -42),
        },
        batch_size=[],
    )
    tu.assign_non_tensor(
        data,
        dp_size=dp_size,
        batch_num_tokens=int(mask.sum()) if token_count is None else token_count,
        global_batch_size=bsz,
        max_response_len=width,
    )
    return {"log_probs": torch.cat(packed_student)}, data


def evaluate(student, teacher, behavior, cfg=None, **kwargs):
    output, data = inputs(student, teacher, behavior, **kwargs)
    actor = SimpleNamespace(loss_agg_mode="token-mean", global_batch_info={}, loss_scale_factor=None)
    return distillation_loss(actor, SimpleNamespace(distillation_loss=cfg or loss_config()), output, data)


@pytest.mark.parametrize("source", ["current", "rollout"])
@pytest.mark.parametrize("mode", ["token_ppo", "token_is", "reinforce"])
def test_all_combinations_ignore_fsdp_old_and_detach_teacher_behavior(mode, source):
    student = torch.tensor([[-0.7, -1.5]], requires_grad=True)
    teacher = torch.tensor([[-1.0, -1.0]], requires_grad=True)
    behavior = torch.tensor([[-0.9, -1.2]], requires_grad=True)
    output, data = inputs(student, teacher, behavior)
    actor = SimpleNamespace(loss_agg_mode="token-mean", global_batch_info={}, loss_scale_factor=None)
    config = SimpleNamespace(distillation_loss=loss_config(mode, source))
    loss, _ = distillation_loss(actor, config, output, data)
    data["old_log_probs"] = torch.zeros_like(behavior)
    again, _ = distillation_loss(actor, config, output, data)
    torch.testing.assert_close(loss, again)
    loss.backward()
    assert student.grad is not None and student.grad.abs().sum() > 0
    assert teacher.grad is None and behavior.grad is None


def test_expected_current_is_gradient_matches_full_vocabulary_reverse_kl():
    logits = torch.tensor([0.2, -0.6, 0.8], requires_grad=True)
    log_teacher = torch.tensor([0.1, 0.6, 0.3]).log()
    mu = torch.tensor([0.5, 0.2, 0.3])
    lp = logits.log_softmax(-1)
    expected_loss = (lp.exp() * (lp - log_teacher)).sum()
    expected = torch.autograd.grad(expected_loss, logits, retain_graph=True)[0]
    sampled_expectation = sum(
        mu[i] * evaluate(lp[i].reshape(1, 1), log_teacher[i].reshape(1, 1), mu[i].log().reshape(1, 1))[0]
        for i in range(3)
    )
    actual = torch.autograd.grad(sampled_expectation, logits)[0]
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)


def test_no_is_current_gradient_matches_logprob_mse_and_is_nonzero():
    student = torch.tensor([[-1.2, -0.4]], requires_grad=True)
    teacher = torch.tensor([[-0.5, -1.1]])
    loss, metrics = evaluate(student, teacher, student.detach() - 1, loss_config("reinforce"))
    expected = (student.detach() - teacher) / 2
    torch.testing.assert_close(torch.autograd.grad(loss, student)[0], expected)
    assert "distillation/ppo_clip_count" not in metrics
    assert "distillation/log_ratio_guard_count" not in metrics


def test_frozen_no_is_gradient_stays_fixed_across_student_updates():
    teacher, behavior = torch.tensor([[-0.5, -1.1]]), torch.tensor([[-1.2, -0.4]])
    gradients = []
    for offset in (0.0, 0.7):
        student = (behavior + offset).requires_grad_()
        loss, _ = evaluate(student, teacher, behavior, loss_config("reinforce", "rollout"))
        gradients.append(torch.autograd.grad(loss, student)[0])
    torch.testing.assert_close(*gradients)


def test_ppo_only_clips_when_surrogate_changes_and_has_no_dual_clip():
    behavior = torch.full((1, 4), -4.0)
    student = (behavior + torch.tensor([[2.0, 0.5, 0.5, 2.0]]).log()).requires_grad_()
    teacher = behavior + torch.tensor([[1.0, -1.0, 1.0, -1.0]])
    loss, metrics = evaluate(student, teacher, behavior, loss_config("token_ppo", "rollout"))
    grad = torch.autograd.grad(loss, student)[0]
    torch.testing.assert_close(grad, torch.tensor([[0.0, 0.0, -0.125, 0.5]]))
    assert metrics["distillation/ppo_clip_count"].aggregate() == 2
    # A very negative advantage at high ratio must not hit legacy dual-clip.
    large = torch.tensor([[0.0]], requires_grad=True)
    loss, _ = evaluate(large, torch.tensor([[-6.0]]), torch.tensor([[-4.0]]), loss_config("token_ppo", "rollout"))
    assert torch.autograd.grad(loss, large)[0].item() > 100


@pytest.mark.parametrize("mode", ["token_is", "reinforce"])
def test_non_ppo_modes_ignore_clip_parameters(mode):
    student = torch.tensor([[-0.5]], requires_grad=True)
    args = (student, torch.tensor([[-1.0]]), torch.tensor([[-2.0]]))
    a, _ = evaluate(*args, loss_config(mode, clip_ratio_low=0.01, clip_ratio_high=0.01))
    b, _ = evaluate(*args, loss_config(mode, clip_ratio_low=0.9, clip_ratio_high=30))
    torch.testing.assert_close(a, b)


def test_null_clip_widths_fall_back_to_clip_ratio():
    student = torch.tensor([[-0.5]], requires_grad=True)
    args = (student, torch.tensor([[-0.7]]), torch.tensor([[-1.0]]))
    a, _ = evaluate(*args, loss_config("token_ppo", "rollout", clip_ratio_low=None, clip_ratio_high=None))
    b, _ = evaluate(*args, loss_config("token_ppo", "rollout"))
    torch.testing.assert_close(a, b)


@pytest.mark.parametrize("mode", ["token_ppo", "token_is", "reinforce"])
def test_padding_nan_and_empty_local_rank_keep_finite_gradient(mode):
    student = torch.tensor([[-1.0, float("nan")], [float("inf"), float("nan")]], requires_grad=True)
    teacher = torch.tensor([[-0.5, float("nan")], [float("nan"), float("nan")]])
    behavior = torch.tensor([[-1.0, float("nan")], [float("nan"), float("nan")]])
    mask = torch.tensor([[True, False], [False, False]])
    loss, _ = evaluate(student, teacher, behavior, loss_config(mode), mask=mask)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(student.grad).all()
    student.grad = None
    loss, _ = evaluate(
        student, teacher, behavior, loss_config(mode), mask=torch.zeros_like(mask), token_count=5, dp_size=2
    )
    loss.backward()
    assert loss.item() == 0 and torch.equal(student.grad, torch.zeros_like(student))


def test_unequal_prompt_response_lengths_eos_and_nested_behavior_align_once():
    student = torch.tensor([[-0.5, -0.8, -1.0], [-0.6, -0.2, -0.1]], requires_grad=True)
    teacher = torch.tensor([[-0.9, -0.4, -1.2], [-0.3, -0.2, -0.1]])
    behavior = torch.nested.nested_tensor([student[0].detach(), student[1, :1].detach()], layout=torch.jagged)
    loss, _ = evaluate(student, teacher, behavior, loss_config("reinforce"), lengths=[3, 1], prompt_lengths=[2, 4])
    grad = torch.autograd.grad(loss, student)[0]
    expected = (student.detach() - teacher) / 4
    expected[1, 1:] = 0
    torch.testing.assert_close(grad, expected)


def test_log_ratio_guard_and_sign_flip_counts_exclude_zero_and_padding():
    student = torch.tensor([[-1.0, -30.0, -2.0, -2.0]], requires_grad=True)
    teacher = torch.tensor([[-2.0, -20.0, -2.0, 0.0]])
    behavior = torch.tensor([[-30.0, -1.0, -1.0, 0.0]])
    loss, metrics = evaluate(student, teacher, behavior, mask=torch.tensor([[1, 1, 1, 0]]))
    assert metrics["distillation/log_ratio_guard_count"].aggregate() == 2
    assert metrics["distillation/advantage_sign_flip_count"].aggregate() == 2
    assert metrics["distillation/token_observations"].aggregate() == 3
    assert torch.isfinite(loss)
    assert torch.equal(torch.autograd.grad(loss, student)[0], torch.zeros_like(student))


def test_new_mode_skips_ordinary_ppo_diagnostics():
    output, data = inputs(torch.tensor([[-1.0]], requires_grad=True), torch.tensor([[-0.5]]), torch.tensor([[-0.8]]))
    actor = SimpleNamespace(loss_agg_mode="token-mean", global_batch_info={}, loss_scale_factor=None)
    with patch("verl.trainer.distillation.losses.ppo_loss", side_effect=AssertionError("must not run")):
        loss, metrics = distillation_ppo_loss(actor, SimpleNamespace(distillation_loss=loss_config()), output, data)
    assert torch.isfinite(loss) and "distillation/loss" in metrics


def test_vanilla_output_gradient_and_diagnostics_remain_legacy():
    student = torch.tensor([[-0.7, -1.5]], requires_grad=True)
    output, data = inputs(student, torch.tensor([[-1.0, -1.0]]), student.detach())
    data["old_log_probs"] = student.detach().clone()
    actor = SimpleNamespace(loss_agg_mode="token-mean", global_batch_info={}, loss_scale_factor=None)
    cfg = loss_config("vanilla")
    from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla

    expected, _ = compute_policy_loss_vanilla(
        old_log_prob=data["old_log_probs"],
        log_prob=student,
        advantages=(torch.tensor([[-1.0, -1.0]]) - student).detach(),
        response_mask=data["response_mask"],
        loss_agg_mode="token-mean",
        config=cfg,
    )
    with patch("verl.trainer.distillation.losses.ppo_loss", return_value=(torch.tensor(123.0), {"legacy": 1})) as ppo:
        actual, metrics = distillation_ppo_loss(actor, SimpleNamespace(distillation_loss=cfg), output, data)
    ppo.assert_called_once()
    assert metrics["legacy"] == 1
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, student, retain_graph=True)[0], torch.autograd.grad(expected, student)[0]
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"advantage_source": "bad"},
        {"policy_loss_mode": "bad"},
        {"loss_mode": "k3"},
        {"use_policy_gradient": False},
        {"use_task_rewards": True},
        {"loss_max_clamp": 10},
        {"log_prob_min_clamp": -10},
        {"policy_loss_mode": "vanilla", "advantage_source": "rollout"},
        {"policy_loss_mode": "token_ppo", "clip_ratio_low": 1.0},
    ],
)
def test_invalid_local_combinations_fail(overrides):
    with pytest.raises(ValueError):
        loss_config(**overrides)


def full_config():
    return OmegaConf.create(
        {
            "distillation": {
                "enabled": True,
                "distillation_loss": {
                    "loss_mode": "k1",
                    "use_policy_gradient": True,
                    "use_task_rewards": False,
                    "loss_max_clamp": None,
                    "log_prob_min_clamp": None,
                    "policy_loss_mode": "token_is",
                },
            }
        }
    )


@pytest.mark.parametrize(
    "path,value",
    [
        ("trainer.use_v1", False),
        ("trainer.v1.trainer_mode", "colocate_async"),
        ("trainer.v1.sync.parameter_sync_step", 2),
        ("actor_rollout_ref.actor.strategy", "fsdp"),
        ("actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size", 2),
        ("actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype", "float16"),
        ("actor_rollout_ref.actor.optim.optimizer", "Adam"),
        ("actor_rollout_ref.rollout.temperature", 0.7),
        ("actor_rollout_ref.rollout.top_p", 0.9),
        ("actor_rollout_ref.rollout.calculate_log_probs", False),
        ("actor_rollout_ref.rollout.repetition_penalty", 1.2),
        ("actor_rollout_ref.rollout.engine_kwargs.vllm.logits_processors", ["custom.Processor"]),
        ("actor_rollout_ref.rollout.engine_kwargs.vllm.override_generation_config", {"min_p": 0.1}),
        ("actor_rollout_ref.rollout.engine_kwargs.vllm.logprobs_mode", "raw_logprobs"),
        ("actor_rollout_ref.rollout.agent.default_agent_loop", "tool_agent"),
        ("algorithm.rollout_correction.rollout_is", "token"),
        ("algorithm.rollout_correction.bypass_mode", True),
        ("actor_rollout_ref.actor.checkpoint.async_save", True),
        ("actor_rollout_ref.actor.entropy_coeff", 0.1),
    ],
)
def test_unsupported_training_configs_fail_without_restricting_vanilla(path, value):
    cfg = full_config()
    validate_sampled_token_distillation(cfg)
    OmegaConf.update(cfg, path, value, force_add=True)
    with pytest.raises(ValueError):
        validate_sampled_token_distillation(cfg)
    cfg.distillation.distillation_loss.policy_loss_mode = "vanilla"
    validate_sampled_token_distillation(cfg)


@pytest.mark.parametrize("mode", ["token_ppo", "token_is", "reinforce"])
@pytest.mark.parametrize("source", ["current", "rollout"])
def test_gradient_is_invariant_to_dp_and_microbatch_partition(mode, source):
    student = torch.tensor([[-0.4, -1.5], [-0.6, -2.0], [-1.2, -0.8], [-0.5, -0.7]], requires_grad=True)
    teacher = torch.tensor([[-0.8, -1.0], [-0.9, -0.3], [-0.5, -1.3], [-1.1, -0.4]])
    behavior = torch.tensor([[-0.7, -1.2], [-0.4, -1.0], [-1.5, -0.5], [-0.8, -0.2]])
    mask = torch.tensor([[1, 1], [1, 0], [1, 1], [0, 0]], dtype=torch.bool)
    cfg = loss_config(mode, source)
    full, _ = evaluate(student, teacher, behavior, cfg, mask=mask)
    expected = torch.autograd.grad(full, student, retain_graph=True)[0]
    # Four one-sample micro-batches on two ranks; FSDP averages rank gradients.
    partitioned = sum(
        evaluate(
            student[i : i + 1],
            teacher[i : i + 1],
            behavior[i : i + 1],
            cfg,
            mask=mask[i : i + 1],
            dp_size=2,
            token_count=5,
        )[0]
        / 2
        for i in range(4)
    )
    torch.testing.assert_close(torch.autograd.grad(partitioned, student)[0], expected)


@pytest.mark.parametrize(
    "mode,source",
    [
        (mode, source)
        for mode in ("token_is", "token_ppo", "reinforce", "vanilla")
        for source in ("current", "rollout")
        if mode != "vanilla" or source == "current"
    ],
)
def test_observation_switch_preserves_loss_gradients_rng_and_token_alignment(mode, source):
    student = torch.tensor([[-0.7, -1.4], [-0.3, -1.1]], requires_grad=True)
    teacher = torch.tensor([[-0.5, -0.9], [-0.6, -1.2]])
    behavior = torch.tensor([[-0.9, -1.3], [-0.2, -1.0]])
    output, data = inputs(student, teacher, behavior, lengths=[2, 1], prompt_lengths=[2, 3])
    data["old_log_probs"] = behavior.clone()
    actor = SimpleNamespace(loss_agg_mode="token-mean", global_batch_info={}, loss_scale_factor=None)
    cfg = loss_config(mode, source)
    config = SimpleNamespace(distillation_loss=cfg)
    before_rng = torch.random.get_rng_state().clone()
    plain, _ = distillation_loss(actor, config, output, data)
    plain_grad = torch.autograd.grad(plain, student, retain_graph=True)[0]
    tu.assign_non_tensor(data, collect_metrics=True, metrics_quantile=True)
    observed, metrics = distillation_loss(actor, config, output, data)
    grad = torch.autograd.grad(observed, student)[0]
    torch.testing.assert_close(plain, observed, rtol=0, atol=0)
    torch.testing.assert_close(plain_grad, grad, rtol=0, atol=0)
    assert torch.equal(before_rng, torch.random.get_rng_state())
    assert "policy/current_behavior/ratio_p99" in metrics
    assert "advantage/distillation/abs_mean" in metrics
    if mode == "reinforce":
        assert "correction/log_ratio_guard/fraction" not in metrics
    if mode == "vanilla":
        assert "correction/anchor_ppo/active_fraction" in metrics
