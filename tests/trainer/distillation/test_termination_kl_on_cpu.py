# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Stopping-action KL must preserve ordinary metrics and exact student VJPs."""

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from verl.trainer.distillation.full_vocab import dense_rows, full_reverse_kl_result, precompute_kl_rows
from verl.trainer.distillation.losses import distillation_loss
from verl.utils import tensordict_utils as tu
from verl.workers.config.distillation import DistillationConfig, DistillationLossConfig, FullVocabConfig


def coarse_reference(hs, ws, ht, wt, ids):
    p = (hs @ ws.T).softmax(-1)
    q = (ht.detach() @ wt.detach().T).softmax(-1)
    rest = [i for i in range(p.shape[1]) if i not in ids]
    p = torch.cat((p[:, rest], p[:, ids].sum(-1, keepdim=True)), dim=-1)
    q = torch.cat((q[:, rest], q[:, ids].sum(-1, keepdim=True)), dim=-1)
    return (p * (p.log() - q.log())).sum()


@pytest.mark.parametrize("chunk", [1, 3, 10])
@pytest.mark.parametrize("tokens", [0, 7])
def test_loss_vjp_and_raw_rows_match_independent_reference(chunk, tokens):
    torch.manual_seed(42)
    hs = torch.randn(tokens, 5, dtype=torch.float64, requires_grad=True)
    ws = torch.randn(9, 5, dtype=torch.float64, requires_grad=True)
    ht = torch.randn(tokens, 6, dtype=torch.float64, requires_grad=True)
    wt = torch.randn(9, 6, dtype=torch.float64, requires_grad=True)
    expected = coarse_reference(hs, ws, ht, wt, [1, 4])
    actual = full_reverse_kl_result(
        hs, ws, ht, wt, backend="precompute", token_chunk=chunk, termination_token_ids=[1, 4]
    )
    torch.testing.assert_close(actual.loss_sum, expected, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(actual.token_kl, dense_rows(hs, ws, ht, wt), atol=1e-10, rtol=1e-10)
    assert not actual.token_kl.requires_grad
    expected_grad = torch.autograd.grad(expected * 0.37, (hs, ws))
    grads = torch.autograd.grad(actual.loss_sum * 0.37, (hs, ws, ht, wt), allow_unused=True)
    for result, reference in zip(grads[:2], expected_grad, strict=True):
        torch.testing.assert_close(result, reference, atol=1e-10, rtol=1e-10)
    assert grads[2:] == (None, None)


def test_equivalent_stops_do_not_create_a_continuation_incentive():
    p = torch.tensor([[0.8, 1e-6, 0.1, 0.099999]], dtype=torch.float64)
    q = p[:, [1, 0, 2, 3]]
    hs = p.log().requires_grad_()
    eye = torch.eye(4, dtype=torch.float64)
    result = full_reverse_kl_result(hs, eye, q.log(), eye, backend="precompute", termination_token_ids=[0, 1])
    assert result.token_kl.item() > 10 and abs(result.loss_sum.item()) < 1e-12
    torch.testing.assert_close(torch.autograd.grad(result.loss_sum, hs)[0], torch.zeros_like(hs), atol=1e-12, rtol=0)
    with torch.no_grad():
        _, ordinary_gradient = precompute_kl_rows(hs, eye, q.log(), eye)
    assert ordinary_gradient[:, [0, 1]].sum() > 0


def test_no_configuration_keeps_original_loss_even_with_old_experiment_environment(monkeypatch):
    monkeypatch.setenv("VERL_E41_TERMINATION_EQUIV", "1")
    torch.manual_seed(1)
    hs, ws, ht, wt = [torch.randn(shape, dtype=torch.float64) for shape in [(5, 3), (8, 3), (5, 4), (8, 4)]]
    actual = full_reverse_kl_result(hs, ws, ht, wt, backend="precompute")
    torch.testing.assert_close(actual.loss_sum, dense_rows(hs, ws, ht, wt).sum())


@pytest.mark.parametrize("ids", [[1], [1, 1], [-1, 2], [True, 2], [1, 8]])
def test_invalid_kernel_groups_fail(ids):
    with pytest.raises(ValueError, match="supported token"):
        full_reverse_kl_result(
            torch.zeros(2, 3),
            torch.zeros(8, 3),
            torch.zeros(2, 3),
            torch.zeros(8, 3),
            backend="precompute",
            termination_token_ids=ids,
        )


@pytest.mark.parametrize("backend,teacher_vocab", [("dense", 8), ("precompute_fused", 8), ("precompute", 9)])
def test_unsupported_kernel_paths_fail(backend, teacher_vocab):
    with pytest.raises(ValueError, match="precompute and equal"):
        full_reverse_kl_result(
            torch.zeros(2, 3),
            torch.zeros(8, 3),
            torch.zeros(2, 3),
            torch.zeros(teacher_vocab, 3),
            backend=backend,
            termination_token_ids=[1, 4],
        )


@pytest.mark.parametrize("ids", [None, "1,2", [True, 2], [-1, 2], [1], [1, 1]])
def test_configuration_rejects_malformed_groups(ids):
    with pytest.raises(ValueError):
        FullVocabConfig(backend="precompute", termination_token_ids=ids)


def test_configuration_and_real_stop_support_are_checked():
    with pytest.raises(ValueError, match="precompute"):
        FullVocabConfig(backend="dense", termination_token_ids=[1, 4])
    with pytest.raises(ValueError, match="teacher-only"):
        DistillationLossConfig(full_vocab=FullVocabConfig(backend="precompute", termination_token_ids=[1, 4]))
    config = FullVocabConfig(backend="precompute", termination_token_ids=[1, 4])
    config.validate_termination_support(eos_token_ids=1, stop_token_ids=[4], vocab_ids=range(8), vocab_size=8)
    config.validate_termination_support(eos_token_ids=[1, 4], stop_token_ids=[], vocab_ids=range(8), vocab_size=8)
    with pytest.raises(ValueError, match="effective rollout"):
        config.validate_termination_support(eos_token_ids=1, stop_token_ids=[], vocab_ids=range(8), vocab_size=8)
    with pytest.raises(ValueError, match="vocabulary"):
        config.validate_termination_support(eos_token_ids=1, stop_token_ids=[4], vocab_ids=[1, 2], vocab_size=8)


def test_loss_normalizes_the_coarse_objective_but_reports_raw_kl():
    cfg = DistillationConfig(
        distillation_loss=DistillationLossConfig(
            loss_mode="full_reverse_kl",
            use_policy_gradient=False,
            use_task_rewards=False,
            loss_max_clamp=None,
            log_prob_min_clamp=None,
            full_vocab=FullVocabConfig(backend="precompute", termination_token_ids=[1, 4]),
        )
    )
    data = TensorDict({}, batch_size=[])
    tu.assign_non_tensor(data, dp_size=2, batch_num_tokens=11)
    raw = torch.tensor(9.0, requires_grad=True)
    objective = torch.tensor(7.0, requires_grad=True)
    loss, metrics = distillation_loss(
        SimpleNamespace(loss_agg_mode="token-mean"),
        cfg,
        {"full_kl_sum": raw, "termination_kl_sum": objective, "full_kl_count": 3},
        data,
    )
    loss.backward()
    assert raw.grad is None and objective.grad.item() == pytest.approx(2 / 11)
    assert metrics["distillation/full_kl"].state.tolist() == [9, 3]
    assert metrics["distillation/termination_equiv_kl"].state.tolist() == [7, 3]
    assert metrics["distillation/eos_identity_kl"].state.tolist() == [2, 3]
