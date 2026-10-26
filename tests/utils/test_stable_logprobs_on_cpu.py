# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from verl.utils.stable_logprobs import selected_logprobs
from verl.utils.torch_functional import logprobs_from_logits, logprobs_from_logits_v2


@pytest.mark.parametrize("implementation", [selected_logprobs, logprobs_from_logits_v2, logprobs_from_logits])
@pytest.mark.parametrize("offset", [0.0, 32.0, 128.0])
def test_confident_target_gradient_is_preserved(implementation, offset):
    z = (torch.tensor([[18.0, 0.0, 0.0]]) + offset).requires_grad_()
    lp = implementation(z, torch.tensor([0]))
    gradient = torch.autograd.grad(-lp.sum(), z)[0]
    tail = torch.exp(torch.tensor(-18.0, dtype=torch.float64))
    expected = torch.tensor([[-2 * tail, tail, tail]]) / (1 + 2 * tail)
    assert lp.item() < 0 and gradient[0, 0] < 0
    torch.testing.assert_close(gradient.double(), expected, atol=1e-14, rtol=3e-6)
    assert (-lp).item() == pytest.approx(torch.log1p(2 * tail).item(), rel=3e-6)


@pytest.mark.parametrize("shape", [(3, 7), (2, 3, 7), (7,)])
def test_value_and_vjp_match_independent_float64_ce(shape):
    torch.manual_seed(93)
    z = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    labels = torch.randint(shape[-1], shape[:-1])
    actual = selected_logprobs(z, labels)
    expected = z.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), z)[0], torch.autograd.grad(expected.sum(), z)[0])


def test_ignore_empty_and_singleton():
    z = torch.tensor([[0.0, -torch.inf], [2.0, 1.0]], requires_grad=True)
    value = selected_logprobs(z, torch.tensor([0, -100]))
    gradient = torch.autograd.grad(value.sum(), z)[0]
    assert torch.count_nonzero(value) == torch.count_nonzero(gradient) == 0
    single = torch.randn(3, 1, requires_grad=True)
    value = selected_logprobs(single, torch.zeros(3, dtype=torch.long))
    assert torch.count_nonzero(value) == torch.count_nonzero(torch.autograd.grad(value.sum(), single)[0]) == 0
    empty = torch.empty(0, 7, requires_grad=True)
    value = selected_logprobs(empty, torch.empty(0, dtype=torch.long))
    assert value.shape == (0,)
    assert torch.autograd.grad(value.sum(), empty)[0].shape == empty.shape


def test_undefined_distribution_does_not_become_zero_loss():
    z = torch.tensor([[-torch.inf, -torch.inf], [torch.inf, 0.0]], requires_grad=True)
    value = selected_logprobs(z, torch.zeros(2, dtype=torch.long))
    assert not torch.isfinite(value).any()
