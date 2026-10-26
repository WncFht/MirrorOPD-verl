# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from verl.utils.stable_logprobs import selected_logprobs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("vocab", [3, 8193, 151936])
@pytest.mark.parametrize("inplace", [False, True])
def test_tiled_ce_preserves_confident_target_and_amp(dtype, vocab, inplace):
    leaf = torch.full((2, vocab), -40.0, dtype=dtype, device="cuda", requires_grad=True)
    with torch.no_grad():
        leaf[:, 0] = 18
        leaf[:, 1:3] = 0
    z = leaf + 32  # Non-leaf also exercises actual in-place backward.
    label = torch.tensor([0, 1], device="cuda")
    reference = z.detach().double().requires_grad_()
    ref_value = reference.log_softmax(-1).gather(-1, label[:, None]).squeeze(-1)
    # Reconstruct the target component independently, avoiding the oracle CE's subtraction.
    probability = reference.detach().softmax(-1)
    reference_grad = -probability.scatter(-1, label[:, None], 0.0)
    reference_grad.scatter_(-1, label[:, None], -reference_grad.sum(-1, keepdim=True))
    value = selected_logprobs(z, label, inplace_backward=inplace)
    scale = 1024.0 if dtype == torch.float16 else 1.0
    gradient = torch.autograd.grad(value.sum() * scale, leaf)[0].double() / scale
    assert gradient[0, 0] != 0
    torch.testing.assert_close(value.double(), ref_value, atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(gradient, reference_grad, atol=1e-10, rtol=0.009 if dtype != torch.float32 else 4e-6)


def test_cuda_ignored_label_and_singleton():
    z = torch.tensor([[0.0, -torch.inf], [2.0, 1.0]], device="cuda", requires_grad=True)
    value = selected_logprobs(z, torch.tensor([0, -100], device="cuda"))
    assert torch.count_nonzero(value) == torch.count_nonzero(torch.autograd.grad(value.sum(), z)[0]) == 0
    x = torch.ones(3, 1, device="cuda", requires_grad=True)
    v = selected_logprobs(x, torch.zeros(3, dtype=torch.long, device="cuda"))
    assert torch.count_nonzero(v) == torch.count_nonzero(torch.autograd.grad(v.sum(), x)[0]) == 0


def test_cuda_undefined_distribution_remains_nonfinite():
    z = torch.tensor([[-torch.inf, -torch.inf], [torch.inf, 0.0]], device="cuda", requires_grad=True)
    value = selected_logprobs(z, torch.zeros(2, dtype=torch.long, device="cuda"))
    assert not torch.isfinite(value).any()
