# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Actual FA2 forward equivalence for a backward-only diagnostic intervention."""

import importlib
from pathlib import Path

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("heads,kv_heads,head_dim", [(4, 1, 32), (4, 2, 32), (4, 4, 32), (12, 2, 128)])
def test_native_cuda_forward_is_bitwise_preserved(monkeypatch, heads, kv_heads, head_dim):
    torch.cuda.set_per_process_memory_fraction(0.04)
    path = Path(__file__).resolve().parents[3] / "examples/mirroropd/docs/mirror-descent/scripts"
    monkeypatch.syspath_prepend(str(path))
    factors = importlib.import_module("factor_mirror_precision")
    from flash_attn import flash_attn_func

    torch.manual_seed(37)
    q = torch.randn(1, heads, 33, head_dim, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    k, v = [
        torch.randn(1, kv_heads, 33, head_dim, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        for _ in range(2)
    ]

    def native(module, q, k, v, mask, **kwargs):
        return flash_attn_func(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=True, softmax_scale=0.2
        ), None

    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected = native(None, q, k, v, None)[0]
        actual = factors.NativeForwardReferenceBackward.apply(q, k, v, None, native, {"scaling": 0.2}, torch.float64)
    assert torch.equal(actual, expected)
    go = torch.randn_like(actual)
    gradients = torch.autograd.grad(actual, (q, k, v), go)
    reference = factors.dense_attention_vjp(q.detach(), k.detach(), v.detach(), go, 0.2)
    for grad, ref in zip(gradients, reference, strict=True):
        assert torch.equal(grad, ref.bfloat16())
    native_gradients = torch.autograd.grad(expected, (q, k, v), go)
    analysis = importlib.import_module("analyze_attention_capture")
    record = {
        "layer": 0,
        "native_kwargs": {"scaling": 0.2, "dropout": 0.0},
        "dq": native_gradients[0].detach().cpu(),
        "out": expected.detach().cpu(),
    }
    replay = analysis.replay_native(record, q.detach(), k.detach(), v.detach(), go)
    assert all(value["forward_equal_capture"] for value in replay.values())
    assert replay["True"]["dq_repeat_max_error_energy"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_norm_forward_is_bitwise_preserved(monkeypatch):
    torch.cuda.set_per_process_memory_fraction(0.04)
    path = Path(__file__).resolve().parents[3] / "examples/mirroropd/docs/mirror-descent/scripts"
    monkeypatch.syspath_prepend(str(path))
    factors = importlib.import_module("factor_mirror_precision")
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

    torch.manual_seed(42)
    module = Qwen2RMSNorm(128).cuda().bfloat16()
    x = torch.randn(1, 17, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected = module(x)
        actual = factors.NativeNormReferenceBackward.apply(x, module.weight, module.variance_epsilon, module.forward)
    assert torch.equal(actual, expected)
    go = torch.randn_like(actual)
    gradients = torch.autograd.grad(actual, (x, module.weight), go)
    xd, wd = [p.detach().double().requires_grad_() for p in (x, module.weight)]
    exact = xd * torch.rsqrt(xd.square().mean(-1, keepdim=True) + module.variance_epsilon) * wd
    reference = torch.autograd.grad(exact, (xd, wd), go.double())
    for grad, ref in zip(gradients, reference, strict=True):
        assert (grad.double() - ref).norm() / ref.norm() < 0.004
