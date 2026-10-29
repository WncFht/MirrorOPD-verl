# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent differentiation checks for the diagnostic attention backward."""

import importlib

import pytest
import torch


@pytest.fixture
def factors(monkeypatch):
    return importlib.import_module("examples.mirroropd.research.studies.precision.factor_mirror_precision")


@pytest.mark.parametrize("heads,kv_heads", [(4, 4), (4, 2), (4, 1), (12, 2)])
def test_causal_gqa_vjp_matches_independent_autograd(factors, heads, kv_heads):
    torch.manual_seed(42)
    q = torch.randn(2, heads, 7, 5, dtype=torch.float64, requires_grad=True)
    k = torch.randn(2, kv_heads, 7, 5, dtype=torch.float64, requires_grad=True)
    v = torch.randn(2, kv_heads, 7, 5, dtype=torch.float64, requires_grad=True)
    groups = heads // kv_heads
    output = torch.nn.functional.scaled_dot_product_attention(
        q, k.repeat_interleave(groups, 1), v.repeat_interleave(groups, 1), is_causal=True, scale=0.37
    ).transpose(1, 2)
    go = torch.randn_like(output)
    expected = torch.autograd.grad(output, (q, k, v), go)
    actual = factors.dense_attention_vjp(q.detach(), k.detach(), v.detach(), go, 0.37)
    for a, e in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, e, rtol=1e-11, atol=1e-12)


def test_backward_oracle_preserves_supplied_forward_exactly(factors):
    torch.manual_seed(12)
    q, k, v = [torch.randn(1, 2, 5, 4, requires_grad=True) for _ in range(3)]

    def native(module, q, k, v, mask, **kwargs):
        output = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, scale=0.5)
        return output.transpose(1, 2).bfloat16().float(), None

    expected = native(None, q, k, v, None)[0]
    output = factors.NativeForwardReferenceBackward.apply(q, k, v, None, native, {"scaling": 0.5}, torch.float64)
    assert torch.equal(output, expected)
    go = torch.randn_like(output)
    actual = torch.autograd.grad(output, (q, k, v), go)
    reference = factors.dense_attention_vjp(q.detach(), k.detach(), v.detach(), go, 0.5)
    for a, r in zip(actual, reference, strict=True):
        torch.testing.assert_close(a, r.float(), rtol=0, atol=0)


def test_capture_analysis_recognizes_cast_only_errors(factors, tmp_path):
    analysis = importlib.import_module("examples.mirroropd.research.studies.precision.analyze_attention_capture")
    torch.manual_seed(8)
    q, k, v = [torch.randn(1, 2, 7, 5).bfloat16() for _ in range(3)]
    inputs = [x.double().requires_grad_() for x in (q, k, v)]
    out = torch.nn.functional.scaled_dot_product_attention(*inputs, is_causal=True, scale=0.5).transpose(1, 2)
    go = torch.randn_like(out).bfloat16()
    gradients = torch.autograd.grad(out, inputs, go.double())
    capture = {"q": q, "k": k, "v": v, "go": go, "out": out.detach().bfloat16(), "rank": 0, "layer": 0, "scale": 0.5}
    capture.update({n: g.bfloat16() for n, g in zip(("dq", "dk", "dv"), gradients, strict=True)})
    path = tmp_path / "capture.pt"
    torch.save(capture, path)
    result = analysis.analyze(path, "cpu")
    for name in ("dq", "dk", "dv", "output"):
        assert result[name]["error_energy"] == pytest.approx(
            result[name]["cast_only_error_energy"], rel=1e-8, abs=1e-20
        )
    assert result["dq_first_position_native_norm"] == result["dq_first_position_reference_norm"] == 0
    split = result["dq_output_reference_split"]
    assert split["native_minus_ideal_cast_prediction_energy"] < 1e-20
    assert split["ideal_output_cast_prediction_energy"] + split["twice_cross_inner_product"] == pytest.approx(
        split["total_prediction_energy"]
    )


def test_norm_backward_matches_autograd_and_keeps_native_forward(factors):
    torch.manual_seed(42)
    x = torch.randn(2, 3, 7, dtype=torch.float64, requires_grad=True)
    w = torch.randn(7, dtype=torch.float64, requires_grad=True)
    exact = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 0.003) * w
    native = lambda _: exact.detach().bfloat16().double()
    result = factors.NativeNormReferenceBackward.apply(x, w, 0.003, native)
    assert torch.equal(result, native(x))
    go = torch.randn_like(result)
    reference = torch.autograd.grad(exact, (x, w), go)
    actual = torch.autograd.grad(result, (x, w), go)
    for a, r in zip(actual, reference, strict=True):
        torch.testing.assert_close(a, r, atol=1e-12, rtol=1e-12)
