# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA grouped GEMM")


@pytest.mark.parametrize("counts", [[0, 1, 19, 0, 3], [33, 0, 65, 1, 0]])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_grouped_fp32_matches_fp32_oracle_and_graph_replay(counts, dtype):
    from verl.experimental.rollout_policy.grouped_mm import grouped_fp32

    torch.manual_seed(123)
    x = torch.randn(sum(counts), 80, device="cuda", dtype=dtype)
    weights = torch.randn(len(counts), 70, 80, device="cuda", dtype=torch.bfloat16)
    offsets = torch.tensor(counts, device="cuda", dtype=torch.int32).cumsum(0, dtype=torch.int32)

    def oracle():
        parts = x.split(counts)
        return torch.cat(
            [part.double() @ weight.double().t() for part, weight in zip(parts, weights, strict=True)]
        ).float()

    torch.testing.assert_close(grouped_fp32(x, weights, offsets), oracle(), rtol=1e-5, atol=1e-5)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        grouped_fp32(x, weights, offsets)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = grouped_fp32(x, weights, offsets)
    x.mul_(0.5)
    graph.replay()
    torch.testing.assert_close(output, oracle(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("shape", [(3, 71, 128), (35, 67, 81)])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
def test_linear_fp32_keeps_activation_precision(shape, weight_dtype):
    from verl.experimental.rollout_policy.grouped_mm import linear_fp32

    rows, width, inner = shape
    torch.manual_seed(13)
    x = torch.randn(rows, inner, device="cuda")
    weight = torch.randn(width, inner, device="cuda", dtype=torch.bfloat16).to(weight_dtype)
    expected = (x.double() @ weight.double().t()).float()
    torch.testing.assert_close(linear_fp32(x, weight), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("width", [128, 2048])
def test_rms_fp32_oracle_and_row_independence(width):
    from verl.experimental.rollout_policy.grouped_mm import rms_fp32

    torch.manual_seed(42)
    values = torch.randn(17, width, device="cuda")
    weights = torch.randn(width, device="cuda", dtype=torch.bfloat16)
    actual = rms_fp32(values, weights, 1e-6)
    oracle = values.double() * (values.double().square().mean(-1, keepdim=True) + 1e-6).rsqrt() * weights.double()
    torch.testing.assert_close(actual.double(), oracle, atol=2e-6, rtol=1e-6)
    single = torch.cat([rms_fp32(row[None], weights, 1e-6) for row in values])
    torch.testing.assert_close(actual, single, rtol=0, atol=0)


def test_router_is_independent_of_row_batching():
    from types import SimpleNamespace

    from verl.experimental.rollout_policy.precision import _router_forward

    torch.manual_seed(42)
    values = torch.randn(17, 2048, device="cuda")
    router = SimpleNamespace(
        hidden_dim=2048,
        top_k=8,
        norm_topk_prob=True,
        weight=(torch.randn(128, 2048, device="cuda") * 0.02).bfloat16().float(),
    )
    batched = _router_forward(router, values)
    single = [_router_forward(router, row[None]) for row in values]
    for index in range(3):
        torch.testing.assert_close(batched[index], torch.cat([row[index] for row in single]), atol=0, rtol=0)
