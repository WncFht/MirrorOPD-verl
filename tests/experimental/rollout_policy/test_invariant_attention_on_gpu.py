# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
import math

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA invariant GQA")


@pytest.mark.parametrize("length,dim", [(65, 32), (137, 128)])
def test_fp64_oracle_padding_and_prefill_decode_identity(length, dim):
    from verl.experimental.rollout_policy.invariant_attention import causal_gqa

    torch.manual_seed(12)
    q = torch.randn(2, 4, length, dim, device="cuda", dtype=torch.float16)
    k = torch.randn(2, 2, length + 23, dim, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    left = torch.tensor([0, 17], device="cuda", dtype=torch.int32)
    lengths = torch.full((2,), length, device="cuda", dtype=torch.int32)
    q[1, :, :17] = torch.nan
    k[1, :, :17] = v[1, :, :17] = torch.nan
    k[:, :, length:] = v[:, :, length:] = torch.nan
    actual = causal_gqa(q, k, v, lengths, left, dim**-0.5)
    for batch, padding in enumerate([0, 17]):
        query = q[batch : batch + 1, :, padding:].contiguous()
        keys = k[batch : batch + 1, :, padding:length].contiguous()
        vals = v[batch : batch + 1, :, padding:length].contiguous()
        scores = query.double() @ keys.double().repeat_interleave(2, dim=1).transpose(-1, -2) / math.sqrt(dim)
        n = length - padding
        causal = torch.arange(n, device="cuda")[:, None] >= torch.arange(n, device="cuda")[None, :]
        scores = scores.masked_fill(~causal, -torch.inf)
        oracle = (scores.softmax(-1) @ vals.double().repeat_interleave(2, dim=1)).transpose(1, 2)
        torch.testing.assert_close(actual[batch : batch + 1, padding:].double(), oracle, atol=2e-3, rtol=1e-3)
        single = causal_gqa(query, keys, vals, lengths[:1] - padding, torch.zeros_like(left[:1]), dim**-0.5)
        torch.testing.assert_close(actual[batch : batch + 1, padding:], single, rtol=0, atol=0)
    assert torch.count_nonzero(actual[1, :17]) == 0
    decoded = causal_gqa(q[:, :, -1:], k, v, lengths, left, dim**-0.5)
    torch.testing.assert_close(decoded, actual[:, -1:], atol=0, rtol=0)


def test_graph_reads_updated_lengths_without_reading_unused_capacity():
    from verl.experimental.rollout_policy.invariant_attention import causal_gqa

    torch.manual_seed(9)
    keys = torch.randn(2, 1, 150, 32, device="cuda", dtype=torch.float16)
    values = torch.randn_like(keys)
    query = torch.randn(2, 2, 1, 32, device="cuda", dtype=torch.float16)
    lengths = torch.tensor([65, 65], device="cuda", dtype=torch.int32)
    left = torch.tensor([0, 17], device="cuda", dtype=torch.int32)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        causal_gqa(query, keys, values, lengths, left, 32**-0.5)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = causal_gqa(query, keys, values, lengths, left, 32**-0.5)
    lengths.fill_(137)
    graph.replay()
    expected = causal_gqa(query, keys, values, lengths, left, 32**-0.5)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
