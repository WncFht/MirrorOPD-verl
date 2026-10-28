# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Causal GQA with one logical KV reduction order for prefill and decode."""

import torch
import triton
import triton.language as tl
from transformers import AttentionInterface
from transformers.masking_utils import AttentionMaskInterface

NAME = "mirroropd_invariant_attention"


@triton.jit(do_not_specialize=["QLEN", "SQB", "SQH", "SQS", "SKB", "SKH", "SKS", "SVB", "SVH", "SVS"])
def _causal_gqa(
    Q,
    K,
    V,
    Lengths,
    Leftpad,
    Out,
    QLEN,
    SQB,
    SQH,
    SQS,
    SKB,
    SKH,
    SKS,
    SVB,
    SVH,
    SVS,
    HQ: tl.constexpr,
    GROUP: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
):
    tile, head, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    length = tl.load(Lengths + batch)
    left = tl.load(Leftpad + batch)
    valid_length = length - left
    query_index = tile * 16 + tl.arange(0, 16)
    query_position = length - QLEN + query_index - left
    dims = tl.arange(0, D)
    q = tl.load(
        Q + batch * SQB + head * SQH + query_index[:, None] * SQS + dims[None, :],
        (query_index[:, None] < QLEN) & (query_position[:, None] >= 0),
        0,
    )
    maximum = tl.full((16,), -float("inf"), tl.float32)
    denominator = tl.zeros((16,), tl.float32)
    accumulator = tl.zeros((16, D), tl.float32)
    upper = tl.minimum(valid_length, length - QLEN + tl.minimum((tile + 1) * 16, QLEN) - left)
    for block in range(tl.cdiv(tl.maximum(upper, 0), 64)):
        key_index = block * 64 + tl.arange(0, 64)
        key_head = head // GROUP
        keys = tl.load(
            K + batch * SKB + key_head * SKH + (left + key_index[None, :]) * SKS + dims[:, None],
            key_index[None, :] < valid_length,
            0,
        )
        scores = tl.dot(q, keys).to(tl.float32) * SCALE
        allowed = (key_index[None, :] < valid_length) & (key_index[None, :] <= query_position[:, None])
        scores = tl.where(allowed, scores, -float("inf"))
        next_maximum = tl.maximum(maximum, tl.max(scores, axis=1))
        safe_maximum = tl.where(next_maximum == -float("inf"), 0.0, next_maximum)
        correction = tl.exp(maximum - safe_maximum)
        probabilities = tl.exp(scores - safe_maximum[:, None])
        denominator = denominator * correction + tl.sum(probabilities, axis=1)
        accumulator = accumulator * correction[:, None]
        values = tl.load(
            V + batch * SVB + key_head * SVH + (left + key_index[:, None]) * SVS + dims[None, :],
            key_index[:, None] < valid_length,
            0,
        )
        accumulator = tl.dot(probabilities.to(values.dtype), values, accumulator)
        maximum = next_maximum
    output = accumulator / tl.where(denominator > 0, denominator, 1.0)[:, None]
    tl.store(
        Out + ((batch * QLEN + query_index[:, None]) * HQ + head) * D + dims[None, :],
        output,
        query_index[:, None] < QLEN,
    )


def causal_gqa(query, key, value, lengths, leftpad, scaling):
    batch, heads, q_length, dim = query.shape
    if dim not in (32, 64, 128, 256) or heads % key.shape[1] or key.shape != value.shape:
        raise ValueError("Unsupported invariant GQA shape")
    if query.dtype not in (torch.float16, torch.bfloat16) or key.dtype != query.dtype or value.dtype != query.dtype:
        raise ValueError("Invariant GQA requires matching FP16/BF16 Q/K/V")
    if any(tensor.stride(-1) != 1 for tensor in (query, key, value)):
        raise ValueError("Invariant GQA requires contiguous head dimensions")
    output = torch.empty((batch, q_length, heads, dim), device=query.device, dtype=query.dtype)
    _causal_gqa[(triton.cdiv(q_length, 16), heads, batch)](
        query,
        key,
        value,
        lengths,
        leftpad,
        output,
        q_length,
        *query.stride()[:3],
        *key.stride()[:3],
        *value.stride()[:3],
        heads,
        heads // key.shape[1],
        dim,
        scaling,
        num_warps=4,
        num_stages=2,
    )
    return output


def _attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    from .flash_kv import _CONTEXT

    if dropout or module.training:
        raise ValueError("Invariant attention is an inference-only executor")
    if attention_mask is not None and attention_mask.ndim != 2:
        raise ValueError("Invariant attention supports a 2D left-padding mask only")
    context = _CONTEXT.get()
    if context is not None:
        lengths, leftpad, _, _ = context
    else:
        lengths = torch.full((query.shape[0],), key.shape[-2], device=query.device, dtype=torch.int32)
        leftpad = (
            torch.zeros_like(lengths) if attention_mask is None else (attention_mask == 0).sum(-1, dtype=torch.int32)
        )
    return causal_gqa(query, key, value, lengths, leftpad, scaling or query.shape[-1] ** -0.5), None


def _padding_mask(attention_mask=None, **kwargs):
    return attention_mask


AttentionInterface.register(NAME, _attention)
AttentionMaskInterface.register(NAME, _padding_mask)
