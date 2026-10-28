# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Single-query GQA decode directly from the actual prefix KV, without repeat/unpad copies."""

from contextlib import contextmanager
from contextvars import ContextVar

import torch
from transformers import AttentionInterface

_CONTEXT = ContextVar("mirroropd_flash_kv", default=None)
_NAME = "mirroropd_flash_kv"


def _attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    from flash_attn import flash_attn_with_kvcache

    context = _CONTEXT.get()
    if context is None or query.shape[-2] != 1 or dropout or module.training:
        raise ValueError("Flash KV attention requires an active single-token inference context")
    if attention_mask is not None:
        raise ValueError("Additional attention constraints are not supported by Flash KV decode")
    lengths, leftpad, splits, capacity = context
    keys, values = key.transpose(1, 2), value.transpose(1, 2)
    if capacity is not None:
        keys = keys.as_strided((keys.shape[0], capacity, keys.shape[2], keys.shape[3]), keys.stride())
        values = values.as_strided((values.shape[0], capacity, values.shape[2], values.shape[3]), values.stride())
    output = flash_attn_with_kvcache(
        query.transpose(1, 2),
        keys,
        values,
        cache_seqlens=lengths,
        cache_leftpad=leftpad,
        softmax_scale=scaling,
        causal=True,
        num_splits=splits,
    )
    return output, None


AttentionInterface.register(_NAME, _attention)


@contextmanager
def flash_kv_context(model, mask, *, splits=0, cache=None):
    if model.config.model_type not in {"qwen3", "qwen3_moe"}:
        raise ValueError("Flash KV fusion has been implemented for Qwen3 dense/MoE only")
    device = next(model.parameters()).device
    if device.type != "cuda" or mask.ndim != 2:
        raise ValueError("Flash KV fusion requires CUDA and a 2D prefix padding mask")
    leftpad = (mask == 0).sum(-1, dtype=torch.int32).to(device, non_blocking=True)
    lengths = torch.full((mask.shape[0],), mask.shape[1], dtype=torch.int32, device=device)
    # DynamicCache exposes only the current prefix. AppendCache exposes a view
    # into fixed storage whose full capacity can be used for graph replay.
    capacity = getattr(cache.layers[0], "capacity", None) if cache is not None else None
    with flash_kv_state(model, lengths, leftpad, splits=splits, capacity=capacity):
        yield


@contextmanager
def flash_kv_state(model, lengths, leftpad, *, splits=0, capacity=None):
    previous = model.config._attn_implementation
    token = _CONTEXT.set((lengths, leftpad, splits, capacity))
    try:
        model.config._attn_implementation = (
            "mirroropd_invariant_attention" if getattr(model, "_fusion_invariant_attention", False) else _NAME
        )
        yield
    finally:
        model.config._attn_implementation = previous
        _CONTEXT.reset(token)
