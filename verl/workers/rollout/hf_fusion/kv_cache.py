# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded append-only KV storage with dynamic-length views and shared row selection."""

import torch
from transformers.cache_utils import DynamicCache, DynamicLayer


class AppendLayer(DynamicLayer):
    def __init__(self, capacity):
        super().__init__()
        self.capacity = capacity
        self.length = 0
        self.graph_position = None

    def lazy_initialization(self, key_states, value_states=None):
        value_states = key_states if value_states is None else value_states
        self.dtype, self.device = key_states.dtype, key_states.device
        # Sequence-major storage also admits direct FlashAttention KV-cache reads.
        batch, heads, _, dim = key_states.shape
        self.key_buffer = torch.empty((batch, self.capacity, heads, dim), dtype=self.dtype, device=self.device)
        self.value_buffer = torch.empty(
            (batch, self.capacity, value_states.shape[1], value_states.shape[-1]),
            dtype=value_states.dtype,
            device=value_states.device,
        )
        self.is_initialized = True
        self._views()

    def _views(self):
        self.keys = self.key_buffer[:, : self.length].transpose(1, 2)
        self.values = self.value_buffer[:, : self.length].transpose(1, 2)

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        if self.graph_position is not None:
            self.key_buffer.index_copy_(1, self.graph_position, key_states.transpose(1, 2))
            self.value_buffer.index_copy_(1, self.graph_position, value_states.transpose(1, 2))
            self.keys = self.key_buffer.transpose(1, 2)
            self.values = self.value_buffer.transpose(1, 2)
            return self.keys, self.values
        end = self.length + key_states.shape[-2]
        if end > self.capacity:
            raise ValueError(f"KV capacity exceeded: {end} > {self.capacity}")
        self.key_buffer[:, self.length : end].copy_(key_states.transpose(1, 2))
        self.value_buffer[:, self.length : end].copy_(value_states.transpose(1, 2))
        self.length = end
        self._views()
        return self.keys, self.values

    def get_seq_length(self):
        return self.length

    def batch_select_indices(self, indices):
        self.graph_position = None
        if self.is_initialized:
            self.key_buffer = self.key_buffer.index_select(0, indices.to(self.device))
            self.value_buffer = self.value_buffer.index_select(0, indices.to(self.device))
            self._views()

    def reorder_cache(self, indices):
        self.batch_select_indices(indices)

    def crop(self, max_length):
        if max_length < 0:
            max_length = max(0, self.length + max_length)
        self.length = min(self.length, max_length)
        if self.is_initialized:
            self._views()

    def reset(self):
        self.graph_position = None
        self.length = 0
        if self.is_initialized:
            self._views()


class AppendCache(DynamicCache):
    def __init__(self, config, capacity):
        if capacity < 1:
            raise ValueError("KV capacity must be positive")
        layers = getattr(config, "layer_types", None)
        if getattr(config, "sliding_window", None) is not None or (
            layers and any(layer != "full_attention" for layer in layers)
        ):
            raise ValueError("AppendCache currently supports full-attention transformer layers")
        super().__init__()
        self.layers = [AppendLayer(capacity) for _ in range(config.num_hidden_layers)]
