# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Request-local decode graphs; the first real decode warms the capture stream."""

import time

import torch

from .flash_kv import flash_kv_state


class DecodeGraph:
    def __init__(self, model, ids, positions, mask, cache, extra, *, splits=0):
        self.device = next(model.parameters()).device
        self.model, self.cache, self.extra = model, cache, extra
        self.ids = ids.to(self.device).clone()
        self.positions = positions.to(self.device).clone()
        self.write_position = torch.tensor([mask.shape[1] - 1], device=self.device, dtype=torch.long)
        self.lengths = torch.full((ids.shape[0],), mask.shape[1], device=self.device, dtype=torch.int32)
        self.leftpad = (mask == 0).sum(-1, dtype=torch.int32).to(self.device)
        self.splits = splits
        self._set_length(mask.shape[1])
        for layer in cache.layers:
            layer.graph_position = self.write_position
        self.graph = None
        self.capture_seconds = 0.0
        self.stream = torch.cuda.Stream(device=self.device)
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            self.output = self._call()
        # This output is used for sampling: warmup never adds a duplicate forward.
        torch.cuda.current_stream(self.device).wait_stream(self.stream)

    def _set_length(self, length):
        for layer in self.cache.layers:
            if length > layer.capacity:
                raise ValueError("CUDA graph exceeded its KV capacity")
            layer.length = length

    def _context(self):
        return flash_kv_state(
            self.model, self.lengths, self.leftpad, splits=self.splits, capacity=self.cache.layers[0].capacity
        )

    def _call(self):
        with self._context():
            return self.model(
                input_ids=self.ids,
                position_ids=self.positions,
                attention_mask=None,
                past_key_values=self.cache,
                use_cache=True,
                **self.extra,
            )

    def replay(self, ids, positions, mask):
        if ids.shape != self.ids.shape:
            raise ValueError("CUDA graph batch shape changed without invalidation")
        self._set_length(mask.shape[1])
        self.ids.copy_(ids)
        self.positions.copy_(positions)
        self.write_position.fill_(mask.shape[1] - 1)
        self.lengths.fill_(mask.shape[1])
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        if self.graph is None:
            start = time.monotonic()
            self.stream.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.device(self.device), self._context():
                with torch.cuda.graph(self.graph, stream=self.stream, capture_error_mode="thread_local"):
                    self.output = self.model(
                        input_ids=self.ids,
                        position_ids=self.positions,
                        attention_mask=None,
                        past_key_values=self.cache,
                        use_cache=True,
                        **self.extra,
                    )
            self.capture_seconds = time.monotonic() - start
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            self.graph.replay()
        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        return self.output
