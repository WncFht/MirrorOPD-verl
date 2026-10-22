# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Per-instance Qwen2 output adapter; preserve root AND LM-head FSDP hooks."""

from types import MethodType


def install_full_vocab_adapter(model):
    if model.config.model_type != "qwen2" or hasattr(model, "_full_vocab_original_forward"):
        return
    model._full_vocab_original_forward = model.forward
    model.lm_head._full_vocab_original_forward = model.lm_head.forward
    model.forward = MethodType(full_vocab_forward, model)
    model.lm_head.forward = MethodType(full_vocab_head_forward, model.lm_head)


def full_vocab_head_forward(self, hidden, *, full_vocab_callback=None):
    if full_vocab_callback is None:
        return self._full_vocab_original_forward(hidden)
    if self.bias is not None:
        raise ValueError("full reverse KL currently requires a bias-free LM head")
    return full_vocab_callback(hidden, self.weight)


def full_vocab_forward(self, *args, full_vocab_callback=None, **kwargs):
    if full_vocab_callback is None:
        return self._full_vocab_original_forward(*args, **kwargs)
    kwargs.update(return_dict=True, output_hidden_states=False, use_cache=False)
    output = self.model(*args, **kwargs)
    return full_vocab_callback(output.last_hidden_state, self.lm_head)
