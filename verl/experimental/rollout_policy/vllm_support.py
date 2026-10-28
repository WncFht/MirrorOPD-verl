# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Apply the same tokenizer-ID support as HF fusion without per-request vocab payloads."""

import torch
from transformers import AutoTokenizer
from vllm.v1.sample.logits_processor.interface import LogitsProcessor


class TokenizerSupport(LogitsProcessor):
    def __init__(self, vllm_config, device, is_pin_memory):
        model = vllm_config.model_config
        tokenizer = AutoTokenizer.from_pretrained(
            model.tokenizer or model.model, local_files_only=True, trust_remote_code=model.trust_remote_code
        )
        self.vocab_size = model.get_vocab_size()
        legal = set(tokenizer.get_vocab().values())
        if not legal or min(legal) < 0 or max(legal) >= self.vocab_size:
            raise ValueError("Tokenizer IDs exceed the vLLM output head")
        self.invalid_ids = torch.tensor(sorted(set(range(self.vocab_size)) - legal), device=device, dtype=torch.long)

    def is_argmax_invariant(self):
        return False

    def update_state(self, batch_update):
        # The support is model-wide and independent of request positions.
        pass

    def apply(self, logits):
        if self.invalid_ids.numel():
            logits.index_fill_(-1, self.invalid_ids, -torch.inf)
        if logits.shape[-1] > self.vocab_size:
            logits[..., self.vocab_size :] = -torch.inf
        return logits
