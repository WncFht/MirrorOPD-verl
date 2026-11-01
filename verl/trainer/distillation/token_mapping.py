# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""One-to-one ID relabeling for proven equivalent tokenizers, never re-tokenization."""

import hashlib
import json
from dataclasses import dataclass, field

import torch


def tokenizer_definition(tokenizer):
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        raise ValueError("Pair acceptance requires a fast tokenizer definition")
    definition = json.loads(backend.to_str())
    for name in ("padding", "truncation", "post_processor"):
        definition.pop(name, None)
    added = definition.pop("added_tokens", [])
    if not isinstance(definition["model"].get("vocab"), dict):
        raise ValueError("Only tokenizers with explicit token-to-ID vocabularies are accepted")
    definition["model"].pop("vocab", None)
    return definition, {t["content"]: {k: v for k, v in t.items() if k != "id"} for t in added}


@dataclass
class TokenIdMapping:
    source_to_target: tuple[int, ...] = ()
    target_to_source: tuple[int, ...] = ()
    _device_tables: dict = field(default_factory=dict, repr=False)

    @classmethod
    def build(cls, source, target):
        a, b = source.get_vocab(), target.get_vocab()
        sd, sa = tokenizer_definition(source)
        td, ta = tokenizer_definition(target)
        if sd != td or any(ta.get(token) != item for token, item in sa.items()):
            raise ValueError("Tokenizers differ in segmentation, normalization, decoder or added-token semantics")
        if not set(a) <= set(b) or len(set(a.values())) != len(a) or len(set(b.values())) != len(b):
            raise ValueError("A bijective mapping of all source tokens is required")
        if all(b[token] == i for token, i in a.items()):
            return cls()
        forward, reverse = [-1] * (max(a.values()) + 1), [-1] * (max(b.values()) + 1)
        for token, i in a.items():
            forward[i], reverse[b[token]] = b[token], i
        return cls(tuple(forward), tuple(reverse))

    @property
    def is_identity(self):
        return not self.source_to_target

    @property
    def mapping_id(self):
        return (
            "identity" if self.is_identity else hashlib.sha256(json.dumps(self.source_to_target).encode()).hexdigest()
        )

    def tokens(self, values, *, reverse=False):
        if self.is_identity:
            return values
        table = self.target_to_source if reverse else self.source_to_target
        if any(i < 0 or i >= len(table) or table[i] < 0 for i in values):
            raise ValueError("Trajectory contains an ID outside the verified tokenizer mapping")
        return [table[i] for i in values]

    def tensor(self, values):
        if self.is_identity:
            return values
        if values.numel() and (values.min().item() < 0 or values.max().item() >= len(self.source_to_target)):
            raise ValueError("Input ID exceeds the verified mapping")
        key = str(values.device)
        if key not in self._device_tables:
            self._device_tables[key] = torch.tensor(self.source_to_target, device=values.device, dtype=torch.long)
        result = self._device_tables[key][values]
        if torch.any(result < 0):
            raise ValueError("Input ID is absent from the verified mapping")
        return result
