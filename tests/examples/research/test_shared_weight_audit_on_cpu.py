# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Reject ambiguous tied-weight checkpoints before HF chooses one alias."""

import json

import pytest
import torch
from safetensors.torch import save_file

from examples.mirroropd.research.studies.fixed_transfer.models import audit_shared_weights


def test_serialized_shared_weights_across_shards(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(dict(tie_word_embeddings=True)))
    weight = torch.arange(4100, dtype=torch.float32).reshape(2050, 2)
    save_file({"model.embed_tokens.weight": weight}, tmp_path / "first.safetensors")
    assert audit_shared_weights(tmp_path)["duplicate_equal"] is None
    save_file({"lm_head.weight": weight.clone()}, tmp_path / "second.safetensors")
    assert audit_shared_weights(tmp_path)["duplicate_equal"] is True
    weight[-1, -1] += 1
    save_file({"lm_head.weight": weight}, tmp_path / "second.safetensors")
    with pytest.raises(ValueError, match="values differ"):
        audit_shared_weights(tmp_path)
