# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Single-teacher hidden layout and frozen output head (outside student state)."""

import json
from pathlib import Path

import torch
from safetensors import safe_open


def prediction_positions(input_ids, loss_mask):
    """Per-sample positions predicting the masked response, then packed positions.

    loss_mask uses the response layout, including masked tool-result positions.
    A full-sequence mask is also accepted if its first target is masked out.
    """
    positions, packed = [], []
    offset = 0
    for ids, mask in zip(input_ids.unbind(), loss_mask.unbind(), strict=True):
        if mask.ndim != 1 or len(mask) > len(ids) or not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("full KL requires a binary response/full-sequence loss mask")
        local = mask.bool().nonzero().flatten() + len(ids) - len(mask) - 1
        if local.numel() and (local.min() < 0 or local.max() >= len(ids) - 1):
            raise ValueError("A valid next-token target must have a preceding input position")
        positions.append(local)
        packed.append(local + offset)
        offset += len(ids)
    return positions, torch.cat(packed)


class FrozenTeacherHead:
    """Load a dense Qwen safetensors output weight; never register on the student."""

    def __init__(self, path):
        from verl.utils.fs import copy_to_local

        root = Path(copy_to_local(path))
        config = json.loads((root / "config.json").read_text())
        if config.get("model_type") not in ("qwen2", "qwen3"):
            raise ValueError("full reverse KL requires a dense Qwen2/Qwen3 teacher head")
        key = "model.embed_tokens.weight" if config.get("tie_word_embeddings", False) else "lm_head.weight"
        index = root / "model.safetensors.index.json"
        files = (
            [root / json.loads(index.read_text())["weight_map"][key]]
            if index.exists()
            else [root / "model.safetensors"]
        )
        with safe_open(files[0], framework="pt", device="cpu") as stream:
            self.cpu_weight = stream.get_tensor(key).detach().contiguous()
        if self.cpu_weight.shape != (config["vocab_size"], config["hidden_size"]):
            raise ValueError("teacher head does not match model configuration")
        self.weight = None

    def get(self, device, dtype):
        if self.weight is None or self.weight.device != device or self.weight.dtype != dtype:
            self.weight = self.cpu_weight.to(device=device, dtype=dtype)
        return self.weight

    def release(self):
        self.weight = None
