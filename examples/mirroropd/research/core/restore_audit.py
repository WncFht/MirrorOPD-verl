# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Exact restored weight, optimizer and RNG audit shared by single- and multi-device studies."""

from pathlib import Path

import torch
from safetensors import safe_open

from examples.mirroropd.research.core import training_state


def verify_loaded(directory, weights, model, optimizer):
    directory, weights = Path(directory), Path(weights)
    expected = torch.load(directory / "optimizer.pt", map_location="cpu", weights_only=True)
    current_rng = training_state.rng_state()
    assert current_rng["python"] == expected["rng"]["python"]
    assert current_rng["numpy"] == expected["rng"]["numpy"]
    assert torch.equal(current_rng["torch"], expected["rng"]["torch"])
    if current_rng["cuda"] is None:
        assert expected["rng"]["cuda"] is None
    else:
        assert len(current_rng["cuda"]) == len(expected["rng"]["cuda"])
        assert all(torch.equal(a, b) for a, b in zip(current_rng["cuda"], expected["rng"]["cuda"], strict=True))
    actual = optimizer.state_dict()
    assert actual["param_groups"] == expected["optimizer"]["param_groups"]
    checked_slots = 0
    for key, slots in expected["optimizer"]["state"].items():
        for slot, tensor in slots.items():
            value = actual["state"][key][slot]
            if torch.is_tensor(tensor):
                assert value.dtype == tensor.dtype
                assert torch.equal(value.detach().cpu(), tensor)
                checked_slots += 1
            else:
                assert value == tensor
    del actual, expected
    aliases = {}
    for alias, param in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(param), []).append(alias)
    sources = {}
    for path in weights.glob("*.safetensors"):
        with safe_open(path, framework="pt", device="cpu") as f:
            sources.update({key: path for key in f.keys()})
    checked_parameters = 0
    for name, param in model.named_parameters():
        key = next(k for k in aliases[id(param)] if k in sources)
        with safe_open(sources[key], framework="pt", device="cpu") as f:
            tensor = f.get_tensor(key)
            assert tuple(tensor.shape) == tuple(param.shape) and tensor.dtype == param.dtype
            a, b = tensor.reshape(-1), param.detach().reshape(-1)
            for start in range(0, param.numel(), 2**20):
                assert torch.equal(a[start : start + 2**20], b[start : start + 2**20].cpu())
        checked_parameters += param.numel()
    return checked_parameters, checked_slots
