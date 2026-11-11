# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""FP64 direction reductions, streaming one matrix onto the execution device.

Direction payloads are immutable FP32 tensors. Cached host tensors and norms
avoid per-site deserialization; GPU storage is bounded by one parameter matrix,
not the total parameter count. No approximation or TF32 reduction is used.
"""

import torch

BANDS = ("learned", "off", "null")


def reduce_projection(gradient, directions, *, device):
    """Return gradient norm and raw direction dot/norm, reducing in FP64.

    Zero directions have a zero dot/norm and a None unit derivative. Results are
    Python scalars; gradient/direction shape mismatches are errors.
    """
    gradient = gradient.detach().to(device=device, dtype=torch.float64)
    scalars = [gradient.norm()]
    for direction in directions.values():
        if direction.shape != gradient.shape:
            raise ValueError("Gradient and direction shapes differ")
        value = direction.detach().to(device=device, dtype=torch.float64)
        scalars.extend(((gradient * value).sum(), value.norm()))
    values = torch.stack(scalars).cpu().tolist()
    result = {}
    for index, name in enumerate(directions):
        dot, length = values[1 + 2 * index : 3 + 2 * index]
        result[name] = dict(dot=dot, norm=length, unit_derivative=dot / length if length else None)
    return values[0], result


class DirectionProjector:
    """Load each immutable matrix once; cache only the requested direction bands.

    The reference keeps mmap-backed tensors; CUDA pins only the selected FP32 bands.
    GPU projections transfer one matrix at a time. CUDA mode requires CUDA
    gradients, preventing silent accidental execution on the wrong device.
    """

    def __init__(self, directory, *, backend="cpu-reference"):
        if backend not in {"cpu-reference", "cuda-stream"}:
            raise ValueError(f"Unknown projection backend: {backend}")
        self.directory = directory
        self.backend = backend
        self.cache = {}
        self.norms = {}

    def directions(self, index):
        if index not in self.cache:
            data = torch.load(self.directory / f"{index:04d}.pt", weights_only=True, mmap=True)
            self.cache[index] = {
                name: data[name].pin_memory() if self.backend == "cuda-stream" else data[name] for name in BANDS
            }
        return self.cache[index]

    def project(self, index, gradient):
        if self.backend == "cpu-reference":
            return reduce_projection(gradient, self.directions(index), device="cpu")
        if gradient.device.type != "cuda":
            raise ValueError("cuda-stream requires a CUDA gradient")
        g = gradient.detach().double()
        scalars = [g.norm()]
        directions = self.directions(index)
        known = index in self.norms
        for direction in directions.values():
            if direction.shape != gradient.shape:
                raise ValueError("Gradient and direction shapes differ")
            value = direction.to(device=g.device, non_blocking=True).double()
            scalars.append((g * value).sum())
            if not known:
                scalars.append(value.norm())
        values = torch.stack(scalars).cpu().tolist()
        if not known:
            self.norms[index] = values[2::2]
            dots = values[1::2]
        else:
            dots = values[1:]
        result = {}
        for name, dot, length in zip(directions, dots, self.norms[index], strict=True):
            result[name] = dict(dot=dot, norm=length, unit_derivative=dot / length if length else None)
        return values[0], result
