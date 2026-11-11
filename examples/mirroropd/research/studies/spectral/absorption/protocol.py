# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Run paths and cancellation are supplied by callers, never inferred from source."""

from dataclasses import dataclass
from pathlib import Path

import torch

SCOPE = r"model\.layers\.\d+\.(self_attn|mlp)\..*\.weight$"


@dataclass(frozen=True)
class RunContext:
    root: Path
    projection: str = "cpu-reference"
    stop_paths: tuple[Path, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "root", Path(self.root).absolute())
        if self.projection not in {"cpu-reference", "cuda-stream"}:
            raise ValueError(f"Unknown projection backend: {self.projection}")

    def check_stop(self):
        for path in (self.root / "STOP", *self.stop_paths):
            if path.exists():
                raise InterruptedError(f"Experiment stopped by {path}")


def save_tensor(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
