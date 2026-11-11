# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Small atomic work units for expensive diagnostic tasks."""

import fcntl
import json
from pathlib import Path

import torch

from .runtime import write_json, write_tensors


class CheckpointStore:
    """One writer per task; bind every completed part to the same inputs.

    Store small spectra, requests and scalar responses only. Full parameter
    directions remain transient in the caller's device or host memory.
    """

    def __init__(self, root, signature):
        self.root = Path(root)
        self.signature = json.loads(json.dumps(signature))
        self.lock = None

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / "writer.lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            identity = self.root / "identity.json"
            if identity.exists():
                if json.loads(identity.read_text()) != self.signature:
                    raise ValueError(f"Stale partial results: {self.root}")
            else:
                write_json(identity, self.signature)
        except BaseException:
            self.lock.close()
            self.lock = None
            raise
        return self

    def __exit__(self, *exc):
        self.lock.close()
        self.lock = None

    def tensors(self, key):
        path = self.root / f"{key}.pt"
        return torch.load(path, map_location="cpu", weights_only=True) if path.exists() else None

    def json(self, key):
        path = self.root / f"{key}.json"
        return json.loads(path.read_text()) if path.exists() else None

    def save_tensors(self, key, value):
        write_tensors(self.root / f"{key}.pt", value)

    def save_json(self, key, value):
        write_json(self.root / f"{key}.json", value)
