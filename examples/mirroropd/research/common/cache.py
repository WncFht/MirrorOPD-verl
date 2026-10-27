# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Hidden-state caches with explicit model/trajectory identity, never file hashes.

Old tensor-only caches are left untouched. A new namespace stores the actual
input IDs alongside the tensor, so a reused ID cannot silently select a stale
prefix. Model paths must identify immutable weights; callers loading a new
checkpoint must use that checkpoint's path.
"""

from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import quote

import torch

FIELDS = ("source", "id", "replicate", "seed", "generation_seed", "generation_checkpoint", "generation_arm")


def model_identity(model) -> dict:
    config = model.config
    path = str(getattr(config, "_name_or_path", ""))
    if not path:
        raise ValueError("Cached models require an explicit immutable model/checkpoint path")
    return {
        "path": str(Path(path).resolve()),
        "model_type": config.model_type,
        "parameter_dtype": str(next(model.parameters()).dtype),
        "attention": getattr(config, "_attn_implementation", None),
        "hidden_protocol": "native-bf16-backbone-v1",
    }


def trajectory_identity(row) -> dict:
    if not row.get("prompt_ids") or not row.get("response_ids"):
        raise ValueError("A trajectory must contain nonempty prompt and response token IDs")
    return {key: row.get(key) for key in FIELDS} | {
        "prompt_ids": list(row["prompt_ids"]),
        "response_ids": list(row["response_ids"]),
    }


class HiddenCache:
    """Cache one frozen model's response-aligned hidden states on CPU."""

    def __init__(self, root: Path, namespace: str, model: dict):
        self.root = Path(root) / "validated-hidden-v1" / quote(str(namespace), safe="")
        self.model = model

    def path(self, row) -> Path:
        parts = [f"{key}={quote(str(row.get(key)), safe='')}" for key in FIELDS]
        return self.root.joinpath(*parts, "hidden.pt")

    def load(self, row):
        payload = torch.load(self.path(row), map_location="cpu", weights_only=True)
        expected = {"model": self.model, "trajectory": trajectory_identity(row)}
        if not isinstance(payload, dict) or payload.get("identity") != expected:
            raise ValueError(
                f"Hidden cache identity mismatch: {self.path(row)}; use a separate cache for changed inputs/models"
            )
        hidden = payload["hidden"]
        if hidden.ndim != 2 or len(hidden) != len(row["response_ids"]):
            raise ValueError(f"Invalid response-aligned hidden shape: {self.path(row)}")
        return hidden

    def get(self, row, compute):
        trajectory = trajectory_identity(row)
        path = self.path(row)
        if path.exists():
            return self.load(row)
        hidden = compute().detach().cpu()
        if hidden.ndim != 2 or len(hidden) != len(row["response_ids"]):
            raise ValueError("Computed hidden states do not match response token count")
        payload = {"identity": {"model": self.model, "trajectory": trajectory}, "hidden": hidden}
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with NamedTemporaryFile(dir=path.parent, prefix="hidden.", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
            torch.save(payload, temporary)
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return hidden
