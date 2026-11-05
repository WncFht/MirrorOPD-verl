# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Exercise retention only after a complete, synchronously committed state."""

import json

import pytest
from omegaconf import OmegaConf

from examples.mirroropd.research.studies.weighted_targets.checkpoint import LatestTrainingState
from examples.mirroropd.research.studies.weighted_targets.metrics import trusted_metrics


def state(tmp_path, step):
    root = tmp_path / "checkpoints"
    current = root / f"global_step_{step}"
    actor = current / "actor"
    hf = actor / "huggingface"
    hf.mkdir(parents=True)
    for name in ("data.pt", "trainer_state.json"):
        (current / name).write_text("state")
    for kind in ("model", "optim", "extra_state"):
        for rank in range(4):
            (actor / f"{kind}_world_size_4_rank_{rank}.pt").write_text("shard")
    (hf / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": "model.safetensors"}}))
    (hf / "model.safetensors").write_text("weights")
    (root / "latest_checkpointed_iteration.txt").write_text(str(step))
    config = OmegaConf.create(
        {"trainer": {"default_local_dir": str(root), "total_training_steps": 100, "n_gpus_per_node": 4, "nnodes": 1}}
    )
    return LatestTrainingState(config), current


def test_retention_removes_old_entire_state_and_exports_final_model(tmp_path):
    _, old = state(tmp_path, 75)
    callback, current = state(tmp_path, 100)
    callback.on_save(None, 100, str(current))
    assert not old.exists() and current.exists()
    assert (tmp_path / "final-model/model.safetensors").read_text() == "weights"
    assert (current / "actor/optim_world_size_4_rank_3.pt").exists()


def test_incomplete_save_keeps_previous_state(tmp_path):
    _, old = state(tmp_path, 25)
    callback, current = state(tmp_path, 50)
    (current / "actor/optim_world_size_4_rank_3.pt").unlink()
    with pytest.raises(ValueError, match="Incomplete training state"):
        callback.on_save(None, 50, str(current))
    assert old.exists()


def test_metrics_exclusion_requires_matching_bytes_and_preserves_future_appends(tmp_path):
    import hashlib

    path = tmp_path / "metrics.jsonl"
    bad = b'{"duplicate":1}\n'
    path.write_bytes(b"{}\n" + bad + b'{"future":1}\n')
    path.with_name("metrics-exclusions.json").write_text(
        json.dumps(
            {"excluded_byte_ranges": [{"start": 3, "end": 3 + len(bad), "sha256": hashlib.sha256(bad).hexdigest()}]}
        )
    )
    assert trusted_metrics(path)[0] == b'{}\n{"future":1}\n'
    path.write_bytes(path.read_bytes().replace(b"duplicate", b"corrupted"))
    with pytest.raises(ValueError, match="bytes changed"):
        trusted_metrics(path)
