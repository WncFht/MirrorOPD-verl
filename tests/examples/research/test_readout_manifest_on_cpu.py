# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Readout audits distinguish missing data, deliberate pruning, and changed configs."""

import pytest

from examples.mirroropd.research.core.artifacts import file_hash, fingerprint, receipt
from examples.mirroropd.research.core.io import dump as write
from examples.mirroropd.research.core.io import read
from examples.mirroropd.research.studies.fixed_transfer.training import validate_checkpoint


def make_checkpoint(path, step=64):
    (path / "config.json").write_text("{}")
    (path / "model.safetensors").write_bytes(b"original")
    files = [dict(name="model.safetensors", bytes=8, sha256=file_hash(path / "model.safetensors"))]
    receipt(
        path,
        dict(step=step, arm=dict(updates=320)),
        transient=step != 320,
        dtype="float32",
        optimizer_saved=False,
        files=files,
        weight_identity=fingerprint(dict(weights=files, config=file_hash(path / "config.json"))),
    )


def test_readout_detects_config_and_same_size_content_change(tmp_path):
    make_checkpoint(tmp_path)
    validate_checkpoint(tmp_path, verify_hash=True)
    (tmp_path / "model.safetensors").write_bytes(b"changed!")
    with pytest.raises(ValueError, match="content changed"):
        validate_checkpoint(tmp_path, verify_hash=True)
    (tmp_path / "config.json").write_text('{"changed": true}')
    with pytest.raises(ValueError, match="configuration or manifest changed"):
        validate_checkpoint(tmp_path)


def test_missing_weights_need_explicit_transient_cleanup(tmp_path):
    make_checkpoint(tmp_path)
    (tmp_path / "model.safetensors").unlink()
    with pytest.raises(ValueError, match="weight missing"):
        validate_checkpoint(tmp_path)
    saved = read(tmp_path / "complete.json")
    saved["prune_started_at"] = 1
    write(tmp_path / "complete.json", saved)
    validate_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="new evaluation"):
        validate_checkpoint(tmp_path, verify_hash=True)


def test_final_weights_cannot_be_pruned(tmp_path):
    make_checkpoint(tmp_path, step=320)
    saved = read(tmp_path / "complete.json")
    saved["weights_pruned"] = True
    write(tmp_path / "complete.json", saved)
    with pytest.raises(ValueError, match="Final checkpoint"):
        validate_checkpoint(tmp_path)
