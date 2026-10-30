# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Retention rejects final/corrupt weights and preserves evaluation evidence."""

import pytest

from examples.mirroropd.research.core.artifacts import file_hash, receipt
from examples.mirroropd.research.core.io import dump as write
from examples.mirroropd.research.core.io import read
from examples.mirroropd.research.studies.fixed_transfer.retention import prune_weights


def fixture(path, transient=True):
    files = []
    for name in ["model-1.safetensors", "model-2.safetensors"]:
        source = path / name
        source.write_bytes(b"synthetic weight bytes")
        files.append(dict(name=name, bytes=source.stat().st_size, sha256=file_hash(source)))
    receipt(path, dict(step=64), files=files, transient=transient)
    (path / "config.json").write_text("{}")
    return files


def test_prune_dry_run_and_preserves_evidence(tmp_path):
    files = fixture(tmp_path)
    size = sum(x["bytes"] for x in files)
    assert prune_weights(tmp_path) == size
    assert all((tmp_path / x["name"]).exists() for x in files)
    assert prune_weights(tmp_path, apply=True) == size
    assert not list(tmp_path.glob("*.safetensors"))
    assert (tmp_path / "config.json").exists()
    assert read(tmp_path / "complete.json")["weights_pruned"]
    assert prune_weights(tmp_path, apply=True) == 0


def test_prune_checks_every_file_before_any_delete(tmp_path):
    files = fixture(tmp_path)
    (tmp_path / files[1]["name"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="content changed"):
        prune_weights(tmp_path, apply=True)
    assert (tmp_path / files[0]["name"]).exists()


def test_prune_rejects_final_and_unexplained_missing_file(tmp_path):
    files = fixture(tmp_path, transient=False)
    with pytest.raises(ValueError, match="final weights"):
        prune_weights(tmp_path, apply=True)
    saved = read(tmp_path / "complete.json")
    saved["transient"] = True
    write(tmp_path / "complete.json", saved)
    (tmp_path / files[0]["name"]).unlink()
    with pytest.raises(ValueError, match="missing before prune"):
        prune_weights(tmp_path, apply=True)
    saved["prune_started_at"] = 1
    write(tmp_path / "complete.json", saved)
    prune_weights(tmp_path, apply=True)
    assert read(tmp_path / "complete.json")["weights_pruned"]
