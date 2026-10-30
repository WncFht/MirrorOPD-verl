# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Distinct SFT checkpoints share a repository but must never share model/cache identities."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from examples.mirroropd.research.studies.fixed_transfer.spec import Model, fingerprint
from examples.mirroropd.research.studies.fixed_transfer.stage_assets import selected_files


def test_subfolders_are_distinct_and_legacy_identity_is_unchanged():
    model = Model(name="test", repo="org/model", revision="a" * 40, path="unused")
    assert model.identity == fingerprint(dict(repo=model.repo, revision=model.revision))
    a, b = replace(model, subfolder="step80"), replace(model, subfolder="step640")
    assert len({model.identity, a.identity, b.identity}) == 3
    with pytest.raises(ValueError, match="safe relative"):
        replace(model, subfolder="../step80")


def test_only_the_requested_checkpoint_is_in_the_manifest():
    model = Model(name="test", repo="org/model", revision="a" * 40, path="unused", subfolder="step80")
    files = [
        dict(path=f"{folder}/{name}", size=1, sha256="b" * 64)
        for folder in ["step80", "step640", "step800"]
        for name in ["config.json", "model.safetensors"]
    ]
    manifest = dict(repo=model.repo, revision=model.revision, files=files)
    assert {f["path"] for f in selected_files(model, manifest)} == {"step80/config.json", "step80/model.safetensors"}
    manifest["revision"] = "c" * 40
    with pytest.raises(ValueError, match="pinned"):
        selected_files(model, manifest)


def test_assets_reject_missing_subfolder_manifest_without_root_fallback(tmp_path, monkeypatch):
    from examples.mirroropd.research.studies.fixed_transfer import models, stage_assets

    def forbidden(model):
        pytest.fail("A checkpoint subfolder must never fall back to repository-root assets")

    monkeypatch.setattr(models, "materialize", forbidden)
    model = Model(name="Q1", repo="org/model", revision="a" * 40, path=str(tmp_path / "step80"), subfolder="step80")
    study = SimpleNamespace(models={"Q1": model}, directory=tmp_path / "study")
    with pytest.raises(ValueError, match="Freeze a subfolder manifest"):
        stage_assets.assets(study)
