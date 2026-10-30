# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Model locations preserve scientific identities across worktrees and asset moves."""

import json

import pytest

from examples.mirroropd.assets.paths import checkpoint_path, hub_directory, registered_model, resolve_location


def test_canonical_paths_preserve_checkpoint_subfolders_and_reject_traversal(tmp_path):
    root = tmp_path / "models with spaces"
    repo, revision = "org/model", "a" * 40
    assert hub_directory(repo, revision, "step80", root=root) == root / "hub/org--model" / revision / "step80"
    assert hub_directory(repo, revision, "step640", root=root) != hub_directory(repo, revision, "step80", root=root)
    for bad in ["../model", "org/../model", "org/.."]:
        with pytest.raises(ValueError):
            hub_directory(bad, revision, root=root)
    with pytest.raises(ValueError):
        hub_directory(repo, "main", root=root)
    with pytest.raises(ValueError):
        hub_directory(repo, revision, "../step80", root=root)


def test_registry_and_history_projection_follow_explicit_workspace_and_model_root(tmp_path, monkeypatch):
    work = tmp_path / "shared workspace"
    assets = tmp_path / "physical assets"
    monkeypatch.setenv("MIRROROPD_WORKSPACE", str(work))
    monkeypatch.setenv("MODEL_ROOT", str(assets))
    pinned = "hub/org--model/" + "a" * 40
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            dict(
                models=[dict(id="org/model@sha", aliases=["model"], path="model/" + pinned)],
                locations={"model/old": "model/" + pinned, "model/old/nested": "model/hub/special"},
            )
        )
    )
    assert registered_model("model", registry_path=registry) == assets / pinned
    assert resolve_location(work / "model/old/config.json", registry_path=registry) == assets / pinned / "config.json"
    assert (
        resolve_location(work / "model/old/nested/config.json", registry_path=registry)
        == assets / "hub/special/config.json"
    )
    assert resolve_location(work / "other/config.json", registry_path=registry) == work / "other/config.json"
    assert (
        checkpoint_path(work / "wandb-logs/derived/MirrorOPD-verl/run", "checkpoints", "arm")
        == assets / "checkpoints/run/checkpoints/arm"
    )
    assert checkpoint_path(tmp_path / "custom", "checkpoints") == tmp_path / "custom/checkpoints"


def test_aliases_do_not_guess_between_different_weight_variants(tmp_path):
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            dict(
                models=[
                    dict(id="a", aliases=["ambiguous"], path="model/hub/a"),
                    dict(id="b", aliases=["ambiguous"], path="model/hub/b"),
                ]
            )
        )
    )
    with pytest.raises(ValueError, match="exactly one"):
        registered_model("ambiguous", registry_path=registry)


def test_production_checkpoints_keep_logs_separate_and_custom_runs_unique(tmp_path):
    from examples.mirroropd.assets.paths import production_checkpoints

    repository = tmp_path / "checkout"
    assets = tmp_path / "assets"
    path = production_checkpoints(repository / "outputs/run", root=assets, repository=repository)
    assert path == assets / "checkpoints/production/run/checkpoints"
    assert not path.exists()
    left = production_checkpoints(tmp_path / "left/run", root=assets, repository=repository)
    right = production_checkpoints(tmp_path / "right/run", root=assets, repository=repository)
    assert left != right and left.parent.name.startswith("run-")
    assert not left.exists() and not right.exists()


def test_repo_relative_historical_outputs_are_projected_without_mutating_record(tmp_path, monkeypatch):
    from examples.mirroropd.assets import paths

    work = tmp_path / "work"
    monkeypatch.setenv("MIRROROPD_WORKSPACE", str(work))
    monkeypatch.delenv("MODEL_ROOT", raising=False)
    locations = {"MirrorOPD-verl/outputs/run/warm-checkpoint": "model/checkpoints/production/run/warm-checkpoint"}
    monkeypatch.setattr(paths, "registry", lambda **kwargs: dict(locations=locations))
    record = {"weights": "outputs/run/warm-checkpoint"}
    assert resolve_location(record["weights"]) == work / "model/checkpoints/production/run/warm-checkpoint"
    assert record == {"weights": "outputs/run/warm-checkpoint"}
    assert checkpoint_path("outputs/run", "warm-checkpoint") == resolve_location(record["weights"])
    assert checkpoint_path("custom-run", "weights") == __import__("pathlib").Path("custom-run/weights")


def test_asset_names_use_registered_identity_instead_of_revision_basename(tmp_path, monkeypatch):
    from examples.mirroropd.assets import paths

    work = tmp_path / "work"
    monkeypatch.setenv("MIRROROPD_WORKSPACE", str(work))
    monkeypatch.delenv("MODEL_ROOT", raising=False)
    revision = "a" * 40
    relative = f"model/hub/org--teacher/{revision}/step80"
    entry = dict(
        id="teacher-step80", aliases=[], repo="org/teacher", revision=revision, subfolder="step80", path=relative
    )
    monkeypatch.setattr(paths, "registry", lambda **kwargs: dict(models=[entry], locations={"model/old": relative}))
    assert paths.asset_name(work / relative) == "teacher / step80"
    assert paths.asset_name(work / "model/old") == "teacher / step80"
    assert paths.asset_name("/custom/weight-view") == "weight-view"
