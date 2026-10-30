# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Canonical model assets and explicit historical-location projection.

New configurations use canonical paths. resolve_location is reserved for reading
immutable historical records whose original paths remain part of their identity.
No operation probes an old model directory or creates a compatibility link.
"""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

REGISTRY = Path(__file__).with_name("model_registry.json")


def workspace():
    return Path(os.environ.get("MIRROROPD_WORKSPACE", Path(__file__).resolve().parents[4])).resolve()


def model_root():
    return Path(os.environ.get("MODEL_ROOT", workspace() / "model")).expanduser().resolve()


def hub_directory(repo, revision, subfolder=None, *, root=None):
    parts = repo.split("/")
    if len(parts) != 2 or any(not re.fullmatch(r"[A-Za-z0-9_.-]+", p) or p in {".", ".."} for p in parts):
        raise ValueError("Model repository must be an explicit organization/name")
    if not re.fullmatch(r"[a-f0-9]{40}|content-[a-f0-9]{64}", revision):
        raise ValueError("Model revision must be an immutable commit or declared content identity")
    path = Path(root).expanduser() if root is not None else model_root()
    path = path / "hub" / "--".join(parts) / revision
    if subfolder is not None:
        relative = Path(subfolder)
        if not subfolder or relative.is_absolute() or ".." in relative.parts or "\\" in subfolder:
            raise ValueError("Model subfolder must be a safe relative path")
        path /= relative
    return path


def registry(*, path=None):
    return json.loads(Path(path or REGISTRY).read_text())


def model_entry(alias, *, registry_path=None):
    """Return the unique registered asset for an explicit alias."""
    matches = [
        entry
        for entry in registry(path=registry_path)["models"]
        if alias in [entry.get("id"), *entry.get("aliases", [])]
    ]
    if len(matches) != 1:
        raise ValueError(f"Model alias must identify exactly one asset: {alias!r} ({len(matches)} matches)")
    return matches[0]


def _relative_asset_path(entry):
    path = Path(entry["path"])
    if path.is_absolute():
        path = path.relative_to(workspace() / "model")
    elif path.parts[:1] == ("model",):
        path = Path(*path.parts[1:])
    return path


def registered_model(alias, *, root=None, registry_path=None):
    """Resolve one explicit registered alias; ambiguous short names are errors."""
    relative = _relative_asset_path(model_entry(alias, registry_path=registry_path))
    return (Path(root).expanduser() if root is not None else model_root()) / relative


def asset_name(path):
    """Read a model's human name from its identity rather than a revision directory."""
    actual = resolve_location(path)
    actual = actual if actual.is_absolute() else workspace() / actual
    roots = {model_root(), workspace() / "model"}
    for entry in registry()["models"]:
        variants = {root / _relative_asset_path(entry) for root in roots}
        if actual in variants:
            return entry["repo"].split("/")[-1] + (" / " + entry["subfolder"] if entry.get("subfolder") else "")
    return Path(path).name


def resolve_location(path, *, registry_path=None):
    """Project a recorded workspace path onto its current location without changing records."""
    original = Path(path).expanduser()
    absolute = original if original.is_absolute() else workspace() / original
    candidates = [absolute]
    if not original.is_absolute() and original.parts[:1] == ("outputs",):
        candidates.append(workspace() / "MirrorOPD-verl" / original)
    locations = registry(path=registry_path).get("locations", {})
    for before, after in sorted(locations.items(), key=lambda item: len(Path(item[0]).parts), reverse=True):
        old = Path(before)
        old = old if old.is_absolute() else workspace() / old
        suffix = next((candidate.relative_to(old) for candidate in candidates if candidate.is_relative_to(old)), None)
        if suffix is None:
            continue
        new = Path(after)
        if new.is_absolute():
            try:
                new = model_root() / new.relative_to(workspace() / "model")
            except ValueError:
                pass
        elif new.parts[:1] == ("model",):
            new = model_root() / Path(*new.parts[1:])
        else:
            new = workspace() / new
        return new / suffix
    return original


def require_current_location(path):
    original = Path(path)
    projected = resolve_location(original)
    if projected != original and projected != workspace() / original:
        raise ValueError(f"New configuration uses a retired model path: {original}; use {projected}")
    return original


def checkpoint_path(cache, *parts):
    """Place a derived experiment's exported model under the shared checkpoint root."""
    cache = Path(cache)
    candidate = cache.joinpath(*parts)
    projected = resolve_location(candidate)
    if projected != candidate:
        return projected
    derived = workspace() / "wandb-logs/derived/MirrorOPD-verl"
    try:
        relative = (cache if cache.is_absolute() else workspace() / cache).relative_to(derived)
    except ValueError:
        # Explicit test/custom storage remains explicit, without any legacy fallback.
        return cache.joinpath(*parts)
    return model_root().joinpath("checkpoints", relative, *parts)


def production_checkpoints(run_directory, *, root=None, repository=None):
    """Map run outputs to isolated model storage, including custom output roots."""
    run = Path(run_directory).expanduser().resolve()
    repo = Path(repository).resolve() if repository is not None else Path(__file__).resolve().parents[3]
    try:
        relative = run.relative_to(repo / "outputs")
    except ValueError:
        digest = hashlib.sha256(str(run).encode()).hexdigest()[:16]
        relative = Path("custom") / f"{run.name}-{digest}"
    return (
        (Path(root).expanduser() if root is not None else model_root())
        / "checkpoints/production"
        / relative
        / "checkpoints"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--alias")
    selection.add_argument("--run-dir", type=Path)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--model-root", type=Path)
    args = parser.parse_args()
    print(
        registered_model(args.alias, root=args.model_root)
        if args.alias
        else production_checkpoints(args.run_dir, root=args.model_root, repository=args.repo_root)
    )


if __name__ == "__main__":
    main()
