# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Pinned stage assets: preserve checkpoint subfolders and download only selected weight manifests."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from examples.mirroropd.research.core.artifacts import complete, file_hash, receipt
from examples.mirroropd.research.core.io import dump, read
from examples.mirroropd.research.studies.fixed_transfer.spec import Model


def selected_files(model, manifest):
    if (manifest["repo"], manifest["revision"]) != (model.repo, model.revision):
        raise ValueError("Manifest is not the pinned model revision")
    prefix = model.subfolder + "/" if model.subfolder else ""
    result = []
    for entry in manifest["files"]:
        path = entry["path"]
        if not path.startswith(prefix):
            continue
        name = path[len(prefix) :]
        if not name or "/" in name:
            continue
        if name.endswith((".safetensors", ".json", ".jinja", ".txt", ".model", ".md")) or name == "LICENSE":
            result.append(dict(entry, name=name))
    weights = [f for f in result if f["name"].endswith(".safetensors")]
    if not weights or any(not f["sha256"] for f in weights):
        raise ValueError("Pinned safetensors and content hashes are required")
    if "config.json" not in {f["name"] for f in result}:
        raise ValueError("Checkpoint configuration is missing")
    return result


def materialize(model, manifest, metadata_only=False):
    from huggingface_hub import hf_hub_download

    inputs = dict(repo=model.repo, revision=model.revision)
    if model.subfolder is not None:
        inputs["subfolder"] = model.subfolder
    target = model.directory
    if complete(target, inputs):
        return
    files = selected_files(model, manifest)
    root = target
    if model.subfolder:
        for _ in Path(model.subfolder).parts:
            root = root.parent
        if root / model.subfolder != target:
            raise ValueError("Local directory must preserve the declared subfolder")
    target.mkdir(parents=True, exist_ok=True)
    dump(target / "pinned-files.json", dict(inputs=inputs, files=files))
    for entry in files:
        if metadata_only and entry["name"].endswith(".safetensors"):
            continue
        path = Path(hf_hub_download(model.repo, entry["path"], revision=model.revision, local_dir=root, token=False))
        if entry["name"].endswith(".safetensors"):
            if path.stat().st_size != entry["size"] or file_hash(path) != entry["sha256"]:
                raise ValueError(f"Weight integrity failure: {path}")
    if metadata_only:
        dump(target / "metadata-ready.json", inputs)
        return
    index = target / "model.safetensors.index.json"
    if index.exists():
        assert all((target / f).exists() for f in set(read(index)["weight_map"].values()))
    receipt(
        target,
        inputs,
        weights=[
            dict(name=f["name"], bytes=f["size"], sha256=f["sha256"])
            for f in files
            if f["name"].endswith(".safetensors")
        ],
    )
    print("ASSET", model.name, model.revision, model.subfolder, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, required=True)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    models = [Model(**v) for v in read(args.registry)["models"].values()]
    manifests = {x["repo"]: x for x in read(args.manifests)}
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda m: materialize(m, manifests[m.repo], args.metadata_only), models))


def assets(study, workers=3):
    """Use frozen subfolder manifests without falling back to repository-root weights."""
    from examples.mirroropd.research.studies.fixed_transfer.models import materialize as root_materialize

    def ensure(model):
        if model.subfolder is None:
            return root_materialize(model)
        path = model.directory / "pinned-files.json"
        if not path.exists():
            raise ValueError(f"Freeze a subfolder manifest with stage_assets before loading {model.name}")
        frozen = read(path)
        expected = dict(repo=model.repo, revision=model.revision, subfolder=model.subfolder)
        if frozen["inputs"] != expected:
            raise ValueError(f"Subfolder asset identity mismatch for {model.name}")
        manifest = dict(repo=model.repo, revision=model.revision, files=frozen["files"])
        selected_files(model, manifest)
        materialize(model, manifest)
        return read(model.directory / "complete.json")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        values = list(pool.map(ensure, study.models.values()))
    dump(study.directory / "assets.json", dict(zip(study.models, values, strict=True)))


if __name__ == "__main__":
    main()
