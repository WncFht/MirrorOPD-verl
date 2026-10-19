# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Materialize the two pinned math models into the shared model/ directory."""

import argparse
import shutil
import tempfile
from pathlib import Path

from check_assets import MODEL_FILES, MODEL_ROOT, MODELS, check_model, file_sha256, write_json


def main():
    """Copy complete local snapshots or explicitly download fixed Hub revisions, then verify hashes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-root", type=Path, help="existing root containing both named model directories")
    source.add_argument("--download", action="store_true", help="download pinned revisions from Hugging Face")
    args = parser.parse_args()
    if args.source_root is not None:
        args.source_root = args.source_root.expanduser().resolve()
    root = args.model_root.expanduser().resolve()
    for name in MODELS:
        if (root / name).exists():
            raise FileExistsError(f"{root / name} exists; use check_assets.py to verify existing models")
    root.mkdir(parents=True, exist_ok=True)
    reports = {}
    with tempfile.TemporaryDirectory(prefix=".prepare-models-", dir=root) as temp:
        staged = Path(temp)
        for name, spec in MODELS.items():
            destination = staged / name
            if args.download:
                from huggingface_hub import snapshot_download

                source_folder = Path(
                    snapshot_download(
                        repo_id=spec["repo"],
                        revision=spec["revision"],
                        allow_patterns=[*MODEL_FILES, "README.md", "LICENSE*", "chat_template.jinja"],
                    )
                )
            else:
                source_folder = args.source_root / name
            destination.mkdir()
            for filename in MODEL_FILES:
                # Materialize snapshot symlinks so transfers do not depend on a separate HF cache.
                shutil.copy2(source_folder / filename, destination / filename)
            for source_file in [
                source_folder / "README.md",
                source_folder / "chat_template.jinja",
                *source_folder.glob("LICENSE*"),
            ]:
                if source_file.is_file():
                    shutil.copy2(source_file, destination / source_file.name)
            result, _ = check_model(destination, name, verify_sha256=True)
            result["path"] = name
            result["metadata_sha256"] = {
                filename: file_sha256(destination / filename)
                for filename in (*MODEL_FILES, "chat_template.jinja")
                if filename.endswith((".json", ".jinja")) and (destination / filename).is_file()
            }
            reports[name] = result
        for name in MODELS:
            (staged / name).replace(root / name)
    write_json(root / "model_manifest.json", {"mode": "download" if args.download else "import", "models": reports})
    print(f"Prepared pinned model snapshots: {root}")


if __name__ == "__main__":
    main()
