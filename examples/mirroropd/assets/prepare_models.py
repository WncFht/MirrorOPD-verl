# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Import or download pinned models into the canonical shared asset store."""

import argparse
import shutil
import tempfile
from pathlib import Path

if __package__:
    from .check_assets import DEFAULT_MODELS, MODEL_ROOT, MODELS, check_model, registered_model, selected_models
else:
    from check_assets import DEFAULT_MODELS, MODEL_ROOT, MODELS, check_model, registered_model, selected_models


def main():
    """Materialize local assets or download configured Hub models without model manifests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--models", nargs="+", choices=[*MODELS, "all"], default=list(DEFAULT_MODELS))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-root", type=Path, help="existing root containing the selected model directories")
    source.add_argument("--download", action="store_true", help="download the configured models from Hugging Face")
    args = parser.parse_args()
    if args.source_root is not None:
        args.source_root = args.source_root.expanduser().resolve()
    root = args.model_root.expanduser().resolve()
    pending = []
    for name in selected_models(args.models):
        if registered_model(name, root=root).exists():
            check_model(registered_model(name, root=root), name)
        else:
            pending.append(name)
    root.mkdir(parents=True, exist_ok=True)
    patterns = ["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "LICENSE*", "README.md"]
    with tempfile.TemporaryDirectory(prefix=".prepare-models-", dir=root) as temp:
        staged = Path(temp)
        for name in pending:
            spec = MODELS[name]
            destination = staged / name
            if args.download:
                from huggingface_hub import snapshot_download

                source_folder = Path(
                    snapshot_download(
                        repo_id=spec["repo"],
                        revision=spec["revision"],
                        allow_patterns=patterns,
                        local_dir=staged / (name + "-download"),
                    )
                )
            else:
                source_folder = registered_model(name, root=args.source_root)
            check_model(source_folder, name)
            destination.mkdir()
            files = {
                file
                for pattern in patterns
                for file in source_folder.rglob(pattern)
                if not {".cache", ".git"}.intersection(file.relative_to(source_folder).parts)
            }
            for source_file in sorted(files):
                if source_file.is_file():
                    target = destination / source_file.relative_to(source_folder)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_file, target)
            check_model(destination, name)
        for name in pending:
            target = registered_model(name, root=root)
            target.parent.mkdir(parents=True, exist_ok=True)
            (staged / name).replace(target)
    print(f"Prepared model directories: {root}")


if __name__ == "__main__":
    main()
