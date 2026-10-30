# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Content-address the local model assets without inventing Hub revisions."""

import argparse
from pathlib import Path

from examples.mirroropd.research.core.artifacts import file_hash, fingerprint, receipt
from examples.mirroropd.research.core.io import dump, read
from examples.mirroropd.research.studies.prefix_fusion.protocol import MODELS, protocol_identity


def prepare_assets(workspace, out):
    from transformers import AutoTokenizer

    workspace, out = Path(workspace).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    models, common_vocab = {}, None
    for key, name in MODELS.items():
        source = workspace / "model" / name
        directory = out / key
        directory.mkdir()
        files, weights = {}, []
        for path in sorted(source.iterdir()):
            if not path.is_file() or path.name.startswith(".") or path.name == "complete.json":
                continue
            if path.suffix not in {".safetensors", ".json", ".txt", ".jinja", ".model"}:
                continue
            before = path.stat()
            digest = file_hash(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError(f"Model asset changed while hashing: {path}")
            files[path.name] = dict(sha256=digest, bytes=before.st_size, mtime_ns=before.st_mtime_ns)
            (directory / path.name).symlink_to(path.resolve())
            if path.suffix == ".safetensors":
                weights.append(dict(name=path.name, bytes=before.st_size, sha256=digest))
        if not weights:
            raise ValueError(f"No safetensors checkpoint: {source}")
        config = read(directory / "config.json")
        generation = read(directory / "generation_config.json")
        if config["eos_token_id"] != 151643 or generation["eos_token_id"] != 151643:
            raise ValueError("This study requires a shared explicit R1 EOS")
        tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        vocabulary = tokenizer.get_vocab()
        if common_vocab is None:
            common_vocab = vocabulary
        elif common_vocab != vocabulary:
            raise ValueError("Study models have different token-ID mappings")
        legal = sorted(set(vocabulary.values()))
        if legal[-1] >= config["vocab_size"]:
            raise ValueError("Legal token IDs exceed a native head")
        identity = fingerprint({name: value["sha256"] for name, value in files.items()})
        model = dict(
            source=str(source.resolve()),
            directory=str(directory),
            identity=identity,
            files=files,
            weights=weights,
            vocab_size=config["vocab_size"],
            model_type=config["model_type"],
        )
        dump(directory / "complete.json", model)
        models[key] = model
        print("MODEL_IDENTITY", key, identity, flush=True)
    dump(out / "legal-token-ids.json", sorted(set(common_vocab.values())))
    receipt(
        out,
        dict(protocol=protocol_identity()),
        models=models,
        legal_ids_sha256=file_hash(out / "legal-token-ids.json"),
        scope="local file-content identities; no claimed Hub revision",
    )


def verify_model_assets(model):
    """Reject changed symlink targets before a GPU job loads a model."""
    directory = Path(model["directory"])
    for name, expected in model["files"].items():
        path = directory / name
        stat = path.stat()
        if stat.st_size != expected["bytes"] or file_hash(path) != expected["sha256"]:
            raise ValueError(f"Frozen model identity changed: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare_assets(args.workspace, args.out)


if __name__ == "__main__":
    main()
