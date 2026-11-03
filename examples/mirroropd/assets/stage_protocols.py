# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Explicit SFT-tokenizer views of unchanged published stage-chain weights."""

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from .paths import model_root, registered_model

PUBLISHED = {
    "Olmo-3-1025-7B": ("allenai/Olmo-3-1025-7B", "a81bae42db3975be1671e27b9c9a56da1a9f980f"),
    "Olmo3-IVON-SFT-7B": ("BayesRL/Olmo3-IVON-SFT-7B", "4022ca655a865d859ddb8ecb92f008cb130a5542"),
    "Olmo3-C3PO-7B": ("BayesRL/Olmo3-C3PO-7B", "c223ee93e7344487f74fdd5767a42318ca6525af"),
    "Qwen2.5-14B": ("Qwen/Qwen2.5-14B", "97e1e76335b7017d8f67c08a19d103c0504298c9"),
    "DeepSeek-R1-Distill-Qwen-14B": (
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "1df8507178afcc1bef68cd8c393f61a886323761",
    ),
    "Light-R1-14B-DS": ("qihoo360/Light-R1-14B-DS", "a69e32b88daa855ae068ca8601f3ec4085e96086"),
}
VIEWS = {
    "Olmo3-Base-BayesProtocol": ("Olmo-3-1025-7B", "Olmo3-IVON-SFT-7B", ()),
    "Olmo3-SFT-BayesProtocol": ("Olmo3-IVON-SFT-7B", "Olmo3-IVON-SFT-7B", ()),
    "Olmo3-C3PO-BayesProtocol": ("Olmo3-C3PO-7B", "Olmo3-IVON-SFT-7B", ()),
    "Qwen2.5-14B-R1Protocol": ("Qwen2.5-14B", "DeepSeek-R1-Distill-Qwen-14B", tuple(range(151643, 151650))),
    "R1-Distill-14B-R1Protocol": ("DeepSeek-R1-Distill-Qwen-14B", "DeepSeek-R1-Distill-Qwen-14B", ()),
    "Light-R1-14B-R1Protocol": ("Light-R1-14B-DS", "DeepSeek-R1-Distill-Qwen-14B", ()),
}
TOKENIZER_FILES = {
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "special_tokens_map.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
}


def specification(name):
    source, tokenizer, allowed = VIEWS[name]
    return {
        "schema": 1,
        "kind": "shared_sft_tokenizer",
        "weights": PUBLISHED[source],
        "tokenizer": PUBLISHED[tokenizer],
        "allowed_renamed_ids": list(allowed),
        "weight_edit": "none",
        "tokenizer_length_edit": "none",
        "normalize_special_id_metadata": True,
    }


def revision(name):
    return "content-" + hashlib.sha256(json.dumps(specification(name), sort_keys=True).encode()).hexdigest()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def vocabulary(document):
    values = dict(document["model"]["vocab"])
    values.update({token["content"]: token["id"] for token in document.get("added_tokens", [])})
    return {value: token for token, value in values.items()}


def verify_tokenizer_change(source, target, allowed):
    """Allow only the declared ID-label changes, preserving all core segmentation."""
    from tokenizers import Tokenizer

    # Old Hub files encode BPE merges as strings, newer files as pairs and
    # materialize default fields. Compare their parsed definitions.
    source = json.loads(Tokenizer.from_str(json.dumps(source)).to_str())
    target = json.loads(Tokenizer.from_str(json.dumps(target)).to_str())
    left, right = vocabulary(source), vocabulary(target)
    if set(left) != set(right):
        raise ValueError("Stage protocols require identical defined ID support")
    renamed = [i for i in sorted(left) if left[i] != right[i]]
    if renamed != list(allowed):
        raise ValueError(f"Undeclared tokenizer relabeling: {renamed}")
    definitions = []
    for document in (source, target):
        definition = json.loads(json.dumps(document))
        for field in ("padding", "truncation", "post_processor", "added_tokens"):
            definition.pop(field, None)
        definition["model"].pop("vocab")
        definitions.append(definition)
    if definitions[0] != definitions[1]:
        raise ValueError("Stage protocol changes segmentation, normalization, or decoding")
    old_added = {t["id"]: t for t in source.get("added_tokens", [])}
    new_added = {t["id"]: t for t in target.get("added_tokens", [])}
    if any(old_added.get(i) != new_added.get(i) for i in set(old_added) | set(new_added) if i not in allowed):
        raise ValueError("Stage protocol changes undeclared added-token semantics")
    return [{"id": i, "published": left[i], "effective": right[i]} for i in renamed]


def prepare(name, *, root=None):
    """Publish a new asset atomically; original weights and metadata remain untouched."""
    root = Path(root or model_root())
    source_name, tokenizer_name, allowed = VIEWS[name]
    source = registered_model(source_name, root=root)
    tokenizer = registered_model(tokenizer_name, root=root)
    destination = registered_model(name, root=root)
    if destination.exists():
        receipt = json.loads((destination / "stage-protocol.json").read_text())
        if receipt["specification"] != json.loads(json.dumps(specification(name))):
            raise ValueError("Existing stage protocol has another specification")
        if any(digest(destination / file) != value for file, value in receipt["effective_metadata_sha256"].items()):
            raise ValueError("Existing stage protocol metadata changed")
        for file, weight in receipt["weights"].items():
            path = destination / file
            if not path.is_file() or str(path.resolve()) != weight["source"] or path.stat().st_size != weight["bytes"]:
                raise ValueError("Existing stage protocol weight binding changed")
        return destination
    source_definition = json.loads((source / "tokenizer.json").read_text())
    target_definition = json.loads((tokenizer / "tokenizer.json").read_text())
    relabels = verify_tokenizer_change(source_definition, target_definition, allowed)
    target_ids = {token: i for i, token in vocabulary(target_definition).items()}
    tokenizer_config = json.loads((tokenizer / "tokenizer_config.json").read_text())
    special = {}
    for field in ("bos_token", "eos_token", "pad_token"):
        token = tokenizer_config.get(field)
        if isinstance(token, dict):
            token = token["content"]
        special[field + "_id"] = target_ids[token] if token is not None else None
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".stage-protocol-", dir=root) as temporary:
        staged = Path(temporary) / "model"
        staged.mkdir()
        for path in source.iterdir():
            if path.is_file() and path.name not in TOKENIZER_FILES:
                if path.suffix == ".safetensors":
                    os.symlink(path.resolve(), staged / path.name)
                else:
                    shutil.copy2(path, staged / path.name)
        for file in TOKENIZER_FILES:
            if (tokenizer / file).is_file():
                shutil.copy2(tokenizer / file, staged / file)
        config_edits = {}
        for file in ("config.json", "generation_config.json"):
            path = staged / file
            config = json.loads(path.read_text()) if path.exists() else {}
            config_edits[file] = {
                k: {"published": config.get(k), "effective": v} for k, v in special.items() if config.get(k) != v
            }
            config.update(special)
            path.write_text(json.dumps(config, indent=2) + "\n")
        if not (staged / "chat_template.jinja").is_file() and not tokenizer_config.get("chat_template"):
            raise ValueError("Stage protocol requires a published SFT chat template")
        metadata = {
            path.name: digest(path) for path in staged.iterdir() if path.is_file() and path.suffix != ".safetensors"
        }
        receipt = {
            "specification": specification(name),
            "renamed_ids": relabels,
            "config_edits": config_edits,
            "weights": {
                path.name: {"source": str(path.resolve()), "bytes": path.stat().st_size}
                for path in staged.glob("*.safetensors")
            },
            "published_config_sha256": digest(source / "config.json"),
            "effective_metadata_sha256": metadata,
        }
        (staged / "stage-protocol.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
        destination.parent.mkdir(parents=True, exist_ok=True)
        staged.replace(destination)
    return destination


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=[*VIEWS, "all"], default=["all"])
    parser.add_argument("--model-root", type=Path)
    args = parser.parse_args()
    for name in VIEWS if args.models == ["all"] else args.models:
        print(prepare(name, root=args.model_root))


if __name__ == "__main__":
    main()
