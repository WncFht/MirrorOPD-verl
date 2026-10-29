#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Compare local model configurations and pinned remote small metadata files.

No weight downloads, model loading, GPU usage, or modification of model assets.
Token alignment is checked by actual token-ID maps, not vocabulary size alone.
"""

import argparse
from pathlib import Path

import requests

from examples.mirroropd.research.core.io import dump, read

ARCH_KEYS = (
    "model_type",
    "architectures",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "vocab_size",
    "tie_word_embeddings",
    "rope_theta",
    "rope_scaling",
    "max_position_embeddings",
    "rms_norm_eps",
    "hidden_act",
    "use_sliding_window",
)
CORE_KEYS = ARCH_KEYS[2:8]


def token_map(tokenizer):
    """Extract a combined base and added-token ID mapping from tokenizer JSON."""
    base = tokenizer["model"]["vocab"]
    if not isinstance(base, dict):
        raise ValueError("This audit expects a BPE vocabulary dictionary")
    result = {int(index): token for token, index in base.items()}
    result.update({int(row["id"]): row["content"] for row in tokenizer.get("added_tokens", [])})
    return result


def compare(reference, candidate, reference_tokens, candidate_tokens):
    """Record explicit config and token mismatches without assuming compatibility."""
    differences = {
        k: {"student": reference.get(k), "teacher": candidate.get(k)}
        for k in ARCH_KEYS
        if reference.get(k) != candidate.get(k)
    }
    shared = sorted(set(reference_tokens) & set(candidate_tokens))
    mismatches = [
        {"id": i, "student": reference_tokens[i], "teacher": candidate_tokens[i]}
        for i in shared
        if reference_tokens[i] != candidate_tokens[i]
    ]
    return {
        "core_dimensions_equal": all(reference.get(k) == candidate.get(k) for k in CORE_KEYS),
        "checked_config_equal": not differences,
        "config_differences": differences,
        "shared_token_ids": len(shared),
        "shared_id_mismatch_count": len(mismatches),
        "shared_id_mismatch_examples": mismatches[:24],
        "student_only_token_ids": sorted(set(reference_tokens) - set(candidate_tokens))[:32],
        "teacher_only_token_ids": sorted(set(candidate_tokens) - set(reference_tokens))[:32],
        "scope": "Config fields and tokenizer JSON only; no tensor-level or runtime equivalence claim",
    }


def fetch_json(url):
    """Read a public metadata endpoint with an explicit timeout."""
    response = requests.get(url, timeout=40)
    response.raise_for_status()
    return response.json()


def main():
    """Audit existing local models first, then named remote candidates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    student = args.workspace / "model/DeepSeek-R1-Distill-Qwen-1.5B"
    cfg = read(student / "config.json")
    tokens = token_map(read(student / "tokenizer.json"))
    result = {
        "student": str(student.resolve()),
        "student_config": {k: cfg.get(k) for k in ARCH_KEYS},
        "local": [],
        "remote": [],
    }
    for name in [
        "model/JustRL-DeepSeek-1.5B",
        "models/DeepCoder-1.5B-Preview",
        "models/UltraData-IF-1.5B",
        "model/Polaris-7B-Preview",
    ]:
        path = args.workspace / name
        row = {"name": name, "present": path.exists()}
        if path.exists():
            other = read(path / "config.json")
            row.update(compare(cfg, other, tokens, token_map(read(path / "tokenizer.json"))))
        result["local"].append(row)
    for repo in ["nvidia/OpenMath-Nemotron-1.5B", "WeiboAI/VibeThinker-1.5B"]:
        metadata = fetch_json(f"https://huggingface.co/api/models/{repo}")
        revision = metadata["sha"]
        base = f"https://huggingface.co/{repo}/resolve/{revision}"
        destination = args.out / repo.replace("/", "--")
        destination.mkdir(exist_ok=True)
        files = {}
        for name in ["config.json", "tokenizer.json", "tokenizer_config.json"]:
            files[name] = fetch_json(f"{base}/{name}")
            if name != "tokenizer.json":
                dump(destination / name, files[name])
        response = requests.get(f"{base}/README.md", timeout=40)
        response.raise_for_status()
        (destination / "README.md").write_text(response.text)
        row = {
            "repo": repo,
            "revision": revision,
            "model_card": f"https://huggingface.co/{repo}/blob/{revision}/README.md",
            "config_url": f"https://huggingface.co/{repo}/blob/{revision}/config.json",
            "base_model": metadata.get("cardData", {}).get("base_model"),
            "config": {k: files["config.json"].get(k) for k in ARCH_KEYS},
        }
        row.update(compare(cfg, files["config.json"], tokens, token_map(files["tokenizer.json"])))
        result["remote"].append(row)
        print(
            repo,
            "core dimensions",
            row["core_dimensions_equal"],
            "config differences",
            list(row["config_differences"]),
            "token mismatches",
            row["shared_id_mismatch_count"],
            flush=True,
        )
    dump(args.out / "model-comparison.json", result)


if __name__ == "__main__":
    main()
