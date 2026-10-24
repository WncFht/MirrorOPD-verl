#!/usr/bin/env python3
"""Prepare an experiment model directory with shared weights and explicit EOS.

Only generation_config.json is written. All other assets refer to the original
model. Transformers, vLLM and existing checkpoint export can use the same path.
"""

import argparse
import json
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer, GenerationConfig


def prepare(source: Path, output: Path, additional_tokens: list[str]):
    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    tokenizer = AutoTokenizer.from_pretrained(source)
    config = AutoConfig.from_pretrained(source)
    generation = GenerationConfig.from_pretrained(source)
    native = generation.eos_token_id
    ids = set(native if isinstance(native, list) else [native] if native is not None else [])
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    vocab = tokenizer.get_vocab()
    for token in additional_tokens:
        if token not in vocab:
            raise ValueError(f"Additional EOS must be an existing single token: {token!r}")
        ids.add(vocab[token])
    if not ids or any(type(i) is not int or not 0 <= i < config.vocab_size for i in ids):
        raise ValueError("EOS IDs must be nonempty and within the model output vocabulary")
    generation.eos_token_id = sorted(ids)
    output.mkdir(parents=True, exist_ok=False)
    for item in source.iterdir():
        if not item.name.startswith(".") and item.name != "generation_config.json":
            (output / item.name).symlink_to(item, target_is_directory=item.is_dir())
    generation.save_pretrained(output)
    return {"source": str(source), "model_path": str(output), "effective_eos_token_ids": sorted(ids)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--additional-eos-token", action="append", default=[])
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output, args.additional_eos_token), indent=2))


if __name__ == "__main__":
    main()
