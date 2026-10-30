# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Freeze a small training workload with all five validation sources."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--student", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--train-rows", type=int, default=16)
    args = parser.parse_args()
    if args.train_rows < 1 or args.max_prompt_length < 1:
        parser.error("Training row count and prompt cap must be positive")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.student, local_files_only=True)
    manifest = {
        "student": args.student,
        "max_prompt_length": args.max_prompt_length,
        "train_rows": args.train_rows,
        "sources": {},
    }
    for dataset in ["DAPO", "MATH-500", "AMC23", "AIME24", "AIME25", "AIME26"]:
        split = "train" if dataset == "DAPO" else "test"
        source = Path(args.data_root) / "math" / dataset / f"{split}.parquet"
        frame = pd.read_parquet(source)
        indices, lengths = [], []
        target_rows = args.train_rows if split == "train" else 2
        for index, row in frame.iterrows():
            prompt = list(row["prompt"])
            ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_dict=False)
            if len(ids) <= args.max_prompt_length:
                indices.append(index)
                lengths.append(len(ids))
            if len(indices) == target_rows:
                break
        if len(indices) != target_rows:
            raise ValueError(f"Insufficient fitting prompts in {source}")
        target = out / f"{dataset}.parquet"
        frame.loc[indices].to_parquet(target, index=False)
        manifest["sources"][dataset] = {
            "path": str(source.resolve()),
            "row_indices": indices,
            "prompt_lengths": lengths,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
