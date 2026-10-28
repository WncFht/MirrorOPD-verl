# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Functional retention on 64 fixed single-turn IF prompts; no training."""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..common import native as core
from . import filter_common as fc


def main():
    ap = argparse.ArgumentParser()
    for key in ["models", "out", "cache"]:
        ap.add_argument("--" + key, type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("highest")
    prompts = [r for r in core.read(a.out / "test-bank.json") if r["domain"] == "if"]
    assert len(prompts) == 64
    target = a.out / "if-eval" / (a.name + ".json")
    rows = core.read(target) if target.exists() else []
    assert len(rows) % 16 == 0
    if len(rows) == 64:
        return
    if a.model == "checkpoint":
        model_path = a.cache / "checkpoints" / a.name / "step-128"
        model = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch.float32,
                attn_implementation="flash_attention_2",
                local_files_only=True,
            )
            .cuda()
            .eval()
            .requires_grad_(False)
        )
    else:
        model = core.load(a, a.model, student=a.model == "student").requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(a.models / core.NAMES["student"], local_files_only=True)
    controls = set(tokenizer.all_special_ids) - set(tokenizer.convert_tokens_to_ids(["<think>", "</think>"]))
    for start in range(len(rows), 64, 16):
        generated = fc.generate(a, model, prompts[start : start + 16], 99700 + start, 4096, vocab_limit=151936)
        for row in generated:
            row["clean_output"] = tokenizer.decode(
                [t for t in row["response_ids"] if t not in controls],
                skip_special_tokens=False,
            )
            row["model_name"] = a.name
        rows += generated
        core.dump(target, rows)
        print("IF", a.name, len(rows), flush=True)


if __name__ == "__main__":
    main()
