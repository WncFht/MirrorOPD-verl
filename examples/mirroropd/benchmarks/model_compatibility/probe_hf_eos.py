#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Check standard EOS config in real HF generation and checkpoint metadata readback."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa"
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is 1+1?"}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
        enable_thinking=False,
    )
    x = torch.tensor([ids], device="cuda:0")
    records = []
    for label, config in [
        ("native", GenerationConfig.from_pretrained(args.source)),
        ("standard_config", model.generation_config),
    ]:
        for target in [151643, 151645]:
            with torch.inference_mode():
                y = model.generate(
                    x,
                    attention_mask=torch.ones_like(x),
                    generation_config=config,
                    max_new_tokens=4,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    prefix_allowed_tokens_fn=lambda batch_id, prefix, target=target: [target],
                )
            records.append({"policy": label, "forced_id": target, "output_ids": y[0, len(ids) :].tolist()})
    # Match the metadata source used by the existing FSDP checkpoint saver.
    export = args.output.parent / "checkpoint-generation"
    GenerationConfig.from_pretrained(model.config.name_or_path).save_pretrained(export)
    result = {
        "records": records,
        "model_name_or_path": model.config.name_or_path,
        "checkpoint_eos_readback": GenerationConfig.from_pretrained(export).eos_token_id,
        "source_generation_eos_unchanged": GenerationConfig.from_pretrained(args.source).eos_token_id,
        "tokenizer_vocab_equal": tokenizer.get_vocab() == AutoTokenizer.from_pretrained(args.source).get_vocab(),
        "all_weight_links_resolve_to_source": all(
            p.resolve().parent == args.source.resolve() for p in args.model.glob("*.safetensors")
        ),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
