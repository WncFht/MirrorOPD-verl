# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Real-model hidden/head reconstruction audit, one GPU, no optimizer or rollout."""

import argparse
import gc
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from verl.models.transformers.full_vocab import install_full_vocab_adapter
from verl.trainer.distillation.hidden import FrozenTeacherHead


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    names = ["JustRL-DeepSeek-1.5B", "DeepCoder-1.5B-Preview", "UltraData-IF-1.5B"]
    texts = [
        "Compute 17 + 25 and explain your answer.",
        "def add(a, b):\n    return a + b",
        "Reply in exactly three words: describe the sky.",
    ]
    records = []
    for name in names:
        path = args.model_root / name
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = (
            AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
        )
        head = FrozenTeacherHead(str(path))
        install_full_vocab_adapter(model)
        for text in texts:
            ids = tokenizer(text, return_tensors="pt")["input_ids"].cuda()
            with torch.no_grad():
                original = model(ids, use_cache=False).logits
                hidden = model(ids, full_vocab_callback=lambda h, w: h)
                reconstructed = hidden @ head.get(hidden.device, hidden.dtype).T
                error = (original.float() - reconstructed.float()).abs().max().item()
                torch.testing.assert_close(original, reconstructed, atol=0, rtol=0)
            records.append({"teacher": name, "tokens": ids.numel(), "max_logit_error": error})
        head.release()
        del model, head, original, hidden, reconstructed, ids
        gc.collect()
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"dtype": "bfloat16", "attention": "sdpa", "records": records}, indent=2) + "\n")
    print(json.dumps(records))


if __name__ == "__main__":
    main()
