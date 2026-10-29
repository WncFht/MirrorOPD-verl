# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Frozen distributions around completed answers that subsequently repeat."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from verl.trainer.distillation.full_vocab import project


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--native-prompt", action="store_true", help="Frozen teacher check without the empty think block"
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cuda:0",
            local_files_only=True,
        )
        .eval()
        .requires_grad_(False)
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    for module in model.modules():
        if hasattr(module, "inv_freq"):
            assert module.inv_freq.dtype == torch.float32
    records, log_probs = [], []
    for case in json.loads(args.cases.read_text()):
        positions = [p["position"] for p in case["points"]]
        prompt = case["prompt_ids"]
        if args.native_prompt:
            suffix = tokenizer.encode("<think>\n\n</think>\n\n", add_special_tokens=False)
            assert prompt[-len(suffix) :] == suffix
            prompt = prompt[: -len(suffix)]
        tokens = prompt + case["response_ids"][: max(positions)]
        indexes = [len(prompt) - 1 + p for p in positions]
        with torch.inference_mode():
            inputs = torch.tensor([tokens], device="cuda:0")
            hidden = model.model(input_ids=inputs, use_cache=False).last_hidden_state[0, indexes]
            lp = project(hidden, model.get_output_embeddings().weight).double().log_softmax(-1)
            prob = lp.exp()
            for i, point in enumerate(case["points"]):
                values, ids = prob[i].topk(5)
                records.append(
                    point
                    | {
                        "case_id": case["case_id"],
                        "model": args.label,
                        "empty_think_removed": args.native_prompt,
                        "stop_probability": float(prob[i, [151643, 151645]].sum()),
                        "top_tokens": [tokenizer.decode([idx]) for idx in ids.cpu().tolist()],
                        "top_probabilities": values.cpu().tolist(),
                    }
                )
            log_probs.append(lp.float().cpu())
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"{args.label}.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    torch.save(torch.cat(log_probs), args.output / f"{args.label}-logprobs.pt")
    print(f"Scored {len(records)} answer-boundary states with {args.label}", flush=True)


if __name__ == "__main__":
    main()
