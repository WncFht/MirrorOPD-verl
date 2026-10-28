# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Score identical saved E54 token prefixes across initial/trained checkpoints.

Run after the training GPUs are released. Keeps original token IDs and template;
never generates, truncates the selected prefix, or updates model parameters.
"""

import argparse
import gc
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from verl.trainer.distillation.full_vocab import project


def cases(debug):
    result = []
    for outer in (1, 3):
        for line in (debug / f"rollout-{outer}.jsonl").read_text().splitlines():
            row = json.loads(line)
            end = row["length"] - 1
            positions = [end] if row["stopped_on_eos"] else sorted({min(end, p) for p in (512, 2048, 4096, end)})
            result.append({**row, "positions": positions})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True, help="label=/absolute/model/path")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = cases(args.debug)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        for entry in args.model:
            label, model_path = entry.split("=", 1)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="cuda:0",
                local_files_only=True,
            ).eval()
            model.requires_grad_(False)
            for module in model.modules():
                if hasattr(module, "inv_freq"):
                    assert module.inv_freq.dtype == torch.float32
            with torch.inference_mode():
                for case in rows:
                    positions = case["positions"]
                    tokens = case["prompt_ids"] + case["response_ids"][: max(positions)]
                    indexes = [len(case["prompt_ids"]) - 1 + p for p in positions]
                    hidden = model.model(input_ids=torch.tensor([tokens], device="cuda:0"), use_cache=False)
                    h = hidden.last_hidden_state[0, indexes]
                    lp = project(h, model.get_output_embeddings().weight).double().log_softmax(-1)
                    prob = lp[:, [151643, 151645]].exp().cpu().tolist()
                    top = lp.argmax(-1).cpu().tolist()
                    for j, pos in enumerate(positions):
                        record = {
                            "checkpoint": label,
                            "outer_source": case["outer"],
                            "sample_key": case["sample_key"],
                            "position": pos,
                            "response_length": case["length"],
                            "source_ended_on_eos": case["stopped_on_eos"],
                            "student_stop": sum(prob[j]),
                            "student_eot": prob[j][0],
                            "student_im_end": prob[j][1],
                            "top1": top[j],
                        }
                        output.write(json.dumps(record) + "\n")
                    output.flush()
                    del hidden, h, lp
            del model
            gc.collect()
            torch.cuda.empty_cache()
            print(f"Scored {len(rows)} fixed trajectories with {label}", flush=True)


if __name__ == "__main__":
    main()
